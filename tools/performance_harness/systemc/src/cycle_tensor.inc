//==============================================================================
// The cycle-level transfer engine (tnca_tensor.v) of the cycle core
// [CORE_TIMING.md 4.2]. Included inside `struct CycleCore`. State names follow
// the Verilog; every rule is "during cycle t from the pre-edge state, at the
// edge these registers change".
//==============================================================================
struct CycleTensor {
    CycleCore& k;
    enum St {IDLE,DLOAD,DWAIT,DPARSE,TSETUP,TCHK,TMUL,TRUN,TDRAIN,SRUN,SREAD,SDRAIN,DONE,FAULT,XRUN,XDRAIN,XWRITE};
    static constexpr unsigned NTAG=16,NSLOT=8;
    St st=IDLE;
    // tables
    struct Desc {std::shared_ptr<const TensorMapModel> snapshot;unsigned references=0;};
    struct Window {bool valid=false;unsigned base=0,length=0;};
    std::array<Desc,16> descriptors;std::array<Window,32> windows;
    // descriptor load
    unsigned dl_slot=0,dl_word=0;u64 dl_addr=0;std::array<unsigned char,256> dl_buf{};bool dl_waiting=false;u64 dl_rsp=0,dl_data=0;
    // the command
    struct Cmd {bool store=false,transpose=false,active=true,bar_en=false;unsigned desc=0,win=0,bytes=2,esz=1,rctx=0,role=0;u64 token=0,pc=0;std::array<u64,5> coords{};unsigned sb=0;} cmd;
    std::shared_ptr<const TensorMapModel> d;Window win;unsigned total=0,used_registers=0,base=0;u64 owner=0;unsigned fault_code=0;std::string fault_why;
    // the walk: rows enumerated in descriptor order; within a row, units (loads) or memory words (stores)
    unsigned mul_left=0;          // TMUL cycles left
    unsigned row=0,rows=0;        // row index over the outer dims (flat row number) and count
    unsigned t_unit=0;bool t_half=false;unsigned t_flat=0;
    // per-row geometry (recomputed when the row changes)
    struct RowBeat {u64 addr=0;unsigned be=0;u64 data=0;unsigned unit=0;bool high=false;};
    std::vector<RowBeat> row_words;  // store: memory words of the row in order; load: unused
    bool rowv=false;unsigned row_units=0,row_bytes=0,r_m=0;u64 row_addr=0;  // t_P3 of the row (element 0 address)
    // load slots
    struct Slot {bool v=false,nlo=false,nhi=false,glo=false,ghi=false;u64 dlo=0,dhi=0;unsigned vreg=0,lane=0,keep=0,m=0;u64 rsp_lo=0,rsp_hi=0;};
    std::array<Slot,NSLOT> sl;unsigned t_slot=0;
    // store / transposed tags
    struct Tag {bool v=false;u64 rsp=0;unsigned xrow=0,xkeep=0;u64 data=0;};
    std::array<Tag,NTAG> tg;
    // store register in hand
    bool sr_v=false;unsigned sr_idx=0;PhysicalRegisters::Vector sr_data{};u64 st_prev=0;bool sread_req=false,sread_pending=false;
    // window write presented (vw_*)
    bool vw_en=false;unsigned vw_idx=0,vw_mask=0;PhysicalRegisters::Vector vw_data{};
    // transposed
    std::array<u64,16> xbuf{};unsigned x_row=0,x_q=0;
    // done / fault pulses (cycle in which they are high)
    u64 done_at=0;bool done_bar=false;u64 done_token=0;bool done_err=false;u64 fault_at=0;
    bool busy_q=false;
    explicit CycleTensor(CycleCore& core):k(core){}
    bool busy()const{return st!=IDLE;}
    void reset(){st=IDLE;for(auto& s:sl)s=Slot{};for(auto& g:tg)g=Tag{};vw_en=false;sr_v=false;done_at=fault_at=0;
        for(auto& dd:descriptors){dd=Desc{};}
        for(auto& w:windows){w=Window{};}}
    //------------------------------------------------------------------ commands at X
    void config(unsigned function,unsigned slot,u64 address,unsigned immediate){
        arch((function<16 && slot<16) || (function>=16 && slot<32),4,"DESCRIPTOR_FAULT: configuration slot");
        if(function==0){require(address%8==0,"unaligned TensorMap snapshot address is not implemented");
            dl_slot=slot;dl_addr=address;dl_word=0;dl_waiting=false;st=DLOAD;cmd.bar_en=false;cmd.rctx=k.xb.rctx;}
        else if(function==2){arch(descriptors[slot].references==0,6,"RESOURCE_EXHAUSTED: descriptor still referenced");descriptors[slot].snapshot.reset();}
        else if(function==16){unsigned b=(immediate>>8)&255,l=immediate&255;arch(l && b<64 && l<=64-b,4,"DESCRIPTOR_FAULT: window exceeds role register mapping");windows[slot]={true,b,l};}
        else if(function==17)windows[slot]=Window{};
        else throw std::runtime_error("reserved descriptor/window operation");
    }
    void issue(Cmd c0){
        cmd=c0;cmd.esz=cmd.bytes==1?0:cmd.bytes==2?1:cmd.bytes==4?2:3;
        st=cmd.active?TSETUP:DONE;
    }
    //------------------------------------------------------------------ geometry helpers
    // Byte j of the current row: valid and its address (contiguous dim), window byte offset.
    void start_row(){
        // row index -> outer coordinates (descriptor order dims 0..rank-2), contiguous dim rank-1
        unsigned rank=d->rank;std::array<u64,5> org=cmd.coords;unsigned idx=row;rowv=true;
        for(int dim=int(rank)-2;dim>=0;--dim){unsigned b=d->block[dim];unsigned i=idx%b;idx/=b;org[dim]=cmd.coords[dim]+i;
            if((org[dim]>>63) || org[dim]>=d->extent[dim])rowv=false;}
        u64 addr=d->base;for(unsigned dim=0;dim+1<rank;++dim)addr+=org[dim]*d->stride[dim];
        row_addr=addr+cmd.coords[rank-1]*d->stride[rank-1];
        row_bytes=d->block[rank-1]*cmd.bytes;row_units=row_bytes/8;r_m=unsigned(row_addr&7);
        row_org=org;
    }
    std::array<u64,5> row_org{};
    // unit u of the row: aligned words (low, high) with keep masks; valid bytes by the contiguous extent
    struct UnitPlan {bool nlo=false,nhi=false;u64 wlo=0,whi=0;unsigned keep=0;u64 datalo=0,datahi=0;unsigned belo=0,behi=0;};
    UnitPlan unit_plan(unsigned u,const u64* data8)const{
        UnitPlan p;u64 g=row_addr+u*8;p.wlo=g&~7ULL;p.whi=p.wlo+8;
        for(unsigned j=0;j<8;++j){u64 cj=cmd.coords[d->rank-1]+u*(8/cmd.bytes)+j/cmd.bytes;
            bool valid=rowv && !(cj>>63) && cj<d->extent[d->rank-1];
            if(!valid){continue;}
            p.keep|=1U<<j;u64 a=g+j;unsigned off=unsigned(a&7);
            if(a<p.whi){p.nlo=true;p.belo|=1U<<off;if(data8)p.datalo|=((*data8>>(j*8))&255)<<(off*8);}
            else{p.nhi=true;p.behi|=1U<<off;if(data8)p.datahi|=((*data8>>(j*8))&255)<<(off*8);}
        }
        return p;
    }
    u64 fillword()const{if(d->padding!=3){return 0;}u64 f=d->fill(cmd.bytes);u64 w=0;for(unsigned j=0;j<8;j+=cmd.bytes){w|=f<<(j*8);}return w;}
    //------------------------------------------------------------------ requests this cycle (wires)
    struct Req {bool valid=false;u64 addr=0;bool we=false;unsigned be=0;u64 data=0;unsigned tag=0;};
    bool have_slot(unsigned& free)const{for(unsigned i=0;i<NSLOT;++i)if(!sl[i].v){free=i;return true;}return false;}
    bool have_tag(unsigned& free)const{for(unsigned i=0;i<NTAG;++i)if(!tg[i].v){free=i;return true;}return false;}
    bool have_done(unsigned& done)const{for(unsigned i=0;i<NSLOT;++i){const Slot& s=sl[i];if(s.v && (s.glo||!s.nlo) && (s.ghi||!s.nhi)){done=i;return true;}}return false;}
    bool vw_hold()const{  // the presented window write loses to the WB stage on a same-bank cycle
        if(!vw_en){return false;}
        const auto& w=k.w;
        if(!(w.en && w.res.cls==2)){return false;}
        return (k.phys(w.res.rctx,w.res.idx)%4)==(vw_idx%4);}
    Req request()const{
        Req r;unsigned fs=0,ft=0;
        if(st==DLOAD){r.valid=true;r.addr=dl_addr+8*dl_word;r.tag=0;return r;}
        if(st==TRUN){UnitPlan p=unit_plan(t_unit,nullptr);
            if((p.nlo||p.nhi) && (t_half||have_slot(fs))){bool hi=t_half||!p.nlo;r.valid=true;r.addr=hi?p.whi:p.wlo;r.tag=((t_half?t_slot:fs)<<1)|(hi?1:0);}
            return r;}
        if(st==SRUN){if(rowv && !st_need_reg() && t_unit<row_words.size() && row_words[t_unit].be && have_tag(ft)){
                const RowBeat& b=row_words[t_unit];r.valid=true;r.we=true;r.addr=b.addr;r.be=b.be;r.data=b.data;r.tag=ft;}
            return r;}
        if(st==XRUN){if(!x_unit_oob() && have_tag(ft)){r.valid=true;r.addr=x_row_base+t_unit*8;r.tag=ft;}return r;}
        return r;
    }
    // store: which window register the current word needs, and whether it is in hand
    bool st_has_cur()const{return t_unit<row_units;}
    unsigned st_vreg()const{return win.base+(t_flat+t_unit*8)/64;}
    bool st_need_reg()const{return st_has_cur() && !(sr_v && sr_idx==st_vreg());}
    // transposed helpers
    u64 x_row_base=0,x_base0=0;
    bool x_row_oob()const{u64 c=cmd.coords[0]+x_row;return (c>>63) || c>=d->extent[0];}
    bool x_unit_oob()const{if(x_row_oob()){return true;}u64 left=d->extent[1]>cmd.coords[1]?d->extent[1]-cmd.coords[1]:0;unsigned ok=unsigned(std::min<u64>(left,d->block[1]));return t_unit*8>=ok*cmd.bytes;}
    unsigned x_keep()const{u64 left=d->extent[1]>cmd.coords[1]?d->extent[1]-cmd.coords[1]:0;unsigned ok=unsigned(std::min<u64>(left,d->block[1]))*cmd.bytes;unsigned rem=ok>t_unit*8?ok-t_unit*8:0;return rem>=8?255:(1U<<rem)-1;}
    //------------------------------------------------------------------ the edge
    // `fired`: the request of this cycle was accepted (the port answers at `rsp` with `got`).
    void edge(bool fired,u64 rsp,u64 got){
        const u64 t=k.t;
        // sl_done / have_done are combinational from the registered glo/ghi: evaluated before this edge's responses
        unsigned done_slot=0;const bool had_done=have_done(done_slot);
        // responses this cycle
        for(auto& s:sl){if(!s.v){continue;}if(s.nlo && !s.glo && s.rsp_lo==t){s.glo=true;}if(s.nhi && !s.ghi && s.rsp_hi==t)s.ghi=true;}
        bool rsp_now=false;
        for(auto& g:tg)if(g.v && g.rsp==t){g.v=false;rsp_now=true;if(st==XRUN||st==XDRAIN||st==XWRITE){xbuf[g.xrow]=g.data;}}
        for(auto& s:sl)if(s.v && ((s.nlo && s.rsp_lo==t)||(s.nhi && s.rsp_hi==t)))rsp_now=true;
        if(dl_waiting && dl_rsp==t){rsp_now=true;}
        // S_TDRAIN reads sl_v before this edge: a slot freed by the write port now still counts [4.2]
        bool slots_pre=false;for(auto& s:sl){if(s.v){slots_pre=true;}}
        // the presented window write: lands unless held
        bool hold=vw_hold();
        if(vw_en && !hold){Frame& f=k.F(cmd.rctx);PhysicalRegisters::Vector cur=f.physical->read(vw_idx,sc_time_stamp(),owner);
            for(unsigned l=0;l<16;++l)if(vw_mask&(1U<<l))cur[l]=vw_data[l];
            f.physical->write(vw_idx,vw_mask,sc_time_stamp(),cur,owner);f.physical->commit(sc_time_stamp());vw_en=false;}
        // a completed load slot goes to its window register (write port first)
        if(had_done && !cmd.store && !cmd.transpose && !hold){
            Slot& s=sl[done_slot];
            // assemble: the two words shifted by the row misalignment, bytes not kept become fill
            u64 shifted=s.m?((s.dlo>>(s.m*8))|(s.dhi<<(64-s.m*8))):s.dlo;u64 fw=fillword(),out=0;
            for(unsigned j=0;j<8;++j){unsigned char v=(s.keep>>j)&1?(unsigned char)(shifted>>(j*8)):(unsigned char)(fw>>(j*8));out|=u64(v)<<(j*8);}

            vw_en=true;vw_idx=s.vreg;vw_mask=3U<<s.lane;vw_data={};vw_data[s.lane]=std::uint32_t(out);vw_data[s.lane+1]=std::uint32_t(out>>32);
            s.v=false;
        }
        switch(st){
        case IDLE:break;
        case DLOAD:if(fired){dl_waiting=true;dl_rsp=rsp;dl_data=got;st=DWAIT;}break;
        case DWAIT:if(dl_waiting && dl_rsp==t){dl_waiting=false;for(unsigned b=0;b<8;++b)dl_buf[dl_word*8+b]=(unsigned char)(dl_data>>(b*8));
                if(dl_word==31)st=DPARSE;else{++dl_word;st=DLOAD;}}break;
        case DPARSE:descriptors[dl_slot].snapshot=std::make_shared<TensorMapModel>(dl_buf);st=DONE;break;
        case TSETUP:{
            if(!(cmd.desc<16 && descriptors[cmd.desc].snapshot) || !(cmd.win<32 && windows[cmd.win].valid)){fault_code=4;fault_why="DESCRIPTOR_FAULT: unloaded descriptor or undefined window";st=FAULT;break;}
            d=descriptors[cmd.desc].snapshot;win=windows[cmd.win];st=TCHK;break;}
        case TCHK:{
            try{total=d->elements(cmd.bytes,win.length)*cmd.bytes;
                if(d->padding==2)arch(d->inside(cmd.coords),4,"DESCRIPTOR_FAULT: out-of-bounds fault padding");
                if(cmd.transpose)d->check_transpose(cmd.bytes,cmd.store,cmd.coords);
                used_registers=(total+63)/64;base=k.F(cmd.rctx).vbase+win.base;
                arch(win.base<64 && used_registers<=64-win.base,4,"DESCRIPTOR_FAULT: transfer exceeds role mapping");
                owner=k.F(cmd.rctx).physical->new_owner();k.F(cmd.rctx).physical->acquire(base,used_registers,owner,!cmd.store,sc_time_stamp());
                ++descriptors[cmd.desc].references;
            }catch(const ArchFault& e){fault_code=e.code;fault_why=e.what();st=FAULT;break;}
            rows=1;for(unsigned dim=0;dim+1<d->rank;++dim)rows*=d->block[dim];
            mul_left=d->rank;st=TMUL;break;}
        case TMUL:if(--mul_left==0){row=0;t_unit=0;t_half=false;t_flat=0;st_prev=0;sr_v=false;start_row();
                if(cmd.transpose){x_row=0;x_row_base=x_base0=row_addr;st=XRUN;}else st=cmd.store?SRUN:TRUN;}break;
        case TRUN:{
            UnitPlan p=unit_plan(t_unit,nullptr);bool adv=false;unsigned fs=0;
            if(!p.nlo && !p.nhi){
                if(!had_done && !hold && !vw_en){ // the fill written straight into the window
                    unsigned off=t_flat+t_unit*8;vw_en=true;vw_idx=base+off/64;unsigned lane=(off%64)/4;vw_mask=3U<<lane;vw_data={};
                    u64 fw=fillword();vw_data[lane]=std::uint32_t(fw);vw_data[lane+1]=std::uint32_t(fw>>32);adv=true;}
            }else if(fired){
                if(!t_half){have_slot(fs);Slot& s=sl[fs];s=Slot{};s.v=true;s.nlo=p.nlo;s.nhi=p.nhi;s.keep=p.keep;s.m=r_m;
                    unsigned off=t_flat+t_unit*8;s.vreg=base+off/64;s.lane=(off%64)/4;t_slot=fs;
                    bool hi=!p.nlo;
                    if(hi){s.rsp_hi=rsp;s.dhi=got;}else{s.rsp_lo=rsp;s.dlo=got;}
                }else{Slot& s=sl[t_slot];s.rsp_hi=rsp;s.dhi=got;}
                if(!t_half && p.nlo && p.nhi){t_half=true;}else{t_half=false;adv=true;}
            }
            if(adv){if(t_unit+1==row_units){t_unit=0;t_flat+=row_bytes;if(row+1==rows)st=TDRAIN;else{++row;start_row();}}else ++t_unit;}
            break;}
        case TDRAIN:{if(!slots_pre && !rsp_now){st=DONE;}break;}
        case SRUN:{
            bool adv=false,row_end=false;
            if(!rowv)row_end=true;
            else if(st_need_reg()){st=SREAD;sread_req=true;}
            else if(t_unit>=row_words.size() || row_words[t_unit].be==0 || fired){
                if(fired){unsigned ft=0;have_tag(ft);tg[ft].v=true;tg[ft].rsp=rsp;}
                adv=true;
            }
            if(adv){if(t_unit+1==row_words.size()){row_end=true;}else {++t_unit;}}
            if(row_end){t_unit=0;st_prev=0;t_flat+=row_bytes;if(row+1==rows)st=SDRAIN;else{++row;start_row();build_store_row();}}
            break;}
        case SREAD:break;  // advanced by sread_response()
        case SDRAIN:{bool any=false;for(auto& g:tg){if(g.v)any=true;}if(!any && !rsp_now){st=DONE;}break;}
        case XRUN:{
            if(x_unit_oob() || fired){
                if(x_unit_oob()){xbuf[x_row&15]=0;}
                else{unsigned ft=0;have_tag(ft);tg[ft].v=true;tg[ft].rsp=rsp;tg[ft].xrow=x_row&15;tg[ft].xkeep=x_keep();
                    u64 keepm=0;for(unsigned j=0;j<8;++j){if((tg[ft].xkeep>>j)&1)keepm|=255ULL<<(j*8);}tg[ft].data=got&keepm;}
                if(x_row+1==d->block[0]){st=XDRAIN;}else{++x_row;x_row_base+=d->stride[0];}
            }break;}
        case XDRAIN:{bool any=false;for(auto& g:tg){if(g.v)any=true;}if(!any && !rsp_now){x_q=0;st=XWRITE;}break;}
        case XWRITE:if(!hold && !vw_en){
                unsigned rows_b=d->block[0];unsigned drow_bytes=rows_b*2;unsigned ddst=((t_unit<<2)+x_q)*drow_bytes;
                vw_en=true;vw_idx=base+ddst/64;unsigned lane0=(ddst%64)/4;vw_mask=0;vw_data={};
                for(unsigned l=lane0;l<lane0+drow_bytes/4 && l<16;++l)vw_mask|=1U<<l;
                std::array<std::uint16_t,32> halves{};for(unsigned h=0;h<2*16;++h){unsigned src=h&(rows_b-1);halves[h]=std::uint16_t(xbuf[src]>>(x_q*16));}
                for(unsigned l=0;l<16;++l)vw_data[l]=std::uint32_t(halves[2*l])|(std::uint32_t(halves[2*l+1])<<16);
                if(x_q==3){if(t_unit+1==row_units)st=DONE;else{++t_unit;x_row=0;x_row_base=x_base0;st=XRUN;}}else ++x_q;
            }break;
        case DONE:if(!hold){done_at=t+1;done_bar=cmd.bar_en;done_token=cmd.token;done_err=false;finish();st=IDLE;}break;
        case FAULT:if(!hold){done_at=t+1;done_bar=cmd.bar_en;done_token=cmd.token;done_err=true;fault_at=t+1;finish();st=IDLE;}break;
        }
    }
    void finish(){ // leases and references of a completed (or faulted) transfer
        if(owner){Frame& f=k.F(cmd.rctx);f.physical->release(base,used_registers,owner,!cmd.store,sc_time_stamp());owner=0;--descriptors[cmd.desc].references;}
    }
    // the store row's memory words, built from the window registers as they are read (data filled at SREAD)
    void build_store_row(){
        row_words.clear();if(!rowv)return;
        std::map<u64,RowBeat> words;
        for(unsigned u=0;u<row_units;++u){u64 zero=0;UnitPlan p=unit_plan(u,&zero);
            if(p.nlo){auto& b=words[p.wlo];b.addr=p.wlo;b.be|=p.belo;b.unit=u;}
            if(p.nhi){auto& b=words[p.whi];b.addr=p.whi;b.be|=p.behi;b.unit=u;b.high=true;}}
        // in address order = word order; the RTL walks st_nwords = row_units (+1 if misaligned)
        unsigned nwords=row_units+(r_m?1:0);
        for(unsigned wdx=0;wdx<nwords;++wdx){u64 wa=((row_addr)&~7ULL)+wdx*8;auto it=words.find(wa);RowBeat b;b.addr=wa;if(it!=words.end()){b.be=it->second.be;}row_words.push_back(b);}
    }
    // SREAD: the window register read through port 4 of the banked file (arbiter), data next cycle
    void sread_response(const PhysicalRegisters::Vector& data){sr_v=true;sr_idx=st_vreg();sr_data=data;sread_pending=false;st=SRUN;fill_store_data();}
    void fill_store_data(){
        // the data of each memory word of the row from the register(s) in hand: byte j of the row at window offset t_flat+j
        for(auto& b:row_words){u64 data=0;
            for(unsigned off=0;off<8;++off){if(!((b.be>>off)&1)){continue;}u64 a=b.addr+off;u64 rb=a-row_addr;/* row byte */
                unsigned woff=t_flat+unsigned(rb);unsigned reg=base+woff/64,lane=(woff%64)/4,byte=woff%4;
                if(sr_v && sr_idx==reg)data|=u64((sr_data[lane]>>(byte*8))&255)<<(off*8);
                else{ // a register not in hand: read it directly (an earlier register of a misaligned word)
                    auto v=k.F(cmd.rctx).physical->read(reg,sc_time_stamp(),owner);data|=u64((v[lane]>>(byte*8))&255)<<(off*8);}}
            b.data=data;}
    }
};
