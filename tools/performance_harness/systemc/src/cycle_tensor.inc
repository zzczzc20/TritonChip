//==============================================================================
// The cycle-level transfer engine (tnca_tensor.v) of the cycle core
// [CORE_TIMING.md 4.2]. Included inside `struct CycleCore`. State names follow
// the Verilog; every rule is "during cycle t from the pre-edge state, at the
// edge these registers change".
//==============================================================================
struct CycleTensor {
    CycleCore& k;
    enum St {IDLE,DLOAD,DWAIT,DPARSE,TSETUP,TCHK,TMUL,TRUN,TDRAIN,SRUN,SREAD,SDRAIN,DONE,FAULT,XRUN,XDRAIN,XWRITE,
             LRUN,LSREAD,LSRUN,LEND,LDLOAD,LDWAIT,LDFOLD};   // R41 G4 LINE_ENGINE states [4.4]
    static constexpr unsigned NTAG=16,NSLOT=8,CMD_N=8,XB=16;
    St st=IDLE;
    //------------------------------------------------------------------ R41 G4 LINE_ENGINE [4.4]
    // core_tensor_line=1: the engine owns the core's 64-byte line port. A transfer whose rows are whole
    // lines inside the tensor moves as line requests through the line slots, and its command record
    // lets the walk take the next command while its lines are in flight; a line-aligned TensorMap is
    // read as four lines; the transposed walk keeps the lines it fetched. core_tensor_mul_pipe=1: the
    // starting address takes one multiply a cycle, each product registered (MUL_PIPE).
    bool le=false,mul_pipe=false;
    struct LSlot {bool v=false,rdy=false,st=false;unsigned cid=0,vreg=0,rctx=0;u64 rsp=0;std::array<u64,8> line{};};
    std::vector<LSlot> ls;
    struct LCmd {bool v=false,open=false,ben=false,store=false;unsigned cnt=0,rctx=0,base=0,used=0;u64 token=0,owner=0;};
    std::array<LCmd,CMD_N> cm{};
    unsigned t_cid=0,t_line=0;bool pf_v=false;PhysicalRegisters::Vector sr_next{};
    unsigned dlq=0,dlr=0;std::array<u64,4> dl_lrsp{};      // descriptor lines requested / answered, and when
    bool vr_now=false;PhysicalRegisters::Vector vr_data{};  // the engine read port answered this cycle
    unsigned vw_rctx=0;u64 vw_owner=0;                       // whose register the presented window write is
    // the transposed walk's line buffer (g_line_port): an entry per source row of the tile
    struct XEntry {bool v=false,f=false;unsigned li=0;std::array<u64,8> d{};};std::array<XEntry,XB> xb{};
    struct XFill {bool v=false;unsigned e=0,li=0;std::array<u64,8> line{};};std::array<XFill,NTAG> xf{};
    bool lr_v=false;unsigned lr_tag=0;u64 lr_data=0;         // the local answer of a buffer hit
    void configure(bool line_engine,unsigned line_slots,bool pipelined_mul){le=line_engine;mul_pipe=pipelined_mul;ls.assign(line_engine?line_slots:0,LSlot{});}
    bool busy_any()const{if(st!=IDLE){return true;}for(auto& s:ls){if(s.v)return true;}for(auto& c:cm){if(c.v)return true;}return false;}
    bool inflight(unsigned rc)const{for(auto& c:cm){if(c.v && c.rctx==rc)return true;}return false;}
    bool lfree(unsigned& f)const{for(unsigned i=0;i<ls.size();++i)if(!ls[i].v){f=i;return true;}return false;}
    bool lrdy(unsigned& f)const{for(unsigned i=0;i<ls.size();++i)if(ls[i].v && ls[i].rdy){f=i;return true;}return false;}
    bool cfree(unsigned& f)const{for(unsigned i=0;i<CMD_N;++i)if(!cm[i].v){f=i;return true;}return false;}
    bool cdone(unsigned& f)const{for(unsigned i=0;i<CMD_N;++i)if(cm[i].v && !cm[i].open && cm[i].cnt==0){f=i;return true;}return false;}
    unsigned row_lines()const{return row_bytes/64;}
    bool l_more()const{return !(row+1==rows && t_line+1==row_lines());}
    unsigned l_vrel()const{return win.base+t_flat/64+t_line;}   // the line's window register (context-relative)
    // the line a transposed unit falls in, relative to the row base's line, and its buffer entry
    unsigned x_li()const{return unsigned(((x_row_base&63)+(u64(t_unit)&63)*8)>>6);}
    bool x_hit()const{const XEntry& e=xb[x_row&15];return le && e.v && e.f && e.li==x_li() && t_unit<64;}
    // a word-walk response from memory this cycle (mem_lg): it has the response path before a local answer
    bool mem_lg_now()const{const u64 t=k.t;
        for(auto& s:sl)if(s.v && ((s.nlo && s.rsp_lo==t)||(s.nhi && s.rsp_hi==t)))return true;
        for(auto& g:tg)if(g.v && g.rsp==t)return true;
        return dl_waiting && dl_rsp==t;}
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
        for(auto& w:windows){w=Window{};}
        for(auto& s:ls){s=LSlot{};}for(auto& c:cm){c=LCmd{};}pf_v=false;vr_now=false;dlq=dlr=0;
        for(auto& e:xb){e=XEntry{};}for(auto& f:xf){f=XFill{};}lr_v=false;}
    //------------------------------------------------------------------ commands at X
    void config(unsigned function,unsigned slot,u64 address,unsigned immediate){
        arch((function<16 && slot<16) || (function>=16 && slot<32),4,"DESCRIPTOR_FAULT: configuration slot");
        if(function==0){require(address%8==0,"unaligned TensorMap snapshot address is not implemented");
            dl_slot=slot;dl_addr=address;dl_word=0;dl_waiting=false;dlq=dlr=0;dl_lrsp={};
            st=(le && address%64==0)?LDLOAD:DLOAD;cmd.bar_en=false;cmd.rctx=k.xb.rctx;}
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
    // `line`: a 64-byte line request (R41 LINE_ENGINE), its 64 byte enables and data in lbe/ldata
    struct Req {bool valid=false;u64 addr=0;bool we=false;unsigned be=0;u64 data=0;unsigned tag=0;bool line=false;u64 lbe=0;std::array<u64,8> ldata{};};
    // where this cycle's request's response goes (the core registers it under the beat id)
    CycleCore::Target request_target(const Req& r)const{unsigned ft=0;
        if(st==LDLOAD)return CycleCore::Target{7,dlq,0};
        if(st==LRUN || st==LSRUN){unsigned f=0;lfree(f);return CycleCore::Target{6,f,0};}
        if(st==DLOAD)return CycleCore::Target{4,0,0};
        if(st==TRUN){bool hi=(r.tag&1)!=0;unsigned slot=r.tag>>1;return CycleCore::Target{hi?2U:1U,slot,0};}
        if(st==SRUN || st==XRUN){have_tag(ft);return CycleCore::Target{3,ft,0};}
        return CycleCore::Target{};}
    bool have_slot(unsigned& free)const{for(unsigned i=0;i<NSLOT;++i)if(!sl[i].v){free=i;return true;}return false;}
    bool have_tag(unsigned& free)const{for(unsigned i=0;i<NTAG;++i)if(!tg[i].v){free=i;return true;}return false;}
    std::string dump()const{std::ostringstream o;o<<"st="<<st<<" unit="<<t_unit<<" half="<<t_half<<" vw_en="<<vw_en<<" vw_idx="<<vw_idx<<" hold="<<vw_hold()<<" slots:";
        for(unsigned i=0;i<NSLOT;++i){const Slot& s=sl[i];if(s.v)o<<" ["<<i<<" lo="<<s.nlo<<"/"<<s.glo<<"@"<<s.rsp_lo<<" hi="<<s.nhi<<"/"<<s.ghi<<"@"<<s.rsp_hi<<"]";}
        o<<" tags:";for(unsigned i=0;i<NTAG;++i)if(tg[i].v)o<<" "<<i<<"@"<<tg[i].rsp;return o.str();}
    bool have_done(unsigned& done)const{for(unsigned i=0;i<NSLOT;++i){const Slot& s=sl[i];if(s.v && (s.glo||!s.nlo) && (s.ghi||!s.nhi)){done=i;return true;}}return false;}
    bool vw_hold()const{  // the presented window write loses to the WB stage on a same-bank cycle
        if(!vw_en){return false;}
        const auto& w=k.w;
        if(!(w.en && w.res.cls==2)){return false;}
        return (k.phys(w.res.rctx,w.res.idx)%4)==(vw_idx%4);}
    Req request()const{
        Req r;unsigned fs=0,ft=0,lf=0;
        // R41 G4: the four lines of a line-aligned TensorMap; a line of the line walk (a store sends the
        // register in hand; every line request enables all 64 bytes)
        if(st==LDLOAD){if(dlq<4){r.valid=true;r.line=true;r.addr=dl_addr+64*u64(dlq);r.lbe=~0ULL;}return r;}
        if(st==LRUN || st==LSRUN){if(lfree(lf)){r.valid=true;r.line=true;r.addr=row_addr+64*u64(t_line);r.we=st==LSRUN;r.lbe=~0ULL;
                if(r.we)for(unsigned w=0;w<8;++w)r.ldata[w]=u64(sr_data[2*w])|(u64(sr_data[2*w+1])<<32);}
            return r;}
        if(le && lr_v && mem_lg_now())return r;   // lr_free=0: a held local answer blocks the word walk's request
        if(st==DLOAD){r.valid=true;r.addr=dl_addr+8*dl_word;r.tag=0;return r;}
        if(st==TRUN){UnitPlan p=unit_plan(t_unit,nullptr);
            if((p.nlo||p.nhi) && (t_half||have_slot(fs))){bool hi=t_half||!p.nlo;r.valid=true;r.addr=hi?p.whi:p.wlo;r.tag=((t_half?t_slot:fs)<<1)|(hi?1:0);}
            return r;}
        if(st==SRUN){if(rowv && !st_need_reg() && t_unit<row_words.size() && row_words[t_unit].be && have_tag(ft)){
                const RowBeat& b=row_words[t_unit];r.valid=true;r.we=true;r.addr=b.addr;r.be=b.be;r.data=b.data;r.tag=ft;}
            return r;}
        if(st==XRUN){if(!x_unit_oob() && have_tag(ft) && !x_hit()){r.valid=true;r.addr=x_row_base+t_unit*8;r.tag=ft;}return r;}   // a buffer hit is answered locally
        return r;
    }
    // the engine's register read port (port 4): which window register it asks for this cycle
    bool vr_req(unsigned& idx)const{
        if(st==SREAD){idx=st_vreg();return true;}
        if(st==LSREAD){idx=l_vrel();return true;}
        if(st==LSRUN && !pf_v && l_more()){idx=l_vrel()+1;return true;}   // the next line's register, read ahead
        return false;}
    void vr_response(const PhysicalRegisters::Vector& data){
        if(st==SREAD){sread_response(data);return;}
        vr_now=true;vr_data=data;sread_pending=false;}
    // a line request's data, captured when it is issued (the functional access)
    void line_data(const CycleCore::Target& tg,const std::array<u64,8>& line){
        if(tg.kind==6)ls[tg.a].line=line;
        else if(tg.kind==7)for(unsigned w=0;w<8;++w)for(unsigned b=0;b<8;++b)dl_buf[tg.a*64+w*8+b]=(unsigned char)(line[w]>>(8*b));}
    // the eligibility of the transfer the walk is about to start (le_elig): every row whole 64-byte lines
    // inside the tensor, line-aligned, not transposed
    bool le_elig()const{
        if(!le || cmd.transpose || row_addr%64 || row_bytes%64)return false;
        for(unsigned dim=0;dim+1<d->rank;++dim)if(d->stride[dim]%64)return false;
        for(unsigned dim=0;dim<d->rank;++dim){u64 c=cmd.coords[dim];if((c>>63) || c+d->block[dim]>d->extent[dim])return false;}
        return true;}
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
    void edge(bool fired,u64 id){(void)id;
        const u64 t=k.t;
        // sl_done / have_done are combinational from the registered glo/ghi: evaluated before this edge's responses
        unsigned done_slot=0;const bool had_done=have_done(done_slot);
        // the load slot a request of this cycle takes is the pre-edge `free_slot` (tnca_tensor.v 599, 1044):
        // it is the slot in the request's tag, not one freed by the window write at this same edge
        unsigned fs_pre=0;have_slot(fs_pre);
        // likewise the tag a store beat or a transposed-load request takes is the pre-edge `free_tag` (tnca_tensor.v 506-511,
        // 1110, 1164): a tag freed by a response at this same edge is reused by the NEXT request, not this one
        unsigned ft_pre=0;const bool ft_have=have_tag(ft_pre);
        // R41 G4 pre-edge wires [4.4]
        unsigned lf_pre=0,lr_pre=0,cd_pre=0,cf_pre=0;
        lfree(lf_pre);const bool lrdy_have=lrdy(lr_pre),cdone_have=le && cdone(cd_pre),cfree_have=cfree(cf_pre);
        const bool mem_lg=le && mem_lg_now(),pf_v_pre=pf_v,pf_req=st==LSRUN && !pf_v && l_more();
        const bool l_fire=fired && (st==LRUN || st==LSRUN),d_fire=fired && st==LDLOAD;
        const unsigned dlq_pre=dlq,dlr_pre=dlr;
        const Req pre=request();
        const bool xb_clear=le && (st==TSETUP || (pre.valid && pre.we));   // any write request (word or line) empties the buffer
        // transposed: a buffer hit fires locally when the response path is free; a miss's read went to memory
        const bool xreq=st==XRUN && !x_unit_oob() && ft_have,lr_free=!lr_v || !mem_lg;
        const bool xhit=xreq && x_hit() && lr_free,x_alloc=le && xreq && !x_hit() && fired;
        const bool hold=vw_hold();
        bool trun_fill=false;if(st==TRUN){UnitPlan p0=unit_plan(t_unit,nullptr);trun_fill=!p0.nlo && !p0.nhi && !had_done;}
        const bool legacy_vw=(had_done && !cmd.store && !cmd.transpose) || trun_fill || st==XWRITE;
        const bool lw_fire=le && lrdy_have && !hold && !legacy_vw;   // a line slot's register write takes any cycle the walk leaves
        // the local answer of an earlier hit leaves now, unless a memory response has the response path
        unsigned local_tag=NTAG;
        if(lr_v && !mem_lg){Tag& g=tg[lr_tag];g.rsp=t;u64 keepm=0;for(unsigned j=0;j<8;++j)if((g.xkeep>>j)&1)keepm|=255ULL<<(j*8);g.data=lr_data&keepm;lr_v=false;local_tag=lr_tag;}
        // responses this cycle
        for(auto& s:sl){if(!s.v){continue;}if(s.nlo && !s.glo && s.rsp_lo && s.rsp_lo==t){s.glo=true;}if(s.nhi && !s.ghi && s.rsp_hi && s.rsp_hi==t){s.ghi=true;}}
        bool rsp_now=false;
        // the buffer entry a miss allocates (before this edge's fills: a fill of the same entry loses)
        if(x_alloc && !xb_clear){const unsigned e=x_row&15;xb[e].v=true;xb[e].f=false;for(auto& f:xf)if(f.e==e)f.v=false;
            XFill& f=xf[ft_pre];f.v=true;f.e=e;f.li=x_li();const u64 la=(x_row_base+u64(t_unit)*8)&~63ULL;
            for(unsigned w=0;w<8;++w)f.line[w]=k.c.tm->access(la+8*w,false,0,255);}
        for(unsigned i=0;i<NTAG;++i){Tag& g=tg[i];if(!(g.v && g.rsp && g.rsp==t)){continue;}
            g.v=false;rsp_now=true;if(st==XRUN||st==XDRAIN||st==XWRITE){xbuf[g.xrow]=g.data;}
            if(le && i!=local_tag && !xb_clear){
                if(xf[i].v && !(x_alloc && xf[i].e==(x_row&15)) && !(x_alloc && ft_pre==i)){xb[xf[i].e].f=true;xb[xf[i].e].li=xf[i].li;xb[xf[i].e].d=xf[i].line;}
                if(!(x_alloc && ft_pre==i))xf[i].v=false;}}
        if(xb_clear){for(auto& e:xb){e=XEntry{};}for(auto& f:xf){f=XFill{};}}
        if(xhit){lr_v=true;lr_tag=ft_pre;const u64 a=x_row_base+u64(t_unit)*8;lr_data=xb[x_row&15].d[(a>>3)&7];}
        for(auto& s:sl)if(s.v && ((s.nlo && s.rsp_lo && s.rsp_lo==t)||(s.nhi && s.rsp_hi && s.rsp_hi==t)))rsp_now=true;
        if(dl_waiting && dl_rsp && dl_rsp==t){rsp_now=true;}
        // S_TDRAIN reads sl_v before this edge: a slot freed by the write port now still counts [4.2]
        bool slots_pre=false;for(auto& s:sl){if(s.v){slots_pre=true;}}
        // the presented window write: lands unless held
        if(vw_en && !hold){Frame& f=k.F(vw_rctx);PhysicalRegisters::Vector cur=f.physical->read(vw_idx,sc_time_stamp(),vw_owner);
            for(unsigned l=0;l<16;++l)if(vw_mask&(1U<<l))cur[l]=vw_data[l];
            f.physical->write(vw_idx,vw_mask,sc_time_stamp(),cur,vw_owner);f.physical->commit(sc_time_stamp());vw_en=false;}
        // R41 G4 line slots: responses, a register write, the counts and a finished command's completion [4.4]
        if(le){
            for(unsigned q=0;q<4;++q)if(dl_lrsp[q] && dl_lrsp[q]==t)++dlr;
            for(auto& s:ls)if(s.v && s.rsp && s.rsp==t){if(s.st){s.v=false;--cm[s.cid].cnt;}else s.rdy=true;}   // a store's ack frees it
            if(lw_fire){LSlot& s=ls[lr_pre];vw_en=true;vw_idx=s.vreg;vw_mask=0xffff;vw_rctx=s.rctx;vw_owner=cm[s.cid].owner;
                for(unsigned l=0;l<16;++l)vw_data[l]=std::uint32_t(s.line[l/2]>>(32*(l%2)));
                s.v=false;s.rdy=false;--cm[s.cid].cnt;}
            if(l_fire){LSlot& s=ls[lf_pre];s.v=true;s.rdy=false;s.st=st==LSRUN;s.cid=t_cid;s.vreg=base+t_flat/64+t_line;s.rctx=cmd.rctx;s.rsp=0;++cm[t_cid].cnt;}
            if(cdone_have && st!=DONE && st!=FAULT){LCmd& c=cm[cd_pre];done_at=t+1;done_bar=c.ben;done_token=c.token;done_err=false;
                if(c.owner)k.F(c.rctx).physical->release(c.base,c.used,c.owner,!c.store,sc_time_stamp());
                c=LCmd{};}
        }
        // a completed load slot goes to its window register (write port first)
        if(had_done && !cmd.store && !cmd.transpose && !hold){
            Slot& s=sl[done_slot];
            // assemble: the two words shifted by the row misalignment, bytes not kept become fill
            u64 shifted=s.m?((s.dlo>>(s.m*8))|(s.dhi<<(64-s.m*8))):s.dlo;u64 fw=fillword(),out=0;
            for(unsigned j=0;j<8;++j){unsigned char v=(s.keep>>j)&1?(unsigned char)(shifted>>(j*8)):(unsigned char)(fw>>(j*8));out|=u64(v)<<(j*8);}

            vw_en=true;vw_rctx=cmd.rctx;vw_owner=owner;vw_idx=s.vreg;vw_mask=3U<<s.lane;vw_data={};vw_data[s.lane]=std::uint32_t(out);vw_data[s.lane+1]=std::uint32_t(out>>32);
            s.v=false;
        }
        switch(st){
        case IDLE:break;
        case DLOAD:if(fired){dl_waiting=true;dl_rsp=0;st=DWAIT;}break;
        case DWAIT:if(dl_waiting && dl_rsp && dl_rsp==t){dl_waiting=false;for(unsigned b=0;b<8;++b)dl_buf[dl_word*8+b]=(unsigned char)(dl_data>>(b*8));
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
            // S_TMUL: one cycle per outer position and the last; MUL_PIPE adds the registered product's cycle
            mul_left=mul_pipe?(d->rank==1?1:d->rank+1):d->rank;st=TMUL;break;}
        case TMUL:if(--mul_left==0){row=0;t_unit=0;t_half=false;t_flat=0;st_prev=0;sr_v=false;start_row();
                if(cmd.transpose){x_row=0;x_row_base=x_base0=row_addr;st=XRUN;}
                else if(le_elig()){
                    if(cfree_have){   // R41 G4: whole lines; the command gets a record and the walk is free once its lines are issued
                        LCmd& c=cm[cf_pre];c=LCmd{};c.v=c.open=true;c.ben=cmd.bar_en;c.token=cmd.token;c.rctx=cmd.rctx;c.store=cmd.store;
                        c.base=base;c.used=used_registers;c.owner=owner;owner=0;
                        t_cid=cf_pre;t_line=0;cmd.bar_en=false;pf_v=false;st=cmd.store?LSREAD:LRUN;}
                    else mul_left=1;}   // a line command waits for a free record
                else st=cmd.store?SRUN:TRUN;}break;
        case LRUN:case LSRUN:{
            const bool more=l_more(),to_read=st==LSRUN && !pf_v_pre && !(pf_req && vr_now);
            if(st==LSRUN && pf_req && vr_now && !l_fire){sr_next=vr_data;pf_v=true;}   // the next register, read ahead
            if(l_fire){
                if(st==LSRUN && more){if(pf_v_pre){sr_data=sr_next;pf_v=false;}else if(pf_req && vr_now)sr_data=vr_data;}
                if(t_line+1==row_lines()){t_line=0;t_flat+=row_bytes;
                    if(row+1==rows)st=LEND;else{if(to_read)st=LSREAD;++row;start_row();}}
                else{++t_line;if(to_read)st=LSREAD;}
            }break;}
        case LSREAD:if(vr_now){sr_data=vr_data;pf_v=false;st=LSRUN;}break;
        case LEND:cm[t_cid].open=false;--descriptors[cmd.desc].references;st=IDLE;break;   // the walk is free; the lines finish on their own
        case LDLOAD:if(d_fire)dlq=dlq_pre+1;if(dlq_pre==4 || (d_fire && dlq_pre==3))st=LDWAIT;break;
        case LDWAIT:if(dlr_pre==4){dl_word=0;st=LDFOLD;}break;
        case LDFOLD:if(dl_word==4)st=DPARSE;else ++dl_word;break;   // the box element count, one dimension a step
        case TRUN:{
            UnitPlan p=unit_plan(t_unit,nullptr);bool adv=false;unsigned fs=0;
            if(!p.nlo && !p.nhi){
                if(!had_done && !hold && !vw_en){ // the fill written straight into the window
                    unsigned off=t_flat+t_unit*8;vw_en=true;vw_rctx=cmd.rctx;vw_owner=owner;vw_idx=base+off/64;unsigned lane=(off%64)/4;vw_mask=3U<<lane;vw_data={};
                    u64 fw=fillword();vw_data[lane]=std::uint32_t(fw);vw_data[lane+1]=std::uint32_t(fw>>32);adv=true;}
            }else if(fired){
                if(!t_half){fs=fs_pre;Slot& s=sl[fs];s=Slot{};s.v=true;s.nlo=p.nlo;s.nhi=p.nhi;s.keep=p.keep;s.m=r_m;
                    unsigned off=t_flat+t_unit*8;s.vreg=base+off/64;s.lane=(off%64)/4;t_slot=fs;
                    s.rsp_lo=s.rsp_hi=0;
                }
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
                if(fired){tg[ft_pre].v=true;tg[ft_pre].rsp=0;}
                adv=true;
            }
            if(adv){if(t_unit+1==row_words.size()){row_end=true;}else {++t_unit;}}
            if(row_end){t_unit=0;st_prev=0;t_flat+=row_bytes;if(row+1==rows)st=SDRAIN;else{++row;start_row();build_store_row();}}
            break;}
        case SREAD:break;  // advanced by sread_response()
        case SDRAIN:{bool any=false;for(auto& g:tg){if(g.v)any=true;}if(!any && !rsp_now){st=DONE;}break;}
        case XRUN:{
            if(x_unit_oob() || fired || xhit){   // xhit: answered from the line buffer, no memory request
                if(x_unit_oob()){xbuf[x_row&15]=0;}
                else{tg[ft_pre].v=true;tg[ft_pre].rsp=0;tg[ft_pre].xrow=x_row&15;tg[ft_pre].xkeep=x_keep();tg[ft_pre].data=0;}
                if(x_row+1==d->block[0]){st=XDRAIN;}else{++x_row;x_row_base+=d->stride[0];}
            }break;}
        case XDRAIN:{bool any=false;for(auto& g:tg){if(g.v)any=true;}if(!any && !rsp_now){x_q=0;st=XWRITE;}break;}
        case XWRITE:if(!hold && !vw_en){
                // 16-bit elements: four destination rows a column block, a row 2R bytes; 1-byte elements (R41 X8,
                // HAS_XPOSE 2): eight rows a block, a row R bytes
                const bool x8=cmd.bytes==1;const unsigned rows_b=d->block[0],per=x8?8:4,drow_bytes=x8?rows_b:rows_b*2;
                unsigned ddst=(t_unit*per+x_q)*drow_bytes;
                vw_en=true;vw_rctx=cmd.rctx;vw_owner=owner;vw_idx=base+ddst/64;unsigned lane0=(ddst%64)/4;vw_mask=0;vw_data={};
                for(unsigned l=lane0;l<lane0+drow_bytes/4 && l<16;++l)vw_mask|=1U<<l;
                if(x8){std::array<std::uint8_t,64> bytes{};for(unsigned h=0;h<64;++h){unsigned src=h&(rows_b-1);bytes[h]=std::uint8_t(xbuf[src]>>(x_q*8));}
                    for(unsigned l=0;l<16;++l)vw_data[l]=std::uint32_t(bytes[4*l])|(std::uint32_t(bytes[4*l+1])<<8)|(std::uint32_t(bytes[4*l+2])<<16)|(std::uint32_t(bytes[4*l+3])<<24);}
                else{std::array<std::uint16_t,32> halves{};for(unsigned h=0;h<2*16;++h){unsigned src=h&(rows_b-1);halves[h]=std::uint16_t(xbuf[src]>>(x_q*16));}
                    for(unsigned l=0;l<16;++l)vw_data[l]=std::uint32_t(halves[2*l])|(std::uint32_t(halves[2*l+1])<<16);}
                if(x_q==per-1){if(t_unit+1==row_units)st=DONE;else{++t_unit;x_row=0;x_row_base=x_base0;st=XRUN;}}else ++x_q;
            }break;
        case DONE:if(!hold){done_at=t+1;done_bar=cmd.bar_en;done_token=cmd.token;done_err=false;finish();st=IDLE;}break;
        case FAULT:if(!hold){done_at=t+1;done_bar=cmd.bar_en;done_token=cmd.token;done_err=true;fault_at=t+1;finish();st=IDLE;}break;
        }
        vr_now=false;
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
