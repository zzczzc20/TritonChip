import pathlib,shutil,subprocess
base=pathlib.Path('/tmp/r40_mmio_journal25_native');dest=pathlib.Path('/tmp/r40_mmio_journal29_first_unknown')
shutil.copytree(base,dest,ignore=shutil.ignore_patterns('build','__pycache__','.git'))
monitor='''
// Diagnostic-only replay monitor: observes, never drives DUT state.
`ifndef SYNTHESIS
always @(negedge clk) begin
 if (rst_n && !launch_valid &&
     ((^wb_rsv === 1'bx) || (r_valid === 1'bx) ||
      ((r_valid === 1'b1) && (kill_r === 1'b0) && (^r_wb_delay === 1'bx)))) begin
  $display("FIRST_UNKNOWN %m @%0t cycle=%0d running=%b pc=%h ctx=%h major=%h func=%h dst=%h rvalid=%b dvalid=%b xvalid=%b rsv=%b delay=%h stall=%b kill=%b books=%b mem=%b mul=%b fp=%b", $time,clock_ctr,running,r_pc,r_rctx,r_major,r_func,r_dstc,r_valid,d_valid,x_valid,wb_rsv,r_wb_delay,stall_front,kill_r,r_books_wb,r_mem_en,r_is_mul,r_is_fpnorm);
  $fatal(1,"first unknown reservation or live decode state");
 end
end

always @(posedge clk) begin
 if(rst_n && clock_ctr>=332390 && clock_ctr<=332430)
  $display("PRE_UNKNOWN %m @%0t cyc=%0d run=%b r(pc,ctx,valid,major,func,dst)=%h,%h,%b,%h,%h,%h d(pc,ctx,valid,insn)=%h,%h,%b,%h x(pc,ctx,valid,major,func)=%h,%h,%b,%h,%h rsv=%h delay=%h stall=%b kill(r,d,av)=%b%b%b book=%b redirect=%b br(taken,enable)=%b%b exit=%b trap=%b scalar_a=%h avail(valid,pc,insn)=%b,%h,%h fbv=%b f1v=%b imem=%h",$time,clock_ctr,running,r_pc,r_rctx,r_valid,r_major,r_func,r_dstc,d_pc,d_rctx,d_valid,d_insn,x_pc,x_rctx,x_valid,x_major,x_func,wb_rsv,r_wb_delay,stall_front,kill_r,kill_d,kill_av,r_books_wb,x_redirect,br_taken,x_is_branch,x_is_exit,x_is_trap,sa_data,avail,avail_pc,avail_insn,fb_valid,f1_valid,imem_data);
end
`endif
'''
p=dest/'rtl/core/tnca_core.v';text=p.read_text();ix=text.rfind('endmodule');assert ix>0;p.write_text(text[:ix]+monitor+text[ix:]);(dest/'first_unknown_monitor.vh').write_text(monitor)
decoder=dest/'rtl/core/tnca_decoder.v'
text=decoder.read_text();needle='            o_fu          = `TNCA_FU_NONE;\n            o_dst_class   = `TNCA_DST_NONE;'
assert text.count(needle)==1
text=text.replace(needle,"            // Known-illegal instructions carry their trap flag, but must not\n            // select execution units using unspecified raw opcode bits.\n            o_major       = `TNCA_MAJ_SYS;\n            o_func        = `TNCA_SYS_NOP;\n            o_mod         = 10'd0;\n            o_fu          = `TNCA_FU_NONE;\n            o_dst_class   = `TNCA_DST_NONE;")
decoder.write_text(text)
helper=dest/'build_first_unknown.py';helper.write_text('''import sys,json,pathlib
for rel in ['python','compiler/python','tools/asm']:sys.path.insert(0,str(pathlib.Path.cwd()/rel))
import triton
from tnca.backends.multicore_mmio_sim import MulticoreMMIOLink
link=MulticoreMMIOLink(core_count=3,memory_ports=3,board_top=True,maintenance=True,memory_controls=True)
p=link._build()
pathlib.Path('first_unknown_harness.txt').write_text(p)
print(p,flush=True)
''')
script='''source ~/tc_env/env.sh
cd /tmp/r40_mmio_journal29_first_unknown
~/tc_env/torchenv/bin/python build_first_unknown.py > /tmp/r40_mmio_journal29_build.log 2>&1
result=$?
echo "$result" > /tmp/r40_mmio_journal29_build.exit
if [ "$result" -eq 0 ]; then
 python3 tools/r40/replay_mmio_journal.py --journal /tmp/r40_mmio_journal25_native/sim/build/u50-sim/link-1his1__i/journal --diagnostic-harness "$(cat first_unknown_harness.txt)" --out /tmp/r40_mmio_journal29_first_unknown_replay --cap-seconds 1800 > /tmp/r40_mmio_journal29_first_unknown_replay.log 2>&1
 echo $? > /tmp/r40_mmio_journal29_first_unknown_replay.exit
fi
'''
p=pathlib.Path('/tmp/r40_mmio_journal29_first_unknown_runner.sh');p.write_text(script);f=open('/tmp/r40_mmio_journal29_runner.log','w');pr=subprocess.Popen(['bash',str(p)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);print('first_unknown_runner',pr.pid)
