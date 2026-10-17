from pathlib import Path
import os,sys
from tnca import device
bench=Path('tb/device/tnca_device.v');original=bench.read_bytes()
try:
 source=original.decode().replace('endmodule','always @(posedge clk) if (!core_idle && l_req_valid && l_req_ready) $display("RAMACCESS %h %b %h",l_req_addr,l_req_we,l_req_wdata);\nendmodule')
 bench.write_text(source)
 os.environ['TNCA_DEVICE']=str(device.prepare('core-vrf-sync-v5'))
finally:bench.write_bytes(original)
sys.argv=['replay','exact-rtl-trace']
source=Path('/tmp/r40_sdpa_snapshot_replay_any.py').read_text()
source=source.replace('reply=s.dev.upload(0,ram);assert reply.ok',"\nfor lo,hi in [(0,131072),(0x20000,0x20040),(0x20600,0x20e80)]:\n reply=s.dev.upload(lo,ram[lo:hi]);assert reply.ok")
exec(compile(source,'replay','exec'))
