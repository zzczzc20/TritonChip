from tnca.backends.multicore_mmio import MulticoreMMIOBackend
from tnca.backends.u50 import U50Link, R_HOST_ADDR, R_HOST_CTRL, R_HOST_DATA, R_HOST_STATUS, ST_VALID, ST_REFUSED
from tnca.errors import DeviceLost
class Link(U50Link):
 def __init__(self):self.words=1024;self.fail=False;self.address=0;self.value=0;self.st=ST_VALID;self.mem={};self.naddr=0;self.nwrite=0
 def open(self):return self
 def close(self):pass
 def alive(self):return True
 def read_reg(self,reg):return self.st if reg==R_HOST_STATUS else 0
 def write_reg(self,reg,value):
  if reg==R_HOST_ADDR:self.address=value;self.naddr+=1
  if reg==R_HOST_DATA:self.value=value
  if reg==R_HOST_CTRL:
   if self.fail:self.st=ST_REFUSED
   else:self.mem[self.address]=self.value;self.address+=8;self.st=ST_VALID;self.nwrite+=1
l=Link(); b=MulticoreMMIOBackend(l);b.local_limit=512
words=[11,22,33,44,55]; data=lambda a:memoryview(b''.join(i.to_bytes(8,'little') for i in a))
b._put_payload_bytes(32,data(words));assert l.naddr==1 and l.nwrite==5
b._put_payload_bytes(32,data(words));assert l.naddr==1 and l.nwrite==5 and b.code_words_skipped==5
words2=[11,99,33,88,55];b._put_payload_bytes(32,data(words2));assert l.naddr==3 and l.nwrite==7
assert [l.mem[32+i*8] for i in range(5)]==words2
l.fail=True
try:b._put_payload_bytes(40,data([100]))
except DeviceLost:pass
else:raise AssertionError('refusal was ignored')
assert 40 not in b._code_window_shadow
l.fail=False;b._put_payload_bytes(40,data([100]));assert l.mem[40]==100 and b._code_window_shadow[40]==100
b._window(48,101,write=True);b._put_payload_bytes(48,data([33]));assert l.mem[48]==33
print('MMIO sequential, skip, changed spans, refusal/retry, and direct writes: pass')
