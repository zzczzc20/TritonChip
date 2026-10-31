import torch
import triton
import tlm_backend
import tnca.torch_device as td
tlm_backend.register()
td.register()
import types
from unittest import mock
import chat

class Model:
    config = types.SimpleNamespace(get_text_config=lambda: types.SimpleNamespace(num_hidden_layers=1))
    def __init__(self):
        self.weight = torch.zeros(1).to('tnca')
    def named_parameters(self):
        return iter([('weight', self.weight)])
    def named_buffers(self):
        return iter(())
class Session:
    def __init__(self):
        self.model, self.device, self.dev = Model(), 'tnca', torch.device('tnca')
        self.bucket, self.chunk, self.prompts, self.resets = 64, 0, 0, 0
        self.cache, self.cached = object(), []
        self.input = torch.arange(16, dtype=torch.float32).to('tnca')
        self.table = torch.eye(16, dtype=torch.float32).to('tnca')
        self.prefill = torch.compile(lambda x: x * 2, fullgraph=True, dynamic=False)
        self.decode = torch.compile(lambda x, table: torch.nn.functional.embedding(x, table),
                                    fullgraph=True, dynamic=False)
        self.greedy = torch.compile(lambda x: x.float().argmax().reshape(1,1), fullgraph=True, dynamic=False)
    def prefill_prompt(self, inputs, images):
        self.cached = inputs['input_ids'][0].tolist()
        return self.prefill(self.input), [], 0
    def step_device(self, token):
        return self.decode(token.clone(memory_format=torch.contiguous_format), self.table).reshape(-1)
    def reset(self):
        self.cached = []
        self.resets += 1
s = Session()
processor = types.SimpleNamespace(batch_decode=lambda ids, **kw: [str(ids.tolist())])
spec = types.SimpleNamespace(name='probe', repo='probe', revision='probe')
with mock.patch.object(chat.L, 'generation_eos', lambda model: set()), \
     mock.patch.object(chat.L, 'module_bytes', lambda *args: 1), \
     mock.patch.object(chat.L, 'fp8_route_in_use', lambda: {}), \
     mock.patch.object(chat.L, 'cache_bytes', lambda cache: 0):
    r = chat.run_generation(s, processor, spec, {'input_ids': torch.tensor([[1,2]])}, 'probe', new_tokens=4)
print('status',r['status'],'tokens',r['generated_tokens'], 'covered', r['compiled_wrapper_roles_covered'],
      'decode_reads',[row['transfers_inside_step']['host_read_calls'] for row in r['steps'][1:]],
      'retrieval_reads',r['output_retrieval']['host_read_calls'],
      'new_graphs',[row['new_graphs'] for row in r['steps']],flush=True)
assert r['status']=='pass' and r['generated_tokens']==[15,15,15,15]
assert all(row['transfers_inside_step']['host_read_calls']==0 for row in r['steps'][1:])
assert r['output_retrieval']['host_read_calls'] > 0
assert all(row['new_graphs'] == 0 for row in r['steps'][2:])
