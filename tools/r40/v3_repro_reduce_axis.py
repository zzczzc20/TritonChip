"""Repro (R40-03 v3 audit finding): a standard Torch reduction over a non-innermost axis
fails to lower on the native tnca device at the default block cap.

    python3 tools/r40/v3_repro_reduce_axis.py ROOT {tnca|tnca-nomix|cpu|cuda} [width]

Observed on the delivery tree bc5a778e (and on 80dd1a82), torch 2.14.0, GB10:
  tnca        t.t().sum(-1), t.t().amax(-1), t.sum(0), t.t().contiguous().sum(-1):
              InductorError LoweringException AssertionError
              'Expected buf.data.get_reduction_type() is not None'
              (torch/_inductor/ir.py, split reduction under config.triton.mix_order_reduction)
  tnca-nomix  the same graphs with torch._inductor.config.triton.mix_order_reduction=False:
              t_sum, t_amax, dim0_sum PASS on the RTL device
  cpu, cuda   stock Inductor on the same torch: all PASS
Run with Icarus on PATH (source ~/tc_env/env.sh) and the torch environment.
"""
import os
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
mode = sys.argv[2]
width = sys.argv[3] if len(sys.argv) > 3 else '16'
os.environ['TNCA_MAX_BLOCK'] = width
os.environ.setdefault('TNCA_PROFILE', 'core-vrf-sync-v5')
import torch  # noqa: E402

cases = [('t_sum', lambda t: t.t().sum(-1), (127, 33)),
         ('t_amax', lambda t: t.t().amax(-1), (65, 17)),
         ('dim0_sum', lambda t: t.sum(0), (127, 33)),
         ('t_contig_sum', lambda t: t.t().contiguous().sum(-1), (127, 33))]
if mode.startswith('tnca'):
    sys.path[:0] = [str(root), str(root / 'python'), str(root / 'compiler/python'), str(root / 'tools/asm')]
    import triton  # noqa: F401
    import tnca.torch_device as D
    from tnca import device
    device.prepare(os.environ['TNCA_PROFILE'])
    D.register()
    if mode == 'tnca-nomix':
        import torch._inductor.config as cfg
        cfg.triton.mix_order_reduction = False
    dev = 'tnca'
else:
    dev = mode
import torch._inductor.config as cfg  # noqa: E402
print('mode', mode, 'width', width, 'torch', torch.__version__,
      'mix_order_reduction', cfg.triton.mix_order_reduction, flush=True)
torch.manual_seed(0)
for name, fn, shape in cases:
    torch._dynamo.reset()
    x = torch.randn(*shape)
    try:
        out = torch.compile(fn, fullgraph=True)(x.to(dev))
        ok = torch.allclose(out.cpu(), fn(x), rtol=1e-4, atol=1e-4)
        print(f'{name}: {"PASS" if ok else "WRONG"}', flush=True)
    except Exception as e:  # noqa: BLE001
        print(f'{name}: FAIL {type(e).__name__}: {str(e).splitlines()[0][:200]}', flush=True)
