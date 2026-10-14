#!/usr/bin/env python3
"""Freeze single-core image parameters from the common profile contract.

Generate a Tcl input for package_xo.tcl and a JSON audit manifest. The
packager uses TNCA_IMAGE_CONFIG to apply these parameters without editing
RTL defaults in a copied source tree.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parameters(profile):
    spec = json.loads((ROOT / 'spec/r1_profiles.json').read_text())['profiles'][profile]
    if spec['num_cores'] != 1:
        raise ValueError('single-core board wrapper cannot implement a multi-core profile')
    # Explicit zeros matter when migrating from a wrapper whose defaults
    # enable the banked v2 implementation.
    params = dict(USE_BANKED_VRF=0, VRF_BYPASS=0, FE_NOBUBBLE=0,
                  VRF_PREFETCH=0, RCTX_SWITCH=0, LSU_INFLIGHT=1,
                  FPU_OVERLAP=0, RSLICE=0, RPAIR=0, MTX_ABUF=0, HAS_TENSOR_XPOSE=0)
    params.update({k: v for k, v in spec['elaboration'].items() if k != 'PHYS_VREGS'})
    impl = spec.get('implementation', {})
    if impl.get('vector_register_file') == 'banked':
        params['USE_BANKED_VRF'] = 1
    params.update(impl.get('rtl_parameters', {}))
    return params


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--profile', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--hbm', action='store_true')
    ap.add_argument('--multicore-hbm', action='store_true')
    ap.add_argument('--memory-ports', type=int, choices=(1, 2, 3), default=3)
    ap.add_argument('--link-stages', type=int, default=2)
    ap.add_argument('--line-beats', type=int, choices=(0, 2, 4, 8, 16), default=0)
    ap.add_argument('--outstanding', type=int, default=8)
    ap.add_argument('--lines', type=int, default=8)
    ap.add_argument('--waiters', type=int, default=8)
    ap.add_argument('--prefetch', type=int, default=1)
    ap.add_argument('--local-pipeline', action='store_true',
                    help='queue local requests in the burst bridge before BRAM')
    ap.add_argument('--ram-cache', action='store_true',
                    help='store burst lines in synchronous dual-port RAM')
    args = ap.parse_args()
    params = parameters(args.profile)
    if args.multicore_hbm:
        if args.hbm or args.line_beats or args.local_pipeline or args.ram_cache:
            ap.error('multicore HBM uses the native line adapter, not the single-core bridge')
        if args.link_stages < 1 or not 1 <= args.outstanding <= 16:
            ap.error('multicore HBM needs positive pipeline stages and 1..16 outstanding slots')
        params.update(MEM_PORTS=args.memory_ports, AXI_DATA_W=256, AXI_OUT_N=args.outstanding,
                      AXI_ID_W=max(1, (args.outstanding - 1).bit_length()),
                      NOC_LINK_STAGES=args.link_stages, DISPATCH_STAGES=args.link_stages,
                      MEM_HOME_STAGES=args.link_stages, LINK_STAGES=args.link_stages)
    elif args.hbm:
        if not (1 <= args.outstanding <= 16 and 1 <= args.lines <= 16
                and args.waiters >= 1 and args.prefetch >= 0):
            ap.error('HBM slot counts must fit the bridge and its 5-bit AXI IDs')
        params.update(AXI_LINE=args.line_beats, AXI_OUT_N=args.outstanding,
                      AXI_LB_N=args.lines, AXI_WAIT_N=args.waiters, AXI_PF=args.prefetch,
                      AXI_LOCAL_PIPE=int(args.local_pipeline), AXI_RAM_CACHE=int(args.ram_cache))
        if args.local_pipeline and not args.line_beats:
            ap.error('--local-pipeline requires a burst bridge (--line-beats)')
        if args.ram_cache and not args.line_beats:
            ap.error('--ram-cache requires a burst bridge (--line-beats)')
    elif args.line_beats:
        ap.error('--line-beats requires --hbm')
    elif args.local_pipeline:
        ap.error('--local-pipeline requires --hbm')
    elif args.ram_cache:
        ap.error('--ram-cache requires --hbm')
    manifest = dict(profile=args.profile, variant='multicore-hbm' if args.multicore_hbm else ('hbm' if args.hbm else 'onchip'),
                    parameters=params,
                    profiles_sha256=hashlib.sha256((ROOT / 'spec/r1_profiles.json').read_bytes()).hexdigest())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text('set TNCA_IMAGE_PARAMS [dict create ' +
                        ' '.join(f'{k} {v}' for k, v in sorted(params.items())) + ']\n')
    args.out.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
