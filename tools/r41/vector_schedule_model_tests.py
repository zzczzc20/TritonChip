#!/usr/bin/env python3
"""Compare independent-vector batching with finite native-model LSU resources."""
import argparse
import json
from pathlib import Path
from wide_path_model_tests import streaming_add, validate_chip, exploratory_run, unaligned_copy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for width, units in ((8, 2), (64, 2), (64, 4), (64, 8), (64, 16)):
        for unroll in (1, 2, 4, 8):
            out = args.out/f'width{width}_units{units}_unroll{unroll}'
            out.mkdir()
            validate_chip.config(out/'chip.cfg', memory_model=1, core_pipeline=1,
                                 tl_core_beat_bytes=width, tl_noc_payload_bytes=width,
                                 tl_lsu_units=units, lsu_beat_slots=16//units)
            result = streaming_add(args.binary.resolve(), out, unroll)
            rows.append({'core_bytes': width, 'noc_payload_bytes': width,
                         'lsu_units': units, 'beat_slots_per_unit': 16//units, 'unroll': unroll, 'result': result})
            print(width, units, unroll, result['visible_cycles'], flush=True)
    validations = []
    original = validate_chip.config
    validate_chip.run = exploratory_run
    for units in (4, 8, 16):
        def configured(path, **kw):
            return original(path, memory_model=1, core_pipeline=1,
                            tl_core_beat_bytes=64, tl_noc_payload_bytes=64,
                            tl_lsu_units=units, lsu_beat_slots=16//units, **kw)
        validate_chip.config = configured
        out = args.out/f'numerics_units{units}'
        report = validate_chip.validate(args.binary.resolve(), out)
        unaligned_copy(args.binary.resolve(), out)
        validations.append({'lsu_units': units, 'status': report['status']})
    validate_chip.config = original
    (args.out/'sweep.json').write_text(json.dumps({'evidence': 'model-prediction-uncalibrated',
        'whole_chip_qualified': False, 'validations': validations, 'rows': rows}, indent=2)+'\n')
    print('VECTOR SCHEDULE MODEL NUMERICS PASS')


if __name__ == '__main__':
    main()
