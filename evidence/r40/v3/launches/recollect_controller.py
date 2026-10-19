"""Re-collect the complete original functional evidence on the v3 delivery tree.

Product tree files:bc5a778e... (doc/r40_v3/identity_amendment.json), image a88ebdf5,
90 MHz, 4 GiB. Waits until the v3 campaign releases the card. Same stages as the
80dd delivery (tnca_delivery_probe_0958db2_4gib + tnca_v3_functional), unchanged
points and tolerances. Stops at the first failed stage and preserves its output.
"""
from pathlib import Path
import os, json, datetime, subprocess, traceback, sys, time
h = Path.home(); p = Path(__file__).parent; s = h/'tnca_src_r40_v3x'
tree = 'files:bc5a778ebc5f4afebf6c1a9fbf3b2d8717cb02e118134ebbb73fb5a927fc8183'
sha = 'a88ebdf59cfbbb2001ebd5b1b2c739c5d8cd33931f1e9202f18cac07c4c83083'
image = h/'tnca_images/r40_attempt03_90mhz_a88ebdf5/tnca_u280.xclbin'
clock = h/'tnca_build/r40_attempt03_conditional_90mhz/qualified_clock.json'
out = h/'tnca_runs_r40_v3x'; python = str(h/'tc_env/torchenv/bin/python')
config = dict(xclbin=str(image), image_profile='core-vrf-sync-v5', core_count=3, memory_ports=3,
              hbm_dma=True, hbm_bytes=4 << 30, device_clock=True)
env = {k: v for k, v in os.environ.items() if not k.startswith('TNCA_')}
env.update(MAX_JOBS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='',
           PYTHONFAULTHANDLER='1', R40_CONTRACT='v2',
           TORCH_EXTENSIONS_DIR=str(h/'.cache/tnca/r40_conditional_6bd07b7_extensions'),
           PYTHONPATH=os.pathsep.join(str(s/d) for d in ('', 'python', 'compiler/python', 'tools/asm', 'tools/r40')),
           TNCA_PROFILE='core-vrf-sync-v5', TNCA_BACKEND='fpga-u280-multicore',
           TNCA_BACKEND_ARGS=json.dumps(config), TNCA_OPERATING_CLOCK_EVIDENCE=str(clock),
           TNCA_ACTIVE_CORES='3', TNCA_TORCH_DEVICE='native', TNCA_TRANSFER_POLICY='resident',
           TNCA_DEADLINE='24000000000', TNCA_HOST_TIMEOUT='7200')
state = dict(status='waiting-for-card', software_tree=tree, image_sha256=sha, clock_hz=90000000,
             backend_args=config, completed=[], scope='Complete original functional re-collection on the v3 delivery tree')
def save():
    state['observed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    q = p/'state.tmp'; q.write_text(json.dumps(state, indent=2)+'\n'); q.replace(p/'state.json')
save()
try:
    gate = Path(os.environ.get('V3_WAIT_STATE', str(h/'tnca_v3_campaign/run/state.json')))
    while True:
        try:
            if json.loads(gate.read_text()).get('status') == 'complete':
                break
        except (OSError, ValueError):
            pass
        time.sleep(60)
    while subprocess.run(['pgrep', '-f', 'tnca_src_r40_v3x/tools/r40/v3_campaign.py'], capture_output=True).returncode == 0:
        time.sleep(30)
    sys.path[:0] = [str(s), str(s/'python'), str(s/'tools/r40')]
    from archbench.provenance import software_tree
    assert software_tree(refresh=True) == tree
    assert Path('/sys/bus/pci/devices/0000:04:00.1/xclbinuuid').read_text().strip() == '1bada9db-78f3-5bc8-7ba6-cce062e7ade8'
    assert Path('/sys/bus/pci/devices/0000:04:00.0/clock_freq_counter.m.28311552/clock_counter_freqs').read_text().split()[0] == '90'
    common = ['--xclbin', str(image), '--image-sha256', sha, '--software-tree-sha256', tree,
              '--operating-clock-evidence', str(clock)]
    stages = [
        ('actual-window-and-pc-boundaries', [python, '-u', str(s/'tools/r40/hbm_board.py'), '--only', 'window', 'samples',
                                             '--json', str(p/'window_samples.json')]),
        ('full-original-functional-delivery', [python, '-u', str(s/'tools/r40/conditional_delivery.py'), *common,
                                               '--bucket', '--hbm-bytes', str(4 << 30), '--out', str(out)]),
        ('full-bucket-model-capability-audit', [python, str(s/'tools/r40/bucket_model_acceptance.py'),
            '--report', str(out/'model/report.json'), '--source-root', str(s), '--software-tree', tree,
            '--image-sha256', sha, '--xclbin', str(image), '--clock-evidence', str(clock),
            '--hbm-bytes', str(4 << 30), '--out', str(p/'bucket_model_capability.json'), '--evidence']),
        ('physical-multicore-capability-audit', [python, str(s/'tools/r40/multicore_board_acceptance.py'),
            '--native', str(out/'native'), '--checkpoint', str(out/'checkpoint/report.json'), *common,
            '--out', str(p/'multicore_capability.json'), '--evidence']),
        ('final-native-inventory-evidence', [python, str(s/'tools/r40/native_inventory.py'), '--native', str(out/'native'), '--evidence']),
        ('complete-no-cuda-host-conformance', [python, '-u', str(s/'tools/r40/device_tests.py'), '--evidence']),
    ]
    points = json.loads((s/'spec/r40/suites/large.json').read_text())['points']
    stages.append(('original-large24', [python, '-u', '-m', 'archbench.run', '--platform', 'tnca',
        '--profile', 'core-vrf-sync-v5', '--backend', 'fpga-u280-multicore', '--backend-args', json.dumps(config),
        '--point', *[x['id'] for x in points], '--jobs', '1', '--warmup', '1', '--timed', '1',
        '--bundle-dir', str(p/'large24_bundles'), '--out', str(p/'large24.jsonl')]))
    for i, point in enumerate(json.loads((s/'spec/r40/suites/R40-02.graphs3.json').read_text())['points']):
        stages.append((f'graphs24-point{i:02d}', [python, '-u', str(s/'tools/r40/frozen_residency_tests.py'),
                       '--point', point['id'], '--steps', '2', '--source-tree', tree]))
    (p/'commands.json').write_text(json.dumps(stages, indent=2)+'\n'); state['status'] = 'running'; save()
    for name, cmd in stages:
        assert software_tree(refresh=True) == tree
        state['current'] = name; save()
        with (p/(name+'.log')).open('x') as log:
            q = subprocess.Popen(cmd, cwd=s, env=env, stdout=log, stderr=subprocess.STDOUT)
            state['child_pid'] = q.pid; save(); code = q.wait()
        state['completed'].append(dict(name=name, exit_code=code)); save()
        assert code == 0, name+' failed; preserve stage'
        if name == 'original-large24':
            rows = [json.loads(l) for l in (p/'large24.jsonl').read_text().splitlines() if l.strip()]
            assert len(rows) == 24 and {r['point_id'] for r in rows} == {x['id'] for x in points}
            assert all(r['status'] == 'pass' and not r.get('fallbacks') and r['correctness']['max_rel_err'] <= r['correctness']['tol'] for r in rows), 'original large point failed'
    state['status'] = 'complete-review-required'; save()
except Exception:
    state.update(status='failed-preserved', error=traceback.format_exc()); save(); raise
