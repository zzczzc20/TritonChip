#!/usr/bin/env python3
"""Register and queue the missing Q35 scripted-chat baseline; no image load."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

WORK = Path.home() / 'r41run3'
OLD = WORK / 'runtime_activation_equal_20261031'
ROOT = WORK / 'runtime_q35_script_baseline_20261101'
OUT = WORK / 'chat_q35_script_baseline_20261101'
DIRECTIVE = WORK / 'author_directive_20261101.txt'
DIRECTIVE_SHA = '7adc268ec0a0e2baefd678f65585280a23a181fda8e0f05afed7044edd759e58'
NAME = 'r41-q35-script-baseline-20261101'
TEMPLATE = 'spec/r41/run3/board/r3-s2-qwen-chat-v3-vl-nohost-v5-script-20261031.json'
EXPECTED = {
    TEMPLATE: 'e725d8304e74603eed18dc74a7bde376778a3a3a618787dcd4856cc1fe93f9b8',
    'spec/r41/chat/script-text.txt': 'b046c5bf7e2d07c18afb7d12064cd2a117397a5905b295fb09d61da1906de897',
    'tools/models/chat.py': '7121408bfa7189aad2a80dcb2bccce0a9ddce656fdf6b7eb9f7de46feec601ed',
    'compiler/python/tnca_cc/frontend.py': 'a8ff866ea37cd70551d137b59261424a7a4b7df2f0b929cc11c123f79e993bfb',
}

def main():
    sys.path.insert(0, str(OLD / 'tools/r41/run2_board'))
    import q35_timeout_recovery as recovery
    if ROOT.exists() or OUT.exists():
        raise FileExistsError('new scripted-chat source or output already exists')
    if recovery.digest(DIRECTIVE) != DIRECTIVE_SHA:
        raise ValueError('author directive differs from the committed version')
    for path, sha in EXPECTED.items():
        if recovery.digest(OLD / path) != sha:
            raise ValueError('baseline source differs: ' + path)
    OUT.mkdir(); recovery.OUT = OUT
    try:
        subprocess.run(['cp','-a','--reflink=auto',str(OLD),str(ROOT)],check=True)
        for path, sha in EXPECTED.items():
            if recovery.digest(ROOT / path) != sha:
                raise ValueError('copied baseline differs: ' + path)
        target = ROOT / 'evidence/r41/AUTHOR_DIRECTIVE_20261031.md'
        target.unlink(missing_ok=True)
        shutil.copyfile(DIRECTIVE, target)
        spec = copy.deepcopy(json.loads((ROOT / TEMPLATE).read_text()))
        if spec['id'] != 'r3-s2-qwen-chat-v3-vl-nohost-v5-script-20261031' or len(spec['steps']) != 1:
            raise ValueError('unexpected script template')
        step = spec['steps'][0]; argv = step['argv']
        for flag, old, new in [('--model','vl','q35'),('--new-tokens','32','8'),('--out','{out}/vl-script','{out}/q35-script')]:
            if argv.count(flag) != 1 or argv[argv.index(flag)+1] != old:
                raise ValueError('unexpected template argument: ' + flag)
            argv[argv.index(flag)+1] = new
        step['id'] = 'q35-script'; step['outputs'] = ['{out}/q35-script/transcript.json']
        step['env']['TNCA_IEEE_DIV'] = '0'
        spec['id'] = NAME
        spec['hypothesis'] = 'Run-2 Qwen3.5 scripted conversation completes with device placement and no host numerical fallback.'
        spec['registration']['scope'] = ('Author-directed Q35 scripted chat baseline using the existing three-turn script, '
            '8 maximum returned token IDs per turn, default arithmetic and the already loaded image. '
            'No speedup is inferred from the shorter limit; apply the 2026-11-01 scoring decision separately.')
        spec['registration']['author_directive_sha256'] = DIRECTIVE_SHA
        manifest = ROOT / 'spec/r41/run3/board' / (NAME + '.json')
        with manifest.open('x') as f:
            json.dump(spec,f,indent=2); f.write('\n')
        reg = dict(schema='tnca-run2-stage-registration/1',id=NAME,
            image_identity=dict(sha256='791b8af28f2583e4b29cd4bb191d44e8692bed41b7132c493d86818cb0c7ee09',
                                uuid='ea1fa1af-cf1a-fda3-a5d2-e4b88a63abe3'),
            experiments=[dict(id=NAME,manifest=str(manifest.relative_to(ROOT)),sha256=recovery.digest(manifest),
                              budget_seconds=10800,checks=[dict(path='q35-script/transcript.json',field=['turns'],length_equals=3)])])
        regpath=OUT/'registration.json';regpath.write_text(json.dumps(reg,indent=2)+'\n')
        recovery.record('registered',registration_sha256=recovery.digest(regpath),directive_sha256=DIRECTIVE_SHA,
                        scope='8-token three-turn Q35 scripted baseline; speed comparison pending')
        recovery.ROOT=ROOT
        recovery.stage(ROOT,OUT/'script',regpath,OUT/'stage.log')
        transcript=OUT/'script'/NAME/'q35-script/transcript.json';data=json.loads(transcript.read_text())
        if len(data.get('turns',[])) != 3:
            raise ValueError('incomplete script')
        for row in data['turns']:
            report=row.get('report',{})
            if report.get('status')!='pass' or report.get('cpu_numerical_fallback') is not False:
                raise ValueError('script failed or CPU fallback was not excluded')
            if report.get('steady_decode_host_reads_detected') is not False:
                raise ValueError('steady decode host reads were not excluded')
        recovery.record('completed',transcript=str(transcript),sha256=recovery.digest(transcript),
                        scope='script execution; author content/first-token scoring still required')
    except BaseException as exc:
        recovery.record('stopped',error=f'{type(exc).__name__}: {exc}');raise

if __name__ == '__main__':main()
