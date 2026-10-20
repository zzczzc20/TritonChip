import hashlib,json,sys
from pathlib import Path
from tnca.autotune import GraphBudget,run_measurement
budget=GraphBudget()
base=Path('/tmp/r40_board_tuning_supervised')
records=run_measurement([sys.executable,'/tmp/r40_board_tuning_worker.py',str(base.with_suffix('.records.json'))],budget=budget,result_path=base.with_suffix('.records.json'),log_path=base.with_suffix('.worker.log'))
report=dict(functional_only=True,measured_performance=False,budget_seconds=600,elapsed=budget.clock()-budget.started,budget_exceeded=budget.exceeded,records=records,worker_sha256=hashlib.sha256(Path('/tmp/r40_board_tuning_worker.py').read_bytes()).hexdigest())
base.with_suffix('.report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
