"""TNCA runtime backends that launch into the native SystemC model.

    import tlm_backend; tlm_backend.register()

registers two names in the runtime's backend table (python/tnca/backend.py)
from outside the product tree:

    systemc-tlm        per-program launches (Session's serial-grid path, the
                       same launch sequence the sim-file RTL simulator gets)
    systemc-tlm-grid   physical-grid launches (Session's physical_grid path,
                       the sequence fpga-u280-multicore gets: one command per
                       grid, programs dispatched to the model's cores)

One model process per backend instance, i.e. per Session and graph: upload,
download and launch are requests to the long-lived `tnca-chip --serve`
process over two private pipes. The model contains no RTL simulator. Its
cycle counts are UNCALIBRATED predictions; nothing here is board or RTL
evidence, and a model refusal surfaces as a lost device, never as success.

Environment: TNCA_TLM_BINARY (tnca-chip), TNCA_TLM_CONFIG (default
configs/r40-three-core.cfg), TNCA_TLM_OUT (directory for each model
process's outputs; default a temporary directory), TNCA_TLM_MEMORY_BYTES
(model memory window), TNCA_TLM_MONITOR (off|aggregate|trace). The board's
runtime memory controls TNCA_RTL_MEM_PORTS / TNCA_RTL_OUTSTANDING are mapped
onto the model's active outlets and outstanding limit when set.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time

HERE = Path(__file__).resolve().parent
NAMES = ('systemc-tlm', 'systemc-tlm-grid')
EVIDENCE = 'model-prediction-uncalibrated'
MAX_PROGRAMS = 16777216


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _config_values(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        line = line.split('#')[0].strip()
        if '=' in line:
            k, v = line.split('=', 1)
            values[k.strip()] = v.strip()
    return values


def make_backend_class():
    from tnca import backend as _backend
    from tnca import transport as T
    from tnca.errors import DeviceLost, TncaError

    class SystemCModelBackend(_backend.Backend):
        state = _backend.REFERENCE_AVAILABLE
        launch_is_idempotent = False
        instances = []

        def __init__(self, bench=None, *, physical=False, workdir=None, deadline_cycles=2_000_000,
                     host_timeout=600.0, trace=False, epoch=0, binary=None, config=None, out=None, **kwargs):
            self.name = 'systemc-tlm-grid' if physical else 'systemc-tlm'
            self.physical = physical
            self.deadline_cycles = int(deadline_cycles)
            self.host_timeout = host_timeout
            self.epoch = epoch
            self.binary = Path(binary or os.environ.get('TNCA_TLM_BINARY') or 'tnca-chip')
            self.base_config = Path(config or os.environ.get('TNCA_TLM_CONFIG') or HERE/'configs/r40-three-core.cfg')
            root = out or os.environ.get('TNCA_TLM_OUT')
            if root:
                Path(root).mkdir(parents=True, exist_ok=True)
                self.dir = Path(tempfile.mkdtemp(prefix=f'model-{os.getpid()}-', dir=root))
            else:
                self.dir = Path(tempfile.mkdtemp(prefix='tnca-tlm-'))
            self.proc = None
            self.seq = 0
            self.launches = []
            self._io = threading.RLock()
            self.memory_bytes = None
            self.core_count = None
            self.host_seconds = dict(upload=0.0, download=0.0, launch=0.0)
            SystemCModelBackend.instances.append(self)

        # ---- lifetime --------------------------------------------------
        def _write_config(self):
            values = _config_values(self.base_config)
            if os.environ.get('TNCA_TLM_MEMORY_BYTES'):
                values['memory_bytes'] = str(int(os.environ['TNCA_TLM_MEMORY_BYTES'], 0))
            if os.environ.get('TNCA_TLM_MONITOR'):
                pass  # passed through the environment to the process
            ports = os.environ.get('TNCA_RTL_MEM_PORTS')
            if ports:
                values['active_outlets'] = '1' if int(ports) == 1 else values['outlets']
            outstanding = os.environ.get('TNCA_RTL_OUTSTANDING')
            if outstanding:
                values['outstanding'] = str(int(outstanding))
            path = self.dir/'model.cfg'
            path.write_text(''.join(f'{k}={v}\n' for k, v in values.items()))
            return path, values

        def start(self):
            from tnca._child_exec import child_argv
            if self.proc is not None:
                raise TncaError('this model backend already has a process', code='SESSION_BUSY')
            config, values = self._write_config()
            to_model_r, to_model_w = os.pipe()
            from_model_r, from_model_w = os.pipe()
            env = dict(os.environ, SYSTEMC_DISABLE_COPYRIGHT_MESSAGE='1',
                       TNCA_TLM_MONITOR=os.environ.get('TNCA_TLM_MONITOR', 'aggregate'))
            self.stdout = (self.dir/'stdout.log').open('w')
            self.stderr = (self.dir/'stderr.log').open('w')
            self.proc = subprocess.Popen(child_argv([str(self.binary), '--serve', str(config), str(self.dir),
                                          str(to_model_r), str(from_model_w)]),
                                         pass_fds=(to_model_r, from_model_w), env=env,
                                         stdout=self.stdout, stderr=self.stderr, cwd=self.dir)
            os.close(to_model_r)
            os.close(from_model_w)
            self.wfile = os.fdopen(to_model_w, 'wb', buffering=0)
            self.rfile = os.fdopen(from_model_r, 'rb', buffering=1 << 20)
            ready = self._line()
            fields = ready.split()
            if len(fields) != 4 or fields[0] != 'READY':
                self._lost(f'model did not start: {ready!r}')
            self.core_count = int(fields[1])
            self.memory_base = int(fields[2])
            self.memory_bytes = int(fields[3])
            if self.memory_base != 0:
                self._lost('the runtime addresses device memory from 0; model memory_base must be 0')
            self.identity = dict(backend=self.name, binary=str(self.binary.resolve()), binary_sha256=_sha(self.binary),
                                 config=values, config_sha256=_sha(config), model_dir=str(self.dir),
                                 calibration='not-measured', execution='native SystemC/TLM-2.0 model, no RTL',
                                 dram_path=os.environ.get('TNCA_DRAM_PATH') or ('slow' if values.get('dram_path') == '1' else 'fast'))
            mapping=os.environ.get('TNCA_DRAM_MAP')
            self.identity['address_map']=(dict(domain='axi-bo-physical-pc',path=str(Path(mapping).resolve()),sha256=_sha(mapping))
                                          if mapping else dict(domain='synthetic-logical'))
            self.identity['bank_row_mapping']='unverified'
            self.timed = True
            if os.environ.get('TNCA_TLM_UNTIMED_COLD') == '1':
                # The cold call runs untimed (its cycles are not a measurement);
                # run_graphs.py turns timing on before the steady calls.
                self.set_timing(False)
            self.identity['untimed_cold_call'] = not self.timed
            (self.dir/'identity.json').write_text(json.dumps(self.identity, indent=2) + '\n')
            return self

        def set_timing(self, on):
            with self._io:
                self.wfile.write(f'T {1 if on else 0}\n'.encode())
                if self._line() != 'OK':
                    self._lost('timing switch not acknowledged')
            self.timed = bool(on)

        def alive(self):
            return self.proc is not None and self.proc.poll() is None

        def stop(self):
            if self.proc is None:
                return
            try:
                if self.alive():
                    with self._io:
                        try:
                            self.counters = self.counters_snapshot()
                            (self.dir/'counters.json').write_text(json.dumps(self.counters, indent=1, sort_keys=True) + '\n')
                        except Exception:                    # noqa: BLE001
                            pass
                        self.wfile.write(b'X\n')
                        self._line()
                self.proc.wait(timeout=60)
            except Exception:                                # noqa: BLE001
                self.proc.kill()
                self.proc.wait(timeout=10)
            finally:
                (self.dir/'launches.json').write_text(json.dumps(dict(
                    launches=self.launches, host_seconds=self.host_seconds,
                    returncode=self.proc.returncode), indent=1) + '\n')
                for f in ('wfile', 'rfile', 'stdout', 'stderr'):
                    try:
                        getattr(self, f).close()
                    except Exception:                        # noqa: BLE001
                        pass
                self.proc = None

        # ---- protocol --------------------------------------------------
        def _lost(self, why):
            tail = ''
            try:
                self.stderr.flush()
                tail = (self.dir/'stderr.log').read_text(errors='replace')[-2000:]
            except Exception:                                # noqa: BLE001
                pass
            raise DeviceLost(f'SystemC model: {why}', detail={'model_dir': str(self.dir), 'stderr': tail},
                             remedy='the model refused or failed; see its stderr. Nothing is substituted.')

        def _line(self):
            raw = self.rfile.readline()
            if not raw:
                self.proc.wait(timeout=30)
                self._lost(f'process ended (returncode {self.proc.returncode})')
            text = raw.decode().rstrip('\n')
            if text.startswith('ERR '):
                self._lost(text[4:])
            return text

        def _reply(self, status, r0=0, r1=0):
            self.seq += 1
            return T.Reply(self.seq, status, r0, r1)

        # ---- operations ------------------------------------------------
        def upload(self, addr, data):
            _backend.check_transfer(addr, len(data), what='upload')
            t = time.perf_counter()
            with self._io:
                self.wfile.write(f'W {int(addr)} {len(data)}\n'.encode())
                self.wfile.write(bytes(data))
                if self._line() != 'OK':
                    self._lost('upload not acknowledged')
            self.host_seconds['upload'] += time.perf_counter() - t
            return self._reply(T.ST_OK)

        def download(self, addr, nbytes):
            _backend.check_transfer(addr, nbytes, what='download')
            t = time.perf_counter()
            with self._io:
                self.wfile.write(f'R {int(addr)} {int(nbytes)}\n'.encode())
                head = self._line().split()
                if head != ['OK', str(nbytes)]:
                    self._lost(f'malformed download reply {head}')
                data = self.rfile.read(nbytes)
                if len(data) != nbytes:
                    self._lost('short download payload')
            self.host_seconds['download'] += time.perf_counter() - t
            return self._reply(T.ST_OK), data

        def _run(self, command, record):
            t = time.perf_counter()
            with self._io:
                self.wfile.write((command + '\n').encode())
                fields = self._line().split()
            host = time.perf_counter() - t
            self.host_seconds['launch'] += host
            if len(fields) != 7 or fields[0] != 'DONE':
                self._lost(f'malformed launch reply {fields}')
            status, code, pc, start, end, retired = (int(v) for v in fields[1:])
            record.update(status=status, fault_code=code, fault_pc=pc, start_cycle=start, end_cycle=end,
                          cycles=end - start, retired=retired, host_s=round(host, 6))
            self.launches.append(record)
            return status, code, pc, start, end, retired

        def launch(self, pc, arg_base, role_mask=1, pid=(0, 0, 0), npid=(1, 1, 1), timeout=None):
            px, py, pz = (int(v) for v in pid)
            nx, ny, nz = (int(v) for v in npid)
            status, code, fpc, start, end, retired = self._run(
                f'L {int(pc)} {int(arg_base)} {int(role_mask)} {px} {py} {pz} {nx} {ny} {nz} {self.deadline_cycles}',
                dict(kind='program', pc=int(pc), role_mask=int(role_mask), pid=[px, py, pz], npid=[nx, ny, nz]))
            if status == 1:
                return self._reply(T.ST_FAULT, code, fpc)
            if status == 2:
                return self._reply(T.ST_TIMEOUT, end - start, 0)
            reply = self._reply(T.ST_OK, end - start, retired)
            reply.device_timing = None
            return reply

        def physical_grid_capability(self):
            return (self.core_count, MAX_PROGRAMS) if self.physical else None

        def physical_chunk_capability(self):
            return 4 if self.physical else None

        def memory_port_capability(self):
            return None

        def large_grid_limit(self):
            return MAX_PROGRAMS

        def launch_grid(self, pc, arg_base, grid, *, arg_stride=0, active_cores=None, role_mask=1,
                        group_width=1, timeout=None):
            if not self.physical:
                raise ValueError('this binding is per-program; use systemc-tlm-grid for physical grids')
            gx, gy, gz = (int(v) for v in grid)
            cores = (1 << self.core_count) - 1 if active_cores is None else int(active_cores)
            if gx * gy * gz == 0:
                return self._reply(T.ST_OK, 0, 0)
            status, code, fpc, start, end, retired = self._run(
                f'G {int(pc)} {int(arg_base)} {int(role_mask)} {gx} {gy} {gz} {cores} {int(arg_stride)} '
                f'{int(group_width)} {self.deadline_cycles}',
                dict(kind='grid', pc=int(pc), role_mask=int(role_mask), grid=[gx, gy, gz], cores=cores,
                     arg_stride=int(arg_stride), group_width=int(group_width)))
            if status == 1:
                return self._reply(T.ST_FAULT, code, fpc)
            if status == 2:
                return self._reply(T.ST_TIMEOUT, 0, 0)
            cycles = end - start
            reply = self._reply(T.ST_OK, cycles & 0xFFFFFFFF, cycles >> 32)
            reply.device_timing = None
            return reply

        def quiesce(self):
            # Every model command completes before its reply: nothing is in flight.
            return self._reply(T.ST_OK)

        def ping(self):
            return self._reply(T.ST_OK if self.alive() else T.ST_REFUSED)

        def counters_snapshot(self):
            with self._io:
                self.wfile.write(b'C\n')
                head = self._line().split()
                if len(head) != 2 or head[0] != 'OK':
                    self._lost(f'malformed counter reply {head}')
                return json.loads(self.rfile.read(int(head[1])))

        def capabilities(self):
            out = super().capabilities()
            out.update(physical_grid=self.physical, memory_bytes=self.memory_bytes,
                       reaches='tools/performance_harness/systemc tnca-chip --serve (no RTL)',
                       evidence=EVIDENCE, calibration='not-measured')
            return out

    return SystemCModelBackend


_CLASS = None


def backend_class():
    global _CLASS
    if _CLASS is None:
        _CLASS = make_backend_class()
    return _CLASS


def register():
    """Add both names to the runtime's registry (idempotent)."""
    from tnca import backend as _backend
    cls = backend_class()
    what = ('native SystemC/TLM-2.0 performance model (no RTL in its execution path); cycle counts are '
            'uncalibrated predictions, not measurements')
    _backend.register('systemc-tlm', lambda **kw: cls(physical=False, **kw), state=_backend.REFERENCE_AVAILABLE,
                      needs_elaborator=False, what='per-program launches: ' + what)
    _backend.register('systemc-tlm-grid', lambda **kw: cls(physical=True, **kw), state=_backend.REFERENCE_AVAILABLE,
                      needs_elaborator=False, physical_grid=True, what='physical-grid launches: ' + what)
    return cls
