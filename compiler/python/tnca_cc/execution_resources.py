"""Static execution resource facts shared by compilation and model analysis.

This describes the emitted program. It is neither a numerical oracle nor a
cycle predictor. Dynamic masks, loop counts and overlap still require native
execution; a static command count must not be presented as executed traffic.

Three steps fill the sidecar, all bookkeeping that never changes code:

  gemm_sites(imp)   the logical `tl.dot` sites of the TTIR, with the problem
                    extents the TTIR states as constants (loop bounds and
                    `index < bound` masks); anything else stays unknown.
  describe(kernel)  the emitted IR: per-role command counts, loops,
                    dependencies, transfers and accumulator traffic.
  finalize(plan)    once the code is assembled: physical fragments,
                    valid/padding MACs, resource peaks against the profile's
                    limits, and every selection or refusal reason.

The compiler's admission decisions and the TNCA tuner read their resource
numbers from the finalized plan (`admission`, `tuning_resources`).
"""
from collections import Counter

SCHEMA = 'tnca-execution-resources/2'
MMA_ROWS = MMA_COLS = 16
# A command naming a TensorMap (the transfer engine, ISA sec 9E) versus one
# naming a pointer (the LSU). The IR kinds are the emitted commands.
DESCRIPTOR_TRANSFERS = ('dma_load', 'dma_store')
POINTER_TRANSFERS = ('load', 'store', 'gather', 'scatter', 'sload', 'sstore',
                     'atomic', 'atomic_cas')


def _cdiv(a, b):
    return -(-a // b)


def describe(kernel):
    roles = []
    for role in kernel.roles:
        ops = role.region.ops
        labels = {op.attrs['name']: i for i, op in enumerate(ops) if op.kind == 'label'}
        loops = []
        for end, op in enumerate(ops):
            start = labels.get(op.attrs.get('target'))
            if op.kind in ('branch', 'branch_nz') and start is not None and start < end:
                body = Counter(item.kind for item in ops[start:end + 1])
                loops.append(dict(label=op.attrs['target'], start=start, end=end,
                                  static_mma_commands=body['mma_async'],
                                  static_acc_reads=body['acc_read'],
                                  static_acc_writes=body['acc_write'],
                                  static_acc_zeros=body['acc_zero'],
                                  static_descriptor_transfers=sum(body[k] for k in DESCRIPTOR_TRANSFERS),
                                  static_pointer_transfers=sum(body[k] for k in POINTER_TRANSFERS)))
        counts = Counter(op.kind for op in ops)
        shared = set(kernel.windows + kernel.accumulators)
        uses = {v: [] for v in shared}
        dependencies = []
        for i, op in enumerate(ops):
            for v in op.operands + op.results:
                if v in uses:
                    uses[v].append(i)
            if op.kind in ('mma_async', 'dma_load', 'dma_store',
                           'barrier_arm', 'barrier_wait', 'event_wait'):
                dependencies.append(dict(position=i, kind=op.kind,
                                         tokens=[v.name for v in op.operands + op.results
                                                 if v.kind in ('token', 'event', 'barrier')]))
        # The innermost loops that issue matrix commands: the K loops.
        mma_loops = [lp for lp in loops if lp['static_mma_commands'] and not any(
            other is not lp and other['static_mma_commands'] and
            lp['start'] <= other['start'] and other['end'] <= lp['end'] for other in loops)]
        in_k = Counter()
        for lp in mma_loops:
            in_k.update(item.kind for item in ops[lp['start']:lp['end'] + 1])
        roles.append(dict(role=role.index, static_op_counts=dict(sorted(counts.items())),
                          loops=loops, command_dependencies=dependencies,
                          shared_use_intervals=[dict(name=v.name, kind=v.kind,
                                                     first=min(sites), last=max(sites))
                                                for v, sites in uses.items() if sites],
                          k_loops=[dict(label=lp['label'], start=lp['start'], end=lp['end'])
                                   for lp in mma_loops],
                          transfers=dict(
                              descriptor=sum(counts[k] for k in DESCRIPTOR_TRANSFERS),
                              pointer=sum(counts[k] for k in POINTER_TRANSFERS),
                              descriptor_in_k_loops=sum(in_k[k] for k in DESCRIPTOR_TRANSFERS),
                              pointer_in_k_loops=sum(in_k[k] for k in POINTER_TRANSFERS),
                              descriptor_snapshots=counts['desc_snapshot']),
                          commands=dict(transfer=sum(counts[k] for k in DESCRIPTOR_TRANSFERS),
                                        mma=counts['mma_async'],
                                        transfer_in_k_loops=sum(in_k[k] for k in DESCRIPTOR_TRANSFERS),
                                        mma_in_k_loops=in_k['mma_async'],
                                        barrier_waits=counts['barrier_wait'],
                                        dependencies=len(dependencies)),
                          accumulator_round_trips=dict(
                              k_loop_reads=in_k['acc_read'], k_loop_writes=in_k['acc_write'],
                              k_loop_zeros=in_k['acc_zero'],
                              outside_k_loop_reads=counts['acc_read'] - in_k['acc_read'],
                              outside_k_loop_writes=counts['acc_write'] - in_k['acc_write'])))
    return dict(schema=SCHEMA, scope='static emitted IR',
                dynamic_counts='require native execution',
                valid_and_padding_macs='per launch from constant TTIR extents (gemm); '
                                       'otherwise require runtime valid masks',
                overlap='not inferred from command admission',
                window_registers=sum(w.length for w in kernel.windows),
                roles=roles)


# ---------------------------------------------------------------------------
# TTIR: logical dot sites and their constant problem extents
# ---------------------------------------------------------------------------
_PRESERVING = ('tt.fp_to_fp', 'arith.extf', 'arith.truncf', 'ttg.convert_layout',
               'arith.bitcast', 'tt.bitcast')
_INDEX_ARITH = ('arith.addi', 'arith.subi', 'arith.muli', 'arith.remsi', 'arith.remui',
                'arith.divsi', 'arith.divui', 'arith.extsi', 'arith.extui', 'arith.trunci')
_CMPI = ['eq', 'ne', 'slt', 'sle', 'sgt', 'sge', 'ult', 'ule', 'ugt', 'uge']


class _Ttir:
    def __init__(self, imp):
        from .frontend import TTIRImporter, _parse_type
        self.imp, self.parse, self.constant_value = imp, _parse_type, TTIRImporter._constant_value
        self.ops = list(getattr(imp, '_all_ops', None) or [])
        self.users = getattr(imp, '_value_users', None) or {}
        self.defs, self.owner = {}, {}
        for op in self.ops:
            for i in range(op.get_num_results()):
                self.defs[op.get_result(i).id()] = op
            for r in range(op.get_num_regions()):
                self.owner[op.get_region(r).id()] = op

    def const(self, value, depth=0):
        """An integer constant (scalar or splat), through splat/broadcast."""
        vid = value.id()
        known = getattr(self.imp, '_const', {}).get(vid)
        if isinstance(known, int) and not isinstance(known, bool):
            return known
        op = self.defs.get(vid)
        if op is None or depth > 8:
            return None
        name = op.get_name()
        if name == 'arith.constant':
            k = self.constant_value(op)
            return k if isinstance(k, int) and not isinstance(k, bool) else None
        if name in ('tt.splat', 'tt.broadcast', 'tt.expand_dims', 'arith.extsi', 'arith.trunci'):
            return self.const(op.get_operand(0), depth + 1)
        return None

    def loops(self, op):
        """Enclosing scf.for operations, innermost first."""
        out = []
        try:
            region = op.get_block().get_parent()
        except Exception:                                    # noqa: BLE001
            return out
        seen = 0
        while region is not None and seen < 64:
            owner = self.owner.get(region.id())
            if owner is not None and owner.get_name() == 'scf.for':
                out.append(owner)
            try:
                region = region.get_parent_region()
            except Exception:                                # noqa: BLE001
                break
            seen += 1
        return out

    def trips(self, loop):
        lo, hi, step = (self.const(loop.get_operand(i)) for i in range(3))
        if None in (lo, hi, step) or step <= 0:
            return None
        return max(0, _cdiv(hi - lo, step))

    def axis(self, value, depth=0):
        """Which tile axis an index varies along: 0 rows, 1 columns,
        'uniform' for a constant or splat, None when unknown."""
        if depth > 16:
            return None
        if self.const(value) is not None:
            return 'uniform'
        op = self.defs.get(value.id())
        if op is None:
            return None
        name = op.get_name()
        if name == 'tt.splat':
            return 'uniform'
        if name == 'tt.broadcast':
            return self.axis(op.get_operand(0), depth + 1)
        if name == 'tt.expand_dims':
            shape, _ = self.parse(str(op.get_result(0).get_type()))
            if not isinstance(shape, tuple) or len(shape) != 2:
                return None
            # [R] -> [R, 1] varies down the rows; [C] -> [1, C] across.
            return 0 if op.get_int_attr('axis') == 1 else 1
        if name in _INDEX_ARITH:
            found = {self.axis(op.get_operand(i), depth + 1) for i in range(op.get_num_operands())}
            found.discard('uniform')
            if len(found) == 1 and None not in found:
                return found.pop()
            return 'uniform' if not found else None
        return None

    def bounds(self, mask, depth=0):
        """[(axis, kind, constant)] from conjunctions of `index < bound`, or
        None when any term is something else (so nothing is inferred from
        what was not understood).

        kind 'absolute': index < C. kind 'remaining': index < C - offset,
        Triton's `offs_k < K - k * BLOCK_K` for a loop-relative index."""
        if mask is None or depth > 16:
            return None
        op = self.defs.get(mask.id())
        if op is None:
            return None
        name = op.get_name()
        if name == 'tt.broadcast':
            return self.bounds(op.get_operand(0), depth + 1)
        if name == 'arith.andi':
            left = self.bounds(op.get_operand(0), depth + 1)
            right = self.bounds(op.get_operand(1), depth + 1)
            return None if left is None or right is None else left + right
        if name != 'arith.cmpi':
            return None
        p = op.get_int_attr('predicate')
        if p is None or not 0 <= p < len(_CMPI) or _CMPI[p] not in ('slt', 'ult'):
            return None
        axis = self.axis(op.get_operand(0))
        if axis not in (0, 1):
            return None
        rhs = op.get_operand(1)
        c = self.const(rhs)
        if c is not None:
            return [(axis, 'absolute', c)]
        node = self.defs.get(rhs.id())
        for _ in range(8):
            if node is None or node.get_name() not in ('tt.splat', 'tt.broadcast'):
                break
            node = self.defs.get(node.get_operand(0).id())
        if node is not None and node.get_name() == 'arith.subi':
            c = self.const(node.get_operand(0))
            if c is not None:
                return [(axis, 'remaining', c)]
        return None

    def source(self, value):
        """The memory operation an MMA operand was read by, through casts."""
        for _ in range(8):
            op = self.defs.get(value.id())
            if op is None:
                return None
            if op.get_name() in ('tt.load', 'tt.descriptor_load'):
                return op
            if op.get_name() not in _PRESERVING:
                return None
            value = op.get_operand(0)
        return None

    def descriptor_extents(self, desc):
        """Constant [rows, cols] of a tt.make_tensor_descriptor, or None."""
        op = self.defs.get(desc.id())
        if op is None or op.get_name() != 'tt.make_tensor_descriptor':
            return None
        shape, _ = self.parse(str(op.get_result(0).get_type()).replace('!tt.tensordesc<', 'tensor<'))
        rank = len(shape) if isinstance(shape, tuple) else 1
        dims = [self.const(op.get_operand(1 + i)) for i in range(rank)]
        return dims[-2:] if rank >= 2 and None not in dims[-2:] else None

    def operand_bounds(self, value):
        """Problem bounds an operand's load states: ('mask', [...]),
        ('extents', [rows, cols]) or ('unmasked', [])."""
        op = self.source(value)
        if op is None:
            return None
        if op.get_name() == 'tt.descriptor_load':
            ext = self.descriptor_extents(op.get_operand(0))
            return ('extents', ext) if ext else None
        mask = op.get_operand(1) if op.get_num_operands() > 1 else None
        return ('mask', self.bounds(mask)) if mask is not None else ('unmasked', [])

    def result_store(self, dot):
        """The store the dot's value reaches, through the loop carry and
        an elementwise epilogue."""
        shape = self.parse(str(dot.get_result(0).get_type()))[0]
        frontier, seen = [dot.get_result(0)], set()
        while frontier and len(seen) < 256:
            value = frontier.pop(0)
            if value.id() in seen:
                continue
            seen.add(value.id())
            for user, k in self.users.get(value.id(), []):
                name = user.get_name()
                if name in ('tt.store', 'tt.descriptor_store'):
                    if name == 'tt.store' and k == 1 or name == 'tt.descriptor_store':
                        return user
                    continue
                if name == 'scf.yield':
                    try:
                        loop = self.owner.get(user.get_block().get_parent().id())
                    except Exception:                        # noqa: BLE001
                        loop = None
                    if loop is not None and loop.get_name() == 'scf.for' and k < loop.get_num_results():
                        frontier.append(loop.get_result(k))
                    continue
                # Elementwise epilogue only: the same tile, value for value.
                if (name.startswith(('arith.', 'math.', 'tt.fp_to_fp')) and user.get_num_results() and
                        self.parse(str(user.get_result(0).get_type()))[0] == shape):
                    frontier.append(user.get_result(0))
        return None


def _extent(spec, axis, remaining=False):
    """The constant bound a load/store states along one tile axis."""
    if spec is None or spec[1] is None:
        return None
    kind, data = spec
    if kind == 'extents':
        return data[axis]
    kinds = ('absolute', 'remaining') if remaining else ('absolute',)
    values = {c for ax, k, c in data if ax == axis and k in kinds}
    return min(values) if values else None


def _unbounded(spec, axis):
    """True only when the access is known to state no bound on `axis`."""
    if spec is None or spec[1] is None or spec[0] == 'extents':
        return False
    return not any(ax == axis for ax, _k, _c in spec[1])


def gemm_sites(imp):
    """Logical `tl.dot` sites, their physical fragments and constant extents."""
    t = _Ttir(imp)
    retiles = dict(getattr(imp, '_mma_retiles', ()) or ())
    k_steps = set(getattr(imp.target, 'mma', {}).get('k_steps', ()))
    axes = sorted({op.get_int_attr('axis') or 0 for op in t.ops if op.get_name() == 'tt.get_program_id'})
    sites = []
    for index, op in enumerate(o for o in t.ops if o.get_name() == 'tt.dot'):
        a_shape, a_elem = t.parse(str(op.get_operand(0).get_type()))
        b_shape, b_elem = t.parse(str(op.get_operand(1).get_type()))
        if not (isinstance(a_shape, tuple) and isinstance(b_shape, tuple)):
            continue
        bm, bk = a_shape[-2:]
        bn = b_shape[-1]
        site = dict(site=index, logical_tile=[bm, bn, bk], dtype=a_elem)
        loops = t.loops(op)
        trips = [t.trips(lp) for lp in loops]
        site['k_loop_trips'] = trips[0] if loops else 1
        site['enclosing_loops'] = len(loops)
        if a_elem == 'f32' or a_elem != b_elem:
            site['engine'] = 'vector-fma' if a_elem == 'f32' else 'refused'
            sites.append(site)
            continue
        k_cmd = retiles.get(bk, max(16, bk))
        site.update(engine='mma', mma_k=k_cmd, mma_k_supported=f'k{k_cmd}' in k_steps,
                    physical_fragments=_cdiv(bm, MMA_ROWS) * _cdiv(bn, MMA_COLS))
        commands = site['physical_fragments'] * _cdiv(bk, k_cmd)
        physical = commands * MMA_ROWS * MMA_COLS * k_cmd
        logical = bm * bn * bk
        site['per_k_step'] = dict(mma_commands=commands, physical_macs=physical, logical_macs=logical,
                                  fragment_padding_macs=physical - logical)
        # Problem extents, only from constants the TTIR states.
        a_b, b_b = t.operand_bounds(op.get_operand(0)), t.operand_bounds(op.get_operand(1))
        store = t.result_store(op)
        out_b = None
        if store is not None and store.get_name() == 'tt.store':
            out_b = (('mask', t.bounds(store.get_operand(2))) if store.get_num_operands() > 2
                     else ('unmasked', []))
        elif store is not None:
            ext = t.descriptor_extents(store.get_operand(0))
            out_b = ('extents', ext) if ext else None
        m = _extent(out_b, 0) or _extent(a_b, 0)
        n = _extent(out_b, 1) or _extent(b_b, 1)
        k = _extent(a_b, 1, remaining=True) or _extent(b_b, 0, remaining=True)
        k_trips = site['k_loop_trips']
        if (k is None and _unbounded(a_b, 1) and _unbounded(b_b, 0) and
                isinstance(k_trips, int) and len(loops) <= 1):
            # Neither operand bounds K: every K index the loop loads is in
            # the problem by construction (Inductor's EVEN_K).
            k = k_trips * bk
        site['problem'] = dict(M=m, N=n, K=k,
                               source='constant TTIR loop bounds, load masks/extents and the result store mask',
                               program_axes=axes)
        if None in (m, n, k) or not isinstance(k_trips, int) or len(loops) > 1 or None in trips:
            site['macs'] = dict(status='unknown', reason='problem extents or K trip count not constant')
        else:
            programs = _cdiv(m, bm) * _cdiv(n, bn)
            physical_total = programs * k_trips * physical
            logical_total = programs * k_trips * logical
            valid = m * n * k
            site['macs'] = dict(
                status='static', scope=('per output matrix; other program axes (e.g. batch) multiply'
                                        if axes != [0] else 'per launch'),
                programs=programs, assumes_grid='one program per ceil(M/BM) x ceil(N/BN) output tile',
                mma_commands=programs * k_trips * commands,
                physical=physical_total, valid=valid, padding=physical_total - valid,
                tile_padding=logical_total - valid, fragment_padding=physical_total - logical_total)
        sites.append(site)
    return sites


def gemm_sites_or_error(imp):
    """gemm_sites, never failing a compile: the plan records the error."""
    try:
        return gemm_sites(imp)
    except Exception as error:                               # noqa: BLE001
        return [dict(status='analysis-error', error=f'{type(error).__name__}: {error}'[:300])]


# ---------------------------------------------------------------------------
# Finalization and the decisions that read the plan
# ---------------------------------------------------------------------------
def finalize(plan, *, measured, entries, code_limit=None):
    """Complete the sidecar once the code is assembled. Returns the plan."""
    plan['allocated_resources'] = measured
    plan['logical_gemm_tiles'] = entries.get('register_blocks', {}).get('gemm_tiles', [])
    plan['mma_retiles'] = entries.get('mma_retiles', [])
    plan['window_lifetime'] = entries.get('window_lifetime', {})
    plan['bank_allocation'] = entries.get('bank_allocation', {})
    plan['layout'] = {k: entries[k] for k in
                      ('opt_views_renamed', 'opt_views_materialised', 'opt_layout_upstream')
                      if k in entries}
    plan['selection_notes'] = {k: v for k, v in entries.items()
                               if k.endswith(('_fallback', '_refused'))}
    sites = entries.get('gemm_sites', [])
    roles = plan.get('roles', [])
    # Each dot site lowers once per program body: the static MMA commands
    # emitted must be the sum of the sites' per-step physical commands.
    emitted = sum(r['commands']['mma'] for r in roles if 'commands' in r)
    expected = sum(s['per_k_step']['mma_commands'] for s in sites if s.get('engine') == 'mma')
    plan['gemm'] = dict(sites=sites,
                        static_mma_commands=dict(expected_from_sites=expected, emitted=emitted,
                                                 match=expected == emitted))
    limits = measured.get('limits', {})
    plan['resources'] = dict(
        vrf_peak_registers=measured.get('vector_regs_used'), vrf_limit=limits.get('vector_regs'),
        window_registers=plan.get('window_registers'), windows=measured.get('windows_used'),
        accumulators=measured.get('accumulators_used'), accumulator_limit=limits.get('accumulators'),
        barriers=measured.get('barriers_used'), barrier_limit=limits.get('barriers'),
        descriptors=measured.get('descriptors_used'), descriptor_limit=limits.get('descriptors'),
        scalar_registers=measured.get('scalar_regs_used'),
        code_bytes=measured.get('code_bytes'), code_limit=code_limit,
        scratch_bytes=measured.get('scratch_bytes'))
    plan['accumulator_round_trips'] = _sum_roles(roles, 'accumulator_round_trips')
    plan['transfers'] = _sum_roles(roles, 'transfers')
    plan['commands'] = _sum_roles(roles, 'commands')
    decisions = []
    for key, value in sorted(entries.items()):
        if key.endswith('_refused'):
            decisions.append(dict(decision=key[:-len('_refused')], status='refused', reason=value))
        elif key.endswith('_fallback'):
            decisions.append(dict(decision=key[:-len('_fallback')], status='fallback', reason=value))
    if entries.get('predicate_spills'):
        decisions.append(dict(decision='predicate_spill', status='selected',
                              reason='live predicate peak exceeds the role capacity',
                              detail=entries['predicate_spills']))
    for logical, step in entries.get('mma_retiles', []):
        decisions.append(dict(decision='mma_k_retile', status='selected',
                              reason=f'K{logical} fragments leave no private VRF; K{step} commands'))
    if entries.get('rctx_spmd'):
        decisions.append(dict(decision='rctx_spmd', status='selected', contexts=entries['rctx_spmd']))
    if entries.get('rctx_roles'):
        decisions.append(dict(decision='rctx_roles', status='selected', pairs=entries.get('rctx_roles_pairs')))
    plan['decisions'] = decisions
    return plan


def _sum_roles(roles, key):
    total = Counter()
    for role in roles:
        total.update(role.get(key, {}))
    return dict(sorted(total.items()))


def record_decision(plan, decision, status, reason, **detail):
    """Append one decision made from the plan (admission, retile, tuning)."""
    if plan is not None:
        plan.setdefault('decisions', []).append(dict(decision=decision, status=status,
                                                     reason=reason, **detail))


def admission(plan, *, code_limit):
    """Resources over their limits, as (resource, used, limit) triples."""
    res = plan.get('resources', {})
    over = []
    for used, limit, name in ((res.get('code_bytes'), code_limit, 'code_bytes'),
                              (res.get('vrf_peak_registers'), res.get('vrf_limit'), 'vector_regs'),
                              (res.get('accumulators'), res.get('accumulator_limit'), 'accumulators'),
                              (res.get('barriers'), res.get('barrier_limit'), 'barriers'),
                              (res.get('descriptors'), res.get('descriptor_limit'), 'descriptors')):
        if type(used) is int and type(limit) is int and used > limit:
            over.append((name, used, limit))
    return over


def tuning_resources(plan, *, code_limit):
    """(resources, capacities) for the tuner's legality check, from the plan."""
    res = plan.get('resources', {})
    pairs = dict(code_bytes=(res.get('code_bytes'), code_limit),
                 vector_regs=(res.get('vrf_peak_registers'), res.get('vrf_limit')),
                 accumulators=(res.get('accumulators'), res.get('accumulator_limit')),
                 barriers=(res.get('barriers'), res.get('barrier_limit')),
                 descriptors=(res.get('descriptors'), res.get('descriptor_limit')))
    used = {k: u for k, (u, lim) in pairs.items() if type(u) is int and type(lim) is int}
    return used, {k: pairs[k][1] for k in used}
