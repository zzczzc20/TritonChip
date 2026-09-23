"""Canonical resource ledger theta0 and the common bound L* (section 4.10).

The ledger is an EVALUATION CONVENTION: published costs for abstract
primitives, the same for every candidate. It is not an area in mm^2 and
not a LUT count. A budget point b is a multiple of the reference
configuration's ledger cost, with three hard caps (logic, storage,
communication) and an overall cap.

L*(i, b, e) never reads the candidate. It is the optimum of the relaxed
allocation problem

    minimise t  s.t.  W_r / x_r <= t,  Q_env / B_env <= t,
                      sum_{r in dim d} c_r x_r <= cap_d   for each dimension,
                      sum_r c_r x_r <= cap_total

whose solution is closed-form: for a single dimension the best allocation
equalises the terms, t_d = sum_r c_r W_r / cap_d, and the overall cap
gives t_total = sum_r c_r W_r / cap_total. L* is the max of those and of
the environment's external-bandwidth term. It is optimistic by
construction (ideal reuse, no conflicts), which is what a lower bound
must be.
"""
import copy

THETA0 = {
    'version': 'theta0',
    'note': 'Relative primitive costs, fixed before any candidate is scored. '
            'Arbitrary units; only ratios matter and they are pre-registered.',
    'primitives': {
        # dimension, cost per unit of rate (per cycle) or per bit
        'fp_lane':   {'dim': 'logic', 'cost': 7.0},       # one binary32 FMA lane
        'int_lane':  {'dim': 'logic', 'cost': 1.0},
        'sfu_lane':  {'dim': 'logic', 'cost': 60.0},      # four-function slice
        'issue':     {'dim': 'logic', 'cost': 20.0},      # decode + scoreboard per issue slot
        'rctx':      {'dim': 'logic', 'cost': 4.0},       # per role context
        'mem_bytes': {'dim': 'communication', 'cost': 2.0},  # per byte/cycle of port
        'storage_bit': {'dim': 'storage', 'cost': 0.001},
    },
}


def config_cost(config, theta=THETA0):
    """K_theta(x) per dimension, from a configuration's primitive inventory.

    config = {'fp_lane': n, 'int_lane': n, 'sfu_lane': n, 'issue': n,
              'rctx': n, 'mem_bytes': n, 'storage_bit': n}
    """
    dims = {'logic': 0.0, 'storage': 0.0, 'communication': 0.0}
    for name, qty in config.items():
        prim = theta['primitives'][name]
        dims[prim['dim']] += prim['cost'] * qty
    dims['total'] = dims['logic'] + dims['storage'] + dims['communication']
    return dims


def reference_config(peak_rates, *, rctx=2, vregs=128, vlen=16, acc_bits=16 * 16 * 16 * 32):
    """The core-sim inventory, the 1x budget's reference."""
    return {
        'fp_lane': peak_rates['fp_lane'], 'int_lane': peak_rates['int_lane'],
        'sfu_lane': peak_rates['sfu_lane'], 'issue': peak_rates['issue'], 'rctx': rctx,
        'mem_bytes': peak_rates['mem_bytes'],
        'storage_bit': vregs * vlen * 32 + acc_bits,
    }


def budget(multiple, reference, theta=THETA0):
    base = config_cost(reference, theta)
    return {d: v * multiple for d, v in base.items()}


def fits(config, budget_caps, theta=THETA0):
    cost = config_cost(config, theta)
    return all(cost[d] <= budget_caps[d] * (1 + 1e-12) for d in budget_caps)


def common_bound(w, budget_caps, env, theta=THETA0):
    """L*(i, b, e) in reference cycles. Reads the workload, the budget and the
    environment -- never a candidate."""
    prim = theta['primitives']
    demand = {'fp_lane': w.fp_ops + w.sfu_ops, 'int_lane': w.int_ops,
              'mem_bytes': w.required_bytes}
    by_dim = {}
    for r, d in demand.items():
        if d:
            by_dim.setdefault(prim[r]['dim'], 0.0)
            by_dim[prim[r]['dim']] += prim[r]['cost'] * d
    terms = {dim: v / budget_caps[dim] for dim, v in by_dim.items()}
    terms['total'] = sum(by_dim.values()) / budget_caps['total']
    if w.required_bytes and env.get('external_bytes_per_cycle'):
        terms['external'] = w.required_bytes / env['external_bytes_per_cycle']
    return max(terms.values()), terms


def scaled(config, factor, keys=('fp_lane', 'int_lane', 'mem_bytes')):
    c = copy.deepcopy(config)
    for k in keys:
        c[k] *= factor
    return c
