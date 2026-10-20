"""Cost model v0: single core, serial grid, per-operator analytical form.

roadmap section 6.1, first version: "analytical + lookup calibration". The
structure is fixed by what the kernels do, and only its coefficients are
fitted:

    cycles = programs * (a_op + b_op * trips + c_op * tail)

    programs   the launch grid (the runtime runs programs one at a time)
    trips      loop iterations inside one program, from the shape:
               ceil(n / 16) for a row loop, K / 16 for a GEMM k-loop,
               1 for a one-block elementwise program
    tail       1 when the last block is partial (a masked tail costs a
               different instruction path), else 0

a/b/c are fitted by least squares on the dev split only. The 90% interval
comes from the validation split's relative residuals (split conformal), so
the interval is calibrated on points the fit never saw. The held-out split
is scored once, by evaluate.py, and never feeds back here.

Operator structure is declared, not learned: FEATURES says which shape
quantity is each operator's trip count. An operator missing from FEATURES
is refused (out of domain) rather than predicted.
"""
import math

VLEN = 16


def _rows(p):
    return p['shape'].get('rows', 1)


FEATURES = {
    # elementwise: one block per program
    'elementwise': lambda s: (math.ceil(s['n'] / VLEN), 1, int(s['n'] % VLEN != 0)),
    # a row loop inside a program, one program per row
    'row': lambda s: (s['rows'], math.ceil(s['n'] / VLEN), int(s['n'] % VLEN != 0)),
    # one 16x16 tile per program
    'tile2d': lambda s: (math.ceil(s['m'] / VLEN) * math.ceil(s['n'] / VLEN), 1,
                         int(s['m'] % VLEN != 0 or s['n'] % VLEN != 0)),
    # a k-loop inside one output tile per program
    'gemm': lambda s: ((s['M'] // VLEN) * (s['N'] // VLEN), s['K'] // VLEN, 0),
}

OPERATOR_KIND = {
    'mul2': 'elementwise', 'relu': 'elementwise', 'leaky_relu': 'elementwise',
    'sigmoid': 'elementwise', 'vector_add': 'elementwise', 'swiglu': 'elementwise',
    'fused_activation': 'elementwise', 'reverse_array': 'elementwise',
    'softmax': 'row', 'layernorm': 'row', 'rmsnorm': 'row',
    'matrix_transpose': 'tile2d', 'matmul_fp32_fp16_fp8': 'gemm',
}


class OutOfDomain(Exception):
    pass


def features(operator, shape):
    kind = OPERATOR_KIND.get(operator)
    if kind is None:
        raise OutOfDomain(f'{operator}: no declared structure in cost model v0')
    return FEATURES[kind](shape)


def _lstsq(X, y):
    """Normal equations with a tiny ridge; X is a list of rows."""
    k = len(X[0])
    A = [[sum(r[i] * r[j] for r in X) + (1e-9 if i == j else 0) for j in range(k)] for i in range(k)]
    b = [sum(r[i] * t for r, t in zip(X, y)) for i in range(k)]
    # Gauss-Jordan
    M = [row + [bi] for row, bi in zip(A, b)]
    for c in range(k):
        piv = max(range(c, k), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        if abs(M[c][c]) < 1e-12:
            continue
        for r in range(k):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [x - f * yv for x, yv in zip(M[r], M[c])]
    return [M[i][k] / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0 for i in range(k)]


class Model:
    def __init__(self):
        self.coef = {}
        self.interval = None          # relative half-width at 90%

    def fit(self, rows):
        """rows: dicts with operator, shape, cycles. Dev split only."""
        by_op = {}
        for r in rows:
            by_op.setdefault(r['operator'], []).append(r)
        for op, rs in by_op.items():
            X, y = [], []
            for r in rs:
                programs, trips, tail = features(op, r['shape'])
                X.append([1.0, trips, tail])
                y.append(r['cycles'] / programs)
            # With too few distinct trip counts, fit the mean only.
            if len({x[1] for x in X}) < 2:
                X = [[1.0, 0.0, x[2]] for x in X]
            self.coef[op] = _lstsq(X, y)
        return self

    def predict(self, operator, shape):
        if operator not in self.coef:
            raise OutOfDomain(f'{operator}: not in the fitted domain')
        programs, trips, tail = features(operator, shape)
        a, b, c = self.coef[operator]
        return max(programs * (a + b * trips + c * tail), 1.0)

    def calibrate(self, rows, coverage=0.9):
        """Split conformal on relative error, from the validation split."""
        if not 0 < coverage < 1:
            raise ValueError('coverage must be strictly between zero and one')
        errs = sorted(abs(self.predict(r['operator'], r['shape']) - r['cycles']) / r['cycles']
                      for r in rows if r['operator'] in self.coef)
        if not errs:
            self.interval = None
            return self
        rank = math.ceil((len(errs) + 1) * coverage)
        # Do not clip an unavailable conformal order statistic to the last
        # sample and advertise finite 90% coverage from too little data.
        self.interval = errs[rank - 1] if rank <= len(errs) else math.inf
        return self

    def predict_interval(self, operator, shape):
        p = self.predict(operator, shape)
        w = self.interval if self.interval is not None else math.inf
        # Residuals divide by actual, not prediction: |p-y| <= w*y.
        # Multiplying p by (1 +/- w) is not the inverse of this inequality.
        return p, p / (1 + w), (p / (1 - w) if w < 1 else math.inf)
