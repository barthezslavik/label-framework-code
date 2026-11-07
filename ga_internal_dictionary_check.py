# 147: step 4 of paper12 (internal quantum numbers). Dictionary between the outer relabelling symmetry (SDiff) and the inner condition "S" (det pi(u) = 1 on H_F).
# Premises I1-I3 as in ga_internal_check.py (algebra C + H + M_N(C), its representation on one generation, chirality).
# ID1 every inner relabelling u a u* of the finite algebra preserves its trace, the noncommutative volume, with any positive weights on
#     the summands; this holds for all 13 generators, including the direction that S removes: volume preservation is automatic inside,
#     and S is not its analogue;
# ID2 every unitary has |det pi(u)| = 1 on H_F; S fixes the phase of det pi(u), which is 4 alpha + 4N beta for lambda = e^{i alpha},
#     m = e^{i beta}: S fixes the phase of the complex volume of H_F, not a real volume;
# ID3 on the abelian plane the S line coincides, for every N = 1..7, with the line where the Fujikawa Jacobian of the fermion measure
#     vanishes in an SU(2) instanton background (Tr gamma T^2 G = 0); in a gravitational background that Jacobian vanishes
#     identically: S equals one anomaly-free condition of the measure, not the vanishing of all Jacobians;
# ID4 the label count and the instanton number are integrals of exact top forms: det dX = d_mu(X^1 C^mu_1) (four labels; with an action F(n), F'' != 0 and
#     non-degenerate gradients, F'(n) is a constant of integration, paper10) and (1/4) eps Tr F F = d_mu K^mu with the Chern-Simons
#     current (SU(2)), checked symbolically on polynomial fields; the theta term is linear in the instanton density, the analogue of
#     F = bn, so theta stays a constant of the action as in the series.
# Run: ~/.venvs/jev/bin/python ga_internal_dictionary_check.py
import numpy as np
import sympy as sp
from scipy.linalg import expm
from scipy.stats import unitary_group
from itertools import permutations
import importlib.util, io, contextlib

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

spec = importlib.util.spec_from_file_location('gi', __file__.replace('ga_internal_dictionary_check.py', 'ga_internal_check.py'))
gi = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(gi)                                        # reuse build(), left_traces(); its own checks run silently
rng = np.random.default_rng(7)

# ---------- ID1 the trace (noncommutative volume) is preserved by every inner relabelling ----------
def rand_quat():
    a = rng.normal(size=4); return np.array([[a[0] + 1j * a[1], a[2] + 1j * a[3]], [-a[2] + 1j * a[3], a[0] - 1j * a[1]]])
def unit_quat():
    q = rand_quat(); return q / np.sqrt(abs(np.linalg.det(q)))
worst = 0.0
for _ in range(200):
    lam = np.exp(1j * rng.uniform(0, 2 * np.pi)); q = unit_quat(); m = unitary_group.rvs(3, random_state=rng)
    a = (rng.normal() + 1j * rng.normal(), rand_quat(), rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))
    w = rng.uniform(0.1, 5, size=3)                                    # positive weights on the summands
    tr = lambda x: w[0] * x[0] + w[1] * np.trace(x[1]) + w[2] * np.trace(x[2])
    ua = (lam * a[0] * np.conj(lam), q @ a[1] @ q.conj().T, m @ a[2] @ m.conj().T)
    worst = max(worst, abs(tr(ua) - tr(a)))
states, A, B, T2, TN, gam = gi.build(3)
ta, tb = gi.left_traces(3)
ok(f"ID1 inner relabellings preserve the weighted trace of A_F (200 random u, a, weights; max deviation {worst:.1e}); this holds for "
   f"all 13 generators, including the direction S removes: volume preservation is automatic inside, S is not its analogue",
   worst < 1e-10)

# ---------- ID2 |det| = 1 always; S fixes the phase ----------
dev_mod = 0.0; dev_ph = 0.0
for _ in range(200):
    al, be = rng.uniform(-np.pi, np.pi, size=2)
    th2 = rng.normal(size=3); th3 = rng.normal(size=len(TN))
    # abelian part of pi(u) on particles + antiparticles: phases with traces (ta, tb) from left_traces; SU(2), SU(3) parts: exp of
    # traceless generators
    ph = ta * al + tb * be
    U = np.exp(1j * ph)                                                # det of the abelian part
    Us = expm(1j * sum(t * M for t, M in zip(th2, T2))) @ expm(1j * sum(t * M for t, M in zip(th3, TN)))
    d = U * np.linalg.det(Us)
    dev_mod = max(dev_mod, abs(abs(d) - 1)); dev_ph = max(dev_ph, abs(np.angle(d / np.exp(1j * ph))))
ok(f"ID2 |det pi(u)| = 1 for every unitary (max dev {dev_mod:.1e}); the SU(2), SU(3) parts have det 1 (phase dev {dev_ph:.1e}); "
   f"det pi(u) = exp i({ta:.0f} alpha + {tb:.0f} beta): S fixes the phase of the complex volume of H_F",
   dev_mod < 1e-10 and dev_ph < 1e-10 and (ta, tb) == (4, 12))

# ---------- ID3 S line = vanishing Fujikawa Jacobian in an SU(2) instanton background, for every N ----------
good = True; rows = []
for N in range(1, 8):
    st, A, B, T2, TN, gam = gi.build(N)
    ta, tb = gi.left_traces(N)
    s2 = [np.trace(gam @ T2[0] @ T2[0] @ G).real for G in (A, B)]
    gr = [np.trace(gam @ G).real for G in (A, B)]
    cross = ta * s2[1] - tb * s2[0]                                    # zero iff (ta, tb) is proportional to (s2_A, s2_B)
    rows.append(N); good &= abs(cross) < 1e-12 and max(map(abs, gr)) < 1e-12
ok("ID3 for N = 1..7 the S line equals the line where the SU(2) instanton Jacobian Tr gamma T^2 G vanishes; the gravitational "
   "Jacobian Tr gamma G vanishes on the whole abelian plane", good)

# ---------- ID4 exact top forms: det dX (Lambda) and Tr F F (theta) ----------
x = sp.symbols('x0:4')
def rpoly(deg=2):
    mons = [sp.Integer(1)] + list(x) + [x[i] * x[j] for i in range(4) for j in range(i, 4)]
    return sum(sp.Integer(int(rng.integers(-3, 4))) * mm for mm in mons[: 1 + 4 + (10 if deg >= 2 else 0)])
X = [rpoly() for _ in range(4)]
J = sp.Matrix(4, 4, lambda a, mu: sp.diff(X[a], x[mu]))
cof = [J.cofactor(0, mu) for mu in range(4)]
lhs = sp.expand(J.det()); rhs = sp.expand(sum(sp.diff(X[0] * cof[mu], x[mu]) for mu in range(4)))
ex1 = sp.simplify(lhs - rhs) == 0
# SU(2): A_mu = sum_a A_mu^a sigma_a / (2i), polynomial components of degree 1
sig = [sp.Matrix([[0, 1], [1, 0]]), sp.Matrix([[0, -sp.I], [sp.I, 0]]), sp.Matrix([[1, 0], [0, -1]])]
def rlin():
    return sum(sp.Integer(int(rng.integers(-2, 3))) * mm for mm in [sp.Integer(1)] + list(x))
Am = [sum((rlin() * s for s in sig), sp.zeros(2)) / (2 * sp.I) for _ in range(4)]
F = [[sp.expand(sp.diff(Am[n], x[m]) - sp.diff(Am[m], x[n]) + Am[m] * Am[n] - Am[n] * Am[m]) for n in range(4)] for m in range(4)]
def eps(p):
    p = list(p); s = 1
    for i in range(4):
        for j in range(i + 1, 4):
            if p[i] > p[j]: s = -s
    return s
perms = list(permutations(range(4)))
FF = sp.expand(sum(eps(p) * (F[p[0]][p[1]] * F[p[2]][p[3]]).trace() for p in perms) / 4)
K = [sp.expand(sum(eps(p) * (Am[p[1]] * sp.diff(Am[p[3]], x[p[2]]) + sp.Rational(2, 3) * Am[p[1]] * Am[p[2]] * Am[p[3]]).trace()
                   for p in perms if p[0] == mu)) for mu in range(4)]
dK = sp.expand(sum(sp.diff(K[mu], x[mu]) for mu in range(4)))
ex2 = sp.simplify(FF - dK) == 0 and FF != 0
ok("ID4 det dX = d_mu(X^1 C^mu_1) for four polynomial labels (label count) and (1/4) eps Tr F F = d_mu K^mu (Chern-Simons) for a "
   "polynomial SU(2) potential (instanton number): both top forms are exact", ex1 and ex2)

print("ALL PASS")
