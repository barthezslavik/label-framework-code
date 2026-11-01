# What extra structure the fluid / dust equation needs beyond the metric-free tower Pi (papers/restart/21-conformal.md), G1-G6.
# Run: uv run --with sympy python ga_conformal_check.py
import itertools
import sympy as sp
from ga_core import *

LOR = {0: 1, 1: -1, 2: -1, 3: -1}
I4 = {(0, 1, 2, 3): 1}
pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
w = vec(list(sp.symbols('w0:4')))
B = clean({p: c for p, c in zip(pairs, sp.symbols('b0:6'))})
dot1 = lambda u, X: grade(gp(u, X, LOR), 1)
# G1  w.B = 0  <=>  w ^ (I B) = 0 : the contraction is the wedge with the I-dual (I = pseudoscalar, the object SDiff preserves)
lhs = dot1(w, B)
rhs = grade(gp(wedge(w, gp(I4, B, LOR), LOR), I4, LOR), 1)
ok(f"G1 w.B = +-(w^(IB)) I: {clean(add(lhs, rhs)) == {} and 'w.B = -(w^(IB))I' or 'w.B = (w^(IB))I'}",
   clean(add(lhs, rhs)) == {} or clean(add(lhs, scal(-1, rhs))) == {})

# G2  Hodge star on k-forms in n dims under g -> lam^2 g scales as lam^(n-2k): only the middle grade k = n/2 is conformally invariant
lam = sp.symbols('lambda', positive=True)
def star_factor(n, k):
    sig = [1] + [-1] * (n - 1)
    g = sp.diag(*[lam**2 * s for s in sig]); gi = g.inv()
    sq = sp.sqrt(abs(g.det()))
    idx = list(range(k)); rest = list(range(k, n))
    F = {tuple(idx): 1}                                   # one basis k-form e^0^...^e^(k-1)
    up = 1
    for i in idx:
        up *= gi[i, i]                                    # raise all k indices
    return sp.simplify(sq * up / sp.sqrt(abs(sp.diag(*sig).det())) / sp.prod([sig[i] for i in idx]))
f42, f41, f32, f31 = star_factor(4, 2), star_factor(4, 1), star_factor(3, 2), star_factor(3, 1)
ok(f"G2 *_(lam^2 g) = lam^(n-2k) *_g: n=4,k=2 -> {f42}; n=4,k=1 -> {f41}; n=3,k=1 -> {f31}; n=3,k=2 -> {f32}: only grade 2 in 4D is invariant",
   f42 == 1 and f41 == lam**2 and f31 == lam and f32 == 1 / lam)

# G3  w^mu Omega_{mu nu} computed with g = lam^2 eta equals lam^-2 times the eta one: the zero set (Euler / geodesic condition) is conformally invariant
X = sp.symbols('t x y z')
wf = [sp.Function(f'w{m}')(*X) for m in range(4)]
eta = sp.diag(1, -1, -1, -1)
Om = sp.Matrix(4, 4, lambda m, n: sp.diff(wf[n], X[m]) - sp.diff(wf[m], X[n]))
L = sp.Function('L')(*X)
g = L**2 * eta
con = lambda G: sp.Matrix([sum((G.inv() * sp.Matrix(wf))[m] * Om[m, n] for m in range(4)) for n in range(4)])
ok("G3 (w.Omega)_g = L^-2 (w.Omega)_eta for any conformal factor L(x): the fluid condition sees only the conformal class",
   sp.simplify(con(g) - con(eta) / L**2) == sp.zeros(4, 1))

# G4  Lichnerowicz in GA terms: if w.Omega = 0 (eta) and h = |w|_eta, then in g = h^2 eta the same w is unit and geodesic
h2 = sum(eta[m, m] * wf[m]**2 for m in range(4))                     # w^2 in eta
G = h2 * eta; Gi = G.inv()
Gam = [[[sum(Gi[r, s] * (sp.diff(G[s, m], X[n]) + sp.diff(G[s, n], X[m]) - sp.diff(G[m, n], X[s])) for s in range(4)) / 2
         for n in range(4)] for m in range(4)] for r in range(4)]
wup = Gi * sp.Matrix(wf)
acc = sp.Matrix([sum(wup[m] * (sp.diff(wf[n], X[m]) - sum(Gam[r][m][n] * wf[r] for r in range(4))) for m in range(4)) for n in range(4)])
unit = sp.simplify(sum(wup[m] * wf[m] for m in range(4)))
ok(f"G4 g = h^2 eta: g(w,w) = {unit}; (w.nabla_g) w = (w.Omega)_g identically -> geodesic in g <=> Euler in eta (barotropic fluid = dust of h^2 eta)",
   unit == 1 and sp.simplify(acc - con(G)) == sp.zeros(4, 1))

# G5  the dust (h = 1) case is the metric-scale choice: a conformal factor L turns dust of eta into a barotropic fluid of L^2 eta with h' = 1/L
ok("G5 |w|_(L^2 eta) = |w|_eta / L: h' = h/L, so 'pressureless' (h = 1) is not conformally invariant; 'barotropic Euler' is",
   sp.simplify(sum((g.inv() * sp.Matrix(wf))[m] * wf[m] for m in range(4)) - h2 / L**2) == 0)

# G6  I^2 = -1 on bivectors: the middle-grade duality squares to -1 in Lorentzian 4D (A2): grade 2 splits into +-I eigen-planes only over C
IB = gp(I4, B, LOR)
ok("G6 I(IB) = -B for every bivector B (the middle-grade duality is a complex structure on grade 2)", clean(add(gp(I4, IB, LOR), B)) == {})
