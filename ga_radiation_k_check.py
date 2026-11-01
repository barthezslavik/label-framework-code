# 69: radiation from K-blindness, in the same label media as 68. RK1-RK5. Known ingredients: conformal invariance <=> traceless stress
# (dilation Ward identity), traceless isotropic fluid has w = 1/3, Maxwell is conformal in 4D (base B3).
# Run: uv run --with sympy --with numpy --with scipy python ga_radiation_k_check.py
import sympy as sp
from ga_core import ok

g = sp.diag(1, -1, -1, -1); ginv = g.inv()
x = sp.symbols('x0:4', real=True)
m, nu, lam = sp.symbols('m nu lambda', positive=True)

def stress(k, F):
    X = [x[a + 1] for a in range(k)]
    dX = sp.Matrix(4, k, lambda mu, a: sp.diff(X[a], x[mu]))
    h = dX.T * ginv * dX
    n = sp.sqrt((-h).det())
    nn = sp.symbols('nn', positive=True)
    L = F(nn); dLdn = sp.diff(L, nn).subs(nn, n); Lv = L.subs(nn, n)
    T = -(-2 * dLdn * (n / 2) * (dX * h.inv() * dX.T) + g * Lv)
    Tm = sp.simplify(ginv * T)
    return sp.simplify(Tm[0, 0]), [sp.simplify(-Tm[i, i]) for i in (1, 2, 3)], sp.simplify(Tm.trace())

# RK1 general power F = -m n^nu with k labels: pressures (nu - 1) rho along the labels, -rho along the other 3 - k directions.
for k in (1, 2, 3):
    rho, ps, tr = stress(k, lambda nn: -m * nn**nu)
    print(f"   k = {k}: rho = {rho}, p = {ps}, trace = {tr}")
rho3, ps3, tr3 = stress(3, lambda nn: -m * nn**nu)
ok(f"RK1 k = 3, F ~ n^nu: p = {ps3[0]} (= (nu-1) rho), trace T = {tr3}", sp.simplify(ps3[0] - (nu - 1) * rho3) == 0)

# RK2 K-blindness of the action int sqrt(-g) F(n): under g -> lam^2 g, sqrt(-g) -> lam^4 sqrt(-g), h -> lam^-2 h, n -> lam^-k n. Invariant iff
#     F ~ n^(4/k): for k = 3 that is nu = 4/3.
k_ = sp.symbols('k', positive=True)
weight = sp.solve(sp.Eq(4 - k_ * nu, 0), nu)[0]
ok(f"RK2 K-blind label medium needs F ~ n^{weight}; k = 3: nu = {weight.subs(k_, 3)}", weight == 4 / k_)

# RK3 then the stress is traceless for every k and the orientation-average w = 1/3: radiation. k = 3 is the isotropic photon-gas-like case.
outs = {}
for k in (1, 2, 3):
    rho, ps, tr = stress(k, lambda nn, k=k: -m * nn**sp.Rational(4, k))
    outs[k] = (rho, ps, tr, sp.simplify(sum(ps) / 3 / rho))
    print(f"   K-blind k = {k}: p = {ps}, trace = {tr}, mean w = {outs[k][3]}")
ok("RK3 K-blindness => traceless => mean w = 1/3 for every k (isotropic only for k = 3): radiation is the K-blind label medium",
   all(o[2] == 0 and o[3] == sp.Rational(1, 3) for o in outs.values()))

# RK4 the trace measures the K-weight (dilation Ward identity): T^mu_mu = (4 - k nu) F up to sign -- dust (k=3, nu=1): rho; radiation: 0;
#     vacuum (k=0): 4 rho. One number, the K-weight of the Lagrangian, orders the inventory: Lambda 4, dust 1, radiation 0.
rho_d, ps_d, tr_d = stress(3, lambda nn: -m * nn)
ok(f"RK4 traces: dust {tr_d} (= rho), radiation 0, vacuum 4 rho: the trace is the K-weight of the action (Ward identity, known)",
   sp.simplify(tr_d - rho_d) == 0)

# RK5 independent check on a field: Maxwell stress is traceless in 4D (its action is K-blind, the middle-grade Hodge of base B3).
E = sp.symbols('E1:4', real=True); Bf = sp.symbols('B1:4', real=True)
E2 = sum(e**2 for e in E); B2 = sum(b**2 for b in Bf)
T00 = (E2 + B2) / 2
Tii = sum(-(E[i]**2 + Bf[i]**2) + (E2 + B2) / 2 for i in range(3))       # sum of Maxwell stresses sigma_ii = -(E_i^2 + B_i^2) + (E^2+B^2)/2
ok(f"RK5 Maxwell: rho - sum p = {sp.simplify(T00 - Tii)}: traceless, w = 1/3 for isotropic radiation", sp.simplify(T00 - Tii) == 0)

# RK6 falsification: the K-blind k = 1, 2 media are anisotropic; dominant energy condition |p_i| <= rho fails for k = 1 (p = 3 rho along the
#     label). NARROWED: physical radiation from the label construction is only the isotropic k = 3 case (p = rho/3).
dec = {k: all(sp.simplify(sp.Abs(q) - o[0]) <= 0 for q in o[1]) for k, o in outs.items()}
ok(f"RK6 NARROWED: dominant energy condition holds for k = {[k for k, v in dec.items() if v]}, fails for k = {[k for k, v in dec.items() if not v]}",
   dec[3] and dec[2] and not dec[1])
