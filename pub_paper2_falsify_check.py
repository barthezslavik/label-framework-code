# Strict attempts to refute the claims of paper2 v3 (papers/current/paper2-theta/main.tex). PF1-PF6. Where a test breaks a statement the
# statement is narrowed or corrected (marked CORRECTION); where it survives, PASS means it survived.
# Run: uv run --with sympy --with numpy --with scipy python pub_paper2_falsify_check.py
import itertools
import numpy as np
import sympy as sp
from scipy.integrate import quad
from ga_core import *

# PF1 Prop 2.1 inverse through Hodge, f^-1(a) = fbar(aI) I^-1 / det f, in Lorentzian signature Cl(1,3) (I^2 = -1) with a random 4x4 map
#     (adjoint taken with the metric): does it survive outside Euclidean 3D?
S = {0: 1, 1: -1, 2: -1, 3: -1}; G = sp.diag(1, -1, -1, -1)
rng = np.random.default_rng(2)
M = sp.Matrix(4, 4, [sp.Rational(int(v), 5) for v in rng.integers(-6, 7, size=16)])
Madj = G.inv() * M.T * G                                    # metric adjoint
e = [{(k,): 1} for k in range(4)]
vecof = lambda col: vec(list(col))
fbar = lambda k: vecof(Madj[:, k])
I4 = {(0, 1, 2, 3): 1}; Iinv = scal(1 / sp.Integer(gp(I4, I4, S)[()]), I4)
def fbar_tri(T):
    out = {}
    for blade, cval in T.items():
        w = {(): 1}
        for k in blade:
            w = wedge(w, fbar(k), S)
        out = add(out, scal(cval, w))
    return out
a = vec([sp.Rational(1, 2), -1, sp.Rational(2, 3), sp.Rational(1, 3)])
lhs = vecof(M.inv() * sp.Matrix([a.get((k,), 0) for k in range(4)]))
rhs = scal(1 / M.det(), grade(gp(fbar_tri(grade(gp(a, I4, S), 3)), Iinv, S), 1))
diff = clean({k: sp.simplify(lhs.get(k, 0) - rhs.get(k, 0)) for k in set(lhs) | set(rhs)})
ok(f"PF1 Lorentzian 4D (det f = {M.det()}): f^-1(a) - fbar(aI)I^-1/det f = {diff}: the Hodge inverse formula survives", diff == {})

# PF2 Prop 3.1 'inertial time is a shift (square 0)': with Lambda > 0 the generator of a geodesic observer's time is P0 + eps K0 with eps ~
#     (H l)^2 > 0, whose square has the sign of eps. CORRECTION: the shift is the Lambda = 0 limit; in our universe every geodesic observer's
#     time is a (weak) stretch, every time generator with Lambda > 0 is hyperbolic.
S6 = {0: 1, 1: -1, 2: -1, 3: -1, 4: 1, 5: -1}
ninf = {(4,): 1, (5,): 1}; n0 = {(5,): sp.Rational(1, 2), (4,): -sp.Rational(1, 2)}
eps = sp.symbols('epsilon', real=True)
Bgen = wedge({(0,): 1}, add(ninf, scal(eps, n0)), S6)
sqg = sp.simplify(gp(Bgen, Bgen, S6).get((), 0))
ok(f"PF2 CORRECTION: (P0 + eps K0)^2 = {sqg}: zero only at eps = 0 (Lambda = 0); for Lambda > 0 (eps > 0) inertial time is hyperbolic -- a stretch",
   sp.simplify(sqg - 2 * eps) == 0)

# PF3 Prop 4.1 'the matter era is a repelling point': add radiation. dOmega_r/dN = 2 Omega_r (q - 1), q = Omega_r + Omega_m/2 - Omega_L.
#     At EdS radiation perturbations decay (eigenvalue -1). CORRECTION: the matter era is a SADDLE (attracting from radiation, repelling
#     towards Lambda and curvature); de Sitter stays an attractor.
OL, Ok, Orr = sp.symbols('Omega_L Omega_k Omega_r', real=True)
Om = 1 - OL - Ok - Orr
q = Orr + Om / 2 - OL
F = sp.Matrix([2 * OL * (1 + q), 2 * Ok * q, 2 * Orr * (q - 1)])
Jm = F.jacobian([OL, Ok, Orr])
eds = sorted(Jm.subs({OL: 0, Ok: 0, Orr: 0}).eigenvals().keys())
dsv = sorted(Jm.subs({OL: 1, Ok: 0, Orr: 0}).eigenvals().keys())
ok(f"PF3 CORRECTION: with radiation EdS eigenvalues {eds} (mixed signs: saddle), de Sitter {dsv} (all negative: attractor)",
   eds == [-1, 1, 3] and all(v < 0 for v in dsv))

# PF4 Prop 5.1 'p = 0 is at once the matter-era homothety and Milgrom's deep-MOND scaling': both map (t, r) -> lam (t, r), but masses differ:
#     deep-MOND scaling keeps M (and a0) fixed; the homothety scales M ~ rho r^3 ~ lam^-2 lam^3 = lam. NARROWED: the identity is kinematic.
#     Then a theorem: the MOND law g mu(g/a0) = g_N (or deep L3 phi = 4 pi G a0 rho) is covariant under the homothety exactly when a0 ~ 1/lam.
lam = sp.symbols('lambda', positive=True)
w_rho_hom = -2; w_r = 1
w_M_hom = w_rho_hom + 3 * w_r
lhs_deep = -3                                              # L3 phi: phi weight 0, three derivatives' worth: |grad|grad ~ lam^-2, div ~ lam^-1
w_a0 = sp.symbols('w')
sol_w = sp.solve(sp.Eq(lhs_deep, w_a0 + w_rho_hom), w_a0)
ok(f"PF4 NARROWED: homothety mass weight {w_M_hom} (deep-MOND scaling: 0) -- the identity is for (t, r, v, a) only. Deep MOND is covariant under "
   f"the homothety iff a0 has weight {sol_w[0]}, i.e. a0 ~ 1/t; Newton (lap phi ~ lam^-2 = rho) is covariant with no condition",
   w_M_hom == 1 and sol_w == [-1])

# PF5 the clock-law evidence: sensitivity to each local anchor and to the derived MUSE bins (harness rows; stat-only SPARC as an extreme).
c = 2.998e8; H0 = 67.4e3 / 3.0857e22; Om0, OL0 = 0.315, 0.685
t_of_z = lambda zz: quad(lambda y: 1 / (y * np.sqrt(Om0 / y**3 + OL0)), 1e-9, 1 / (1 + zz))[0] / H0
t0 = t_of_z(0)
E = lambda zz: np.sqrt(Om0 * (1 + zz)**3 + OL0)
def chi2(points, f):
    z = np.array([p_[0] for p_ in points]); a = np.array([p_[1] for p_ in points]); s = np.array([p_[2] for p_ in points])
    fz = np.array([f(q_) for q_ in z]); A = np.sum(a * fz / s**2) / np.sum(fz**2 / s**2)
    return float(np.sum(((a - A * fz) / s)**2))
clock = lambda zz: t0 / t_of_z(zz); Hz = E
MUSE = (0.868, 2.38, 0.10)
sets = {'SPARC(stat+sys) + MUSE': [(0.0, 1.20, np.hypot(0.02, 0.24)), MUSE],
        'SPARC(stat only) + MUSE': [(0.0, 1.20, 0.02), MUSE],
        'MIGHTEE + MUSE': [(0.04, 1.69, 0.13), MUSE],
        'all three': [(0.0, 1.20, np.hypot(0.02, 0.24)), (0.04, 1.69, 0.13), MUSE]}
res = {k: (chi2(v, clock), chi2(v, Hz)) for k, v in sets.items()}
for k, (cc, hh) in res.items():
    print(f"   {k:26s} chi2 clock {cc:6.2f}   H(z) {hh:6.2f}")
ok("PF5 the verdict flips with the local anchor: stat-only SPARC favours the clock law, MIGHTEE favours H(z) -- the 'undecided' of the paper is "
   "forced by the anchors, not by the high-z data", res['SPARC(stat only) + MUSE'][0] < res['SPARC(stat only) + MUSE'][1]
   and res['MIGHTEE + MUSE'][1] < res['MIGHTEE + MUSE'][0])

# PF6 the per-galaxy spread rests on SPARC velocity errors; the paper's mock argument needs x2-x4 systematics. Statement check only:
#     with errors inflated by sqrt(reduced chi2) the real chi2/dof (5.9) exceeds all mocks (<= 2.7) recorded in 34; a refutation would need
#     the mock systematics doubled -- not done here (data offline). Kept as stated, flagged as the weakest empirical point.
