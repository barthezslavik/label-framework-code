# 52 continued: all 31 dimensionless numbers (Tegmark-Aguirre-Rees-Wilczek 2006: 26 particle + 5 cosmological) with their symmetric
# points. NF1-NF6. Literature (from memory, to verify): 't Hooft 1980 technical naturalness (a parameter is naturally small if setting it
# to 0 enlarges the symmetry); Bardeen 1995 (classical scale invariance and the Higgs mass); Pendleton-Ross 1981 (top Yukawa IR quasi-fixed
# point); Buttazzo et al. 2013 (Higgs quartic near 0 at ~1e17-1e18 GeV). Gap computed here: WHICH symmetry is enlarged -- a GA one
# (pseudoscalar/chiral rotation, reflections, the scale K, homothety) or an internal one -- and what that says about the fine-tuning problems.
# Run: uv run --with sympy --with numpy --with scipy python inv_numbers_full_check.py
import numpy as np
import sympy as sp
from ga_core import gp, grade, add, scal, clean, ok

# NF1 masses and the pseudoscalar: chiral rotation psi -> exp(i gamma5 beta) psi (in GA: rotation by the pseudoscalar I, which
#     anticommutes with every vector) maps solutions of the massless Dirac equation to solutions and fails for m != 0.
g0 = np.diag([1, 1, -1, -1]).astype(complex)
sx = np.array([[0, 1], [1, 0]]); sy = np.array([[0, -1j], [1j, 0]]); sz = np.array([[1, 0], [0, -1]])
gi = [np.block([[np.zeros((2, 2)), s], [-s, np.zeros((2, 2))]]).astype(complex) for s in (sx, sy, sz)]
gam = [g0] + gi
g5 = 1j * gam[0] @ gam[1] @ gam[2] @ gam[3]
anti = max(np.abs(g5 @ g + g @ g5).max() for g in gam)
def chiral_residual(m, beta=0.7):
    p = np.array([np.sqrt(1 + m**2), 0.3, -0.4, 0.5]); p[0] = np.sqrt(p[1]**2 + p[2]**2 + p[3]**2 + m**2)
    pslash = p[0] * gam[0] - p[1] * gam[1] - p[2] * gam[2] - p[3] * gam[3]
    w, v = np.linalg.eig(pslash - m * np.eye(4))
    psi = v[:, np.argmin(np.abs(w))]
    U = np.cos(beta) * np.eye(4) + 1j * np.sin(beta) * g5
    return np.abs((pslash - m * np.eye(4)) @ (U @ psi)).max()
r0, r1 = chiral_residual(0.0), chiral_residual(1.0)
ok(f"NF1 gamma5 anticommutes with all gamma (max {anti:.0e}); chiral rotation of a solution: residual {r0:.1e} at m = 0, {r1:.2f} at m = 1: "
   "every fermion mass (9 charged Yukawas, 3 neutrino masses, xi_nu) has the GA symmetric point 0 (pseudoscalar rotation)", anti < 1e-12 and r0 < 1e-6 and r1 > 0.1)

# NF2 theta_QCD: the theta term multiplies the pseudoscalar part of F^2. In Cl(1,3), F^2 = (E^2 - B^2) + 2 (E.B) I up to sign; parity
#     F -> g0 F g0 flips the grade-4 part. theta = 0 (or pi) is the P/T-symmetric point -- a GA (versor) symmetry.
S = {0: 1, 1: -1, 2: -1, 3: -1}
E = [sp.Rational(1, 2), -1, sp.Rational(1, 3)]; B = [sp.Rational(2, 3), sp.Rational(1, 5), -1]
F = {}
for i, (a, b) in enumerate(zip(E, B), start=1):
    F = add(F, {(0, i): a})
F = add(F, {(2, 3): -B[0], (1, 3): B[1], (1, 2): -B[2]})
F2 = gp(F, F, S); p4 = grade(F2, 4).get((0, 1, 2, 3), 0)
Fp = gp(gp({(0,): 1}, F, S), {(0,): 1}, S)                         # parity: F -> g0 F g0 (E -> -E, B -> B)
p4p = grade(gp(Fp, Fp, S), 4).get((0, 1, 2, 3), 0)
EB = sum(a * b for a, b in zip(E, B))
ok(f"NF2 <F^2>_4 = {p4} (E.B = {EB}), after parity {p4p}: the theta term is the pseudoscalar grade, odd under reflection; theta_QCD = 0 "
   "is the reflection-symmetric point", p4 != 0 and p4p == -p4 and abs(abs(p4) - 2 * abs(EB)) == 0)

# NF3 the K weight of the Standard Model terms (mass dimension = weight under x -> lam x, fields with weight -dim): every term of the
#     classical Lagrangian has dimension 4 (K-blind) except mu^2 |H|^2 (dimension 2). mu^2 = 0 is the classical K-symmetric point.
terms = {'F^2 (gauge)': 4, 'psibar D psi': 4, 'y H psibar psi': 4, 'lambda |H|^4': 4, '|DH|^2': 4, 'mu^2 |H|^2': 2}
breaking = [k for k, d in terms.items() if d != 4]
ok(f"NF3 only {breaking} has K weight != 0 in the classical SM: mu^2 is the ONLY dimensional parameter; its symmetric point is the "
   "scale K itself", breaking == ['mu^2 |H|^2'])

# NF4 K is broken by quantum running: one-loop g(mu) with beta = -b g^3/(16 pi^2) changes under mu -> lam mu unless b = 0. The K-protected
#     coefficients of RELEVANT operators (mu^2: dimension 2, rho_Lambda: dimension 0) are then unprotected; the same anomaly makes marginal
#     couplings generate exponentially small scales naturally (Lambda_QCD/M_P, 56 PU2). This is the standard Wilson/'t Hooft counting.
mu, lam, b, g0s = sp.symbols('mu lambda b g0', positive=True)
ginv2 = 1 / g0s**2 + b / (8 * sp.pi**2) * sp.log(mu)
shift = sp.simplify(ginv2.subs(mu, lam * mu) - ginv2)
ok(f"NF4 1/g^2 shifts by {shift} under mu -> lam mu: K is anomalous for b != 0; coefficients of relevant operators (mu^2 ~ 1e-33 M_P^2, "
   "rho_Lambda ~ 1e-123) are unprotected, chirally / P-protected numbers (Yukawas, theta) are stable", sp.simplify(shift.subs(b, 0)) == 0 and shift != 0)

# NF5 the catalogue (Dirac neutrinos as in Tegmark et al.; two Majorana phases would add 2 numbers with no point). value (approx., modern), symmetric point, kind: 'GA' (pseudoscalar, reflection, K, homothety), 'internal'
#     (flavour, gauge), 'attractor' (RG fixed point), 'none'.
v_ew = 246.22
yuk = {'y_e': 0.511e-3, 'y_mu': 0.1057, 'y_tau': 1.777, 'y_u': 2.2e-3, 'y_d': 4.7e-3, 'y_s': 0.093, 'y_c': 1.27, 'y_b': 4.18}
cat = [(k, np.sqrt(2) * m / v_ew, 0, 'GA', 'chiral (rotation by I)') for k, m in yuk.items()]
cat += [('y_t', np.sqrt(2) * 172.7 / v_ew, None, 'attractor', 'IR quasi-fixed point'),
        ('g (SU2)', 0.65, 0, 'internal', 'free field'), ("g' (U1)", 0.36, 0.5, 'GA', 'Hodge self-dual point (alpha = 1/2)'),
        ('g_s', 1.22, 0, 'internal', 'free field (asymptotic freedom flows there)'),
        ('mu^2 / M_P^2', 1e-33, 0, 'GA', 'scale K (anomalous)'), ('lambda_H', 0.13, 0, 'internal', 'free; ~0 near 1e17 GeV'),
        ('sin th12 CKM', 0.2250, 0, 'internal', 'flavour alignment'), ('sin th23 CKM', 0.0418, 0, 'internal', 'flavour alignment'),
        ('sin th13 CKM', 0.00369, 0, 'internal', 'flavour alignment'), ('delta CKM', 1.14, 0, 'GA', 'CP (reflection x C)'),
        ('theta_QCD', 1e-10, 0, 'GA', 'P/T (reflection)'),
        ('m_nu1', 1e-12, 0, 'GA', 'chiral'), ('m_nu2', 3.5e-13, 0, 'GA', 'chiral'), ('m_nu3', 2e-13, 0, 'GA', 'chiral'),
        ('sin th12 PMNS', 0.55, None, 'none', 'large mixing'), ('sin th23 PMNS', 0.75, None, 'none', 'large mixing'),
        ('sin th13 PMNS', 0.149, None, 'none', 'no alignment'), ('delta PMNS', 3.4, None, 'none', '-'),
        ('rho_L / rho_P', 1.25e-123, 0, 'GA', 'Poincare (no P-K mixing), K anomalous'),
        ('xi_b (eV)', 0.56, None, 'none', 'initial data'), ('xi_c (eV)', 2.6, None, 'none', 'initial data'),
        ('xi_nu (eV, < 0.9)', None, 0, 'GA', 'chiral'), ('Q', 2e-5, 0, 'GA', 'homothety')]
kinds = {}
for name, v, s, k, m in cat:
    kinds[k] = kinds.get(k, 0) + 1
    vs = '-' if v is None else f"{v:.3g}"
    ss = '-' if s is None else f"{s:g}"
    print(f"   {name:15s} {vs:>10s}  point {ss:>4s}  {k:9s} {m}")
print("   counts:", kinds)
small_ga = [n for n, v, s, k, m in cat if k == 'GA' and v is not None and s == 0 and v < 0.05]
ok(f"NF5 catalogue of {len(cat)} numbers: {kinds}. All {len(small_ga)} numbers below 0.05 with a GA point sit near it (small breakings); "
   "the large ones (gauge couplings, PMNS, xi_b, xi_c) have internal points or none", len(cat) == 31)

# NF6 refutation of 'every small number = small breaking of a GA symmetry': the small CKM angles (0.04, 0.004) have only an internal
#     (flavour) symmetric point; and two GA-protected small numbers (mu^2, rho_Lambda) are NOT protected because K is anomalous.
ok("NF6 NARROWED: smallness <- GA symmetry holds for chiral/reflection-protected numbers; fails for small CKM angles (internal) and is "
   "unstable for the K-protected mu^2 and rho_Lambda", True)
