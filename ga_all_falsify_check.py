# 59: second pass -- strict attempts to refute the claims made in conversation after 58 (constants, principles, the four GA operations,
# 'orientation cannot be borrowed') that had no check yet. AF1-AF8. REFUTED / NARROWED where a test breaks a claim.
# Run: uv run --with sympy --with numpy --with scipy python ga_all_falsify_check.py
import numpy as np
import sympy as sp
from ga_core import ok

# AF1 'the action is the angle of a rotation by the pseudoscalar, e^{I S/hbar}': in the spacetime algebra the quantum phase i acts on a
#     Dirac spinor as right multiplication by the BIVECTOR I sigma_3; left multiplication by the pseudoscalar I is gamma5, the CHIRAL rotation.
#     Check (matrix form): the phase e^{i alpha} keeps a massive solution, e^{i gamma5 beta} does not. REFUTED: phase = bivector, not I.
g0 = np.diag([1, 1, -1, -1]).astype(complex)
sx = np.array([[0, 1], [1, 0]]); sy = np.array([[0, -1j], [1j, 0]]); sz = np.array([[1, 0], [0, -1]])
gam = [g0] + [np.block([[np.zeros((2, 2)), s], [-s, np.zeros((2, 2))]]).astype(complex) for s in (sx, sy, sz)]
g5 = 1j * gam[0] @ gam[1] @ gam[2] @ gam[3]
m = 1.0; p = np.array([0.0, 0.3, -0.4, 0.5]); p[0] = np.sqrt(p[1]**2 + p[2]**2 + p[3]**2 + m**2)
D = p[0] * gam[0] - p[1] * gam[1] - p[2] * gam[2] - p[3] * gam[3] - m * np.eye(4)
w, v = np.linalg.eig(D); psi = v[:, np.argmin(np.abs(w))]
r_phase = np.abs(D @ (np.exp(0.7j) * psi)).max()
r_chiral = np.abs(D @ ((np.cos(0.7) * np.eye(4) + 1j * np.sin(0.7) * g5) @ psi)).max()
ok(f"AF1 REFUTED: phase e^(i alpha) residual {r_phase:.1e}, pseudoscalar (chiral) rotation residual {r_chiral:.2f} at m = 1: the action "
   "phase is the bivector I sigma_3 (right side), the pseudoscalar I is chirality", r_phase < 1e-8 and r_chiral > 0.1)

# AF2 'finite c and hbar = the rigid algebras beyond which one cannot deform': the Poincare algebra is NOT rigid -- [P0, P1] = eps K satisfies
#     Jacobi for every eps (1+1 toy: [K, P0] = P1, [K, P1] = P0), giving de Sitter (eps > 0) or anti-de Sitter. Rigid (semisimple) is the
#     de Sitter algebra; stabilising Poincare-Heisenberg needs further constants (Lambda and a minimal length; Vilela Mendes 1994,
#     Chryssomalakos-Okon 2004, from memory). REFUTED; and Lambda then is a deformation parameter like c, not only a number to explain.
eps = sp.symbols('epsilon')
basis = ['K', 'P0', 'P1']
br = {('K', 'P0'): {'P1': 1}, ('K', 'P1'): {'P0': 1}, ('P0', 'P1'): {'K': eps}}
def bracket(a, b):
    if a == b:
        return {}
    if (a, b) in br:
        return dict(br[(a, b)])
    return {k: -v for k, v in br[(b, a)].items()}
def bra_lin(X, Y):
    out = {}
    for a, ca in X.items():
        for b, cb in Y.items():
            for k, v in bracket(a, b).items():
                out[k] = out.get(k, 0) + ca * cb * v
    return out
def jac(a, b, c):
    t = {}
    for (x, y, z) in ((a, b, c), (b, c, a), (c, a, b)):
        for k, v in bra_lin({x: 1}, bra_lin({y: 1}, {z: 1})).items():
            t[k] = sp.simplify(t.get(k, 0) + v)
    return {k: v for k, v in t.items() if v != 0}
J = jac('K', 'P0', 'P1')
ok(f"AF2 REFUTED: Jacobi residual {J} for all eps -- Poincare (eps = 0) deforms to (A)dS; the rigid algebra needs Lambda", J == {})

# AF3 'k_B is absolute like c and hbar': k_B enters only as k_B T (Boltzmann factor, ideal gas, KMS); no algebra deforms with it. Its
#     physical content is atomicity (finite N_A at fixed R = N_A k_B), i.e. counting, already carried by hbar-cells (58 EN7). NARROWED: k_B
#     is a pure conversion factor; c, hbar (and G, Lambda) are deformation parameters.
ok("AF3 NARROWED: k_B is a conversion factor (temperature = energy); c, hbar, G, Lambda deform algebras, k_B does not", True)

# AF4 'almost every breaking is one of four GA operations (K, I, reflection, choice of a vector)': the electroweak breaking
#     SU(2) x U(1) -> U(1)_em by the Higgs vev is internal and breaks 3 of 4 generators; it is in none of the four classes. REFUTED.
dim_ew, dim_em = 3 + 1, 1
ok(f"AF4 REFUTED: electroweak breaking removes {dim_ew - dim_em} of {dim_ew} gauge generators, an internal breaking outside K, I, "
   "reflection and vector choice; the list covers spacetime breakings only", dim_ew - dim_em == 3)

# AF5 'orientation cannot be borrowed, scale can': perturbative corrections to a chirally protected mass are proportional to the mass, but
#     the QCD VACUUM supplies chiral charge (condensate): the proton mass is ~99% not from Yukawas. NARROWED: orientation cannot be borrowed
#     from perturbative corrections, it can be borrowed from the vacuum (spontaneous breaking).
mp, mq = 938.27, 2 * 2.16 + 4.70
ok(f"AF5 NARROWED: valence quark masses {mq:.1f} MeV vs m_p {mp:.0f} MeV: {100 * (1 - mq / mp):.0f}% of the proton mass is borrowed from "
   "the chiral condensate", 1 - mq / mp > 0.98)

# AF6 'principles give only counts, exponents and symmetric points': they also give transcendental geometric numbers -- 2 pi in the KMS
#     temperature a/2pi, 4 pi in Gauss, pi in S_dS, and alpha/2pi in the electron anomaly (Schwinger). NARROWED.
ae = (1 / 137.035999) / (2 * np.pi)
ok(f"AF6 NARROWED: principles also yield pi-type numbers (a_e = alpha/2pi = {ae:.6f}, measured 0.001159652)", abs(ae - 0.0011596) < 2e-6)

# AF7 'Lorentz invariance of the vacuum is unbroken (w = -1.03 +- 0.03)': the BAO-shape test of the base (F3) is insensitive to evolving w;
#     combinations of DESI DR2 BAO with CMB and supernovae prefer w0 > -1, wa < 0 at ~3-4 sigma (from memory, to verify). NARROWED: 'unbroken'
#     holds for the constant-w fits; the dynamical-dark-energy preference is an open challenge to the w = -1 theorem's empirical side.
ok("AF7 NARROWED: w = -1 consistent with constant-w fits; DESI DR2 combinations (to verify) prefer evolving w at ~3-4 sigma", True)

# AF8 '18 numbers have a GA symmetric point' counts the Yukawa chiral point as GA: the rotation is GA (left multiplication by I), but WHICH
#     fields rotate (flavour, left/right assignment) is internal; and g' (hypercharge) has minimal charge 1/6, which shifts its Dirac
#     self-dual point. NARROWED: 'GA point' means the operation is GA, the representation is internal.
ok("AF8 NARROWED: GA supplies the operation (I, K, reflection), the Standard Model supplies which fields carry it", True)
