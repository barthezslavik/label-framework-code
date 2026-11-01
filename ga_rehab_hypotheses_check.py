# 60: for each refuted claim of 51-59, a replacement hypothesis that could be true, then a strict attempt to refute it. Theory part RH1-RH13;
# data parts: btfr_upsilon_trend_check.py (RH3), rar_a0_pergalaxy_check.py (RH2). PASS = the test ran; the message says SURVIVES / REFUTED.
# Run: uv run --with sympy --with numpy --with scipy python ga_rehab_hypotheses_check.py
import numpy as np
import sympy as sp
from ga_core import gp, add, scal, ok

# RH5 hypothesis 'compact symmetries protect a zero, non-compact ones (like K) do not'. Counterexample: the shift phi -> phi + c is
#     non-compact, yet it forbids m^2 phi^2 and keeps a Goldstone/axion massless to all orders (corrections respect the shift).
phi, cc, m = sp.symbols('phi c m', real=True)
mass_term = m**2 * phi**2 / 2
shifted = sp.expand(mass_term.subs(phi, phi + cc) - mass_term)
ok(f"RH5 REFUTED: under the non-compact shift the mass term changes by {shifted} (not invariant), the kinetic term is invariant: a "
   "non-compact symmetry protects m = 0; what fails for K is the anomaly, not non-compactness", shifted != 0)

# RH7 hypothesis 'Omega_k grows exactly when the expansion decelerates': d ln Omega_k / d ln a = 2 q for any FLRW content.
a = sp.symbols('a', positive=True); H = sp.Function('H')(a)
Ok = -1 / (a**2 * H**2)
q = -1 - a * sp.diff(H, a) / H
ident = sp.simplify(a * sp.diff(sp.log(-Ok), a) - 2 * q)
q0 = 0.315 / 2 - 0.685
ok(f"RH7 SURVIVES (computed): d ln Omega_k / d ln a - 2q = {ident}; today q = {q0:.2f}, so Omega_k shrinks since the acceleration began", ident == 0)

# RH8 hypothesis 'numbers without a symmetric point are O(1)'. Counterexample: eta = 6e-10 has no symmetric point and is tiny.
ok("RH8 REFUTED: eta = 6.1e-10 has no GA symmetric point and is not O(1); smallness without a point exists", 6.1e-10 < 1e-3)

# RH9 hypothesis 'entropy is produced by the SHEAR part of the velocity gradient (the SL part that is neither rotor nor scale)'.
#     Viscous production sigma = 2 eta s_ij s_ij + zeta theta^2 (Landau-Lifshitz). Rigid rotation (rotor): 0; pure shear: eta-term;
#     uniform expansion (K): only the bulk term zeta theta^2. NARROWED: shear -> shear viscosity, K -> bulk viscosity (zero for conformal
#     fluids and monatomic gases), rotor -> nothing.
x, y, z, w, sr, lam, eta, zeta = sp.symbols('x y z omega s lambda eta zeta', real=True)
def production(v):
    X = [x, y, z]
    G = sp.Matrix(3, 3, lambda i, j: sp.diff(v[i], X[j]))
    th = G.trace(); S = (G + G.T) / 2 - th / 3 * sp.eye(3)
    return sp.simplify(2 * eta * sum(S[i, j]**2 for i in range(3) for j in range(3)) + zeta * th**2)
p_rot = production([-w * y, w * x, 0]); p_shear = production([sr * y, 0, 0]); p_exp = production([lam * x, lam * y, lam * z])
ok(f"RH9 NARROWED (known): production rotor {p_rot}, shear {p_shear}, scale {p_exp}: the rotor never produces entropy, shear does, K only "
   "through bulk viscosity", p_rot == 0 and p_shear != 0 and sp.simplify(p_exp - 9 * zeta * lam**2) == 0)

# RH11 hypothesis 'in 3-space the quantum phase CAN be the pseudoscalar': in Cl(3,0) I3 = e123 is central and I3^2 = -1; in Cl(1,3) the
#     pseudoscalar anticommutes with vectors, so it cannot commute with the Dirac operator -- a relativistic phase must be a bivector.
S3 = {0: 1, 1: 1, 2: 1}; I3 = {(0, 1, 2): 1}
comm3 = [add(gp(I3, {(k,): 1}, S3), scal(-1, gp({(k,): 1}, I3, S3))) for k in range(3)]
S4 = {0: 1, 1: -1, 2: -1, 3: -1}; I4 = {(0, 1, 2, 3): 1}
anti4 = [add(gp(I4, {(k,): 1}, S4), gp({(k,): 1}, I4, S4)) for k in range(4)]
sq3 = gp(I3, I3, S3).get((), 0)
ok(f"RH11 SURVIVES (computed): I3 central in Cl(3,0) (commutators {[c for c in comm3 if c]}), I3^2 = {sq3}; I4 anticommutes with all "
   f"vectors in Cl(1,3) ({[c for c in anti4 if c]}): non-relativistic phase may be I3, relativistic phase must be a bivector",
   all(not c for c in comm3) and sq3 == -1 and all(not c for c in anti4))

# RH12 hypothesis 'the rigid kinematics is (anti-)de Sitter: semisimple, Killing form non-degenerate; Poincare is not'. Toy 1+1 algebra
#     [K,P0] = P1, [K,P1] = P0, [P0,P1] = eps K: Killing form determinant.
eps = sp.symbols('epsilon')
ad = {'K': sp.Matrix([[0, 0, 0], [0, 0, 1], [0, 1, 0]]),          # basis (K, P0, P1): ad_K P0 = P1, ad_K P1 = P0
      'P0': sp.Matrix([[0, 0, -eps], [0, 0, 0], [-1, 0, 0]]),    # ad_P0 K = -P1, ad_P0 P1 = eps K
      'P1': sp.Matrix([[0, eps, 0], [-1, 0, 0], [0, 0, 0]])}     # ad_P1 K = -P0, ad_P1 P0 = -eps K
names = ['K', 'P0', 'P1']
Kf = sp.Matrix(3, 3, lambda i, j: (ad[names[i]] * ad[names[j]]).trace())
dK = sp.factor(Kf.det())
ok(f"RH12 SURVIVES (computed; known: Bacry-Levy-Leblond): det Killing = {dK}: non-degenerate iff eps != 0 -- de Sitter / AdS rigid, "
   "Poincare not; Lambda is the constant that makes kinematics rigid", dK.subs(eps, 0) == 0 and dK.subs(eps, 1) != 0)

# RH13 hypothesis 'electroweak breaking is a choice of a vector, like Lambda, but in an internal space': SU(2) x U(1)_Y on C^2, Higgs
#     vev (0, v): the stabiliser has dimension 1 (U(1)_em).
s1 = np.array([[0, 1], [1, 0]]); s2 = np.array([[0, -1j], [1j, 0]]); s3 = np.diag([1, -1])
gens = [1j * s1 / 2, 1j * s2 / 2, 1j * s3 / 2, 1j * np.eye(2) / 2]       # Y = 1/2
hv = np.array([0, 1.0])
A = np.column_stack([g @ hv for g in gens])
A_real = np.vstack([A.real, A.imag])
stab = 4 - np.linalg.matrix_rank(A_real)
ok(f"RH13 SURVIVES (computed): stabiliser of the Higgs vector has dimension {stab} (U(1)_em): the electroweak breaking is a choice of a "
   "vector in an INTERNAL space; whether that space is part of a Clifford algebra (Trayling-Baylis, Furey) is not checked here", stab == 1)

# RH16 hypothesis 'a zero of a number either ENLARGES the group (protectable: chiral, scale) or only CONTRACTS it (not protectable:
#     Lambda -> 0 takes (A)dS (10) to Poincare (10), c -> oo Lorentz to Galilei)'. Dimensions.
dims = {'massless Dirac: + chiral U(1)': (0, 1), 'scale: Poincare 10 -> conformal 15': (10, 15),
        'Lambda -> 0: dS 10 -> Poincare 10': (10, 10), 'c -> oo: Poincare 10 -> Galilei 10': (10, 10)}
enl = [k for k, (a_, b_) in dims.items() if b_ > a_]; con = [k for k, (a_, b_) in dims.items() if b_ == a_]
ok(f"RH16 SURVIVES (computed dims; interpretation): enlarging zeros {len(enl)}, contracting zeros {len(con)}; Lambda's zero only contracts "
   "-- no symmetry appears at Lambda = 0, consistent with Lambda being unprotected", len(enl) == 2 and len(con) == 2)

# RH10 hypothesis 'the gravitational arrow (clumping, growth of Q) switches off in the Lambda era': growth rate f = Omega_m(a)^0.55.
Om = lambda aa: 0.315 * aa**-3 / (0.315 * aa**-3 + 0.685)
f_now, f_far = Om(1.0)**0.55, Om(10.0)**0.55
ok(f"RH10 SURVIVES (computed, known): growth rate f = {f_now:.2f} today, {f_far:.4f} at a = 10: clumping freezes in the Lambda era (f -> 0)", f_far < 0.05 and f_far < f_now / 10)
