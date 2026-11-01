# Strict attempts to refute 52 (numbers and symmetric points) and 54 (symmetry web). XN1-XN5. REFUTED / NARROWED where a test breaks a claim.
# Run: uv run --with sympy --with numpy --with scipy python inv_numbers_falsify_check.py
import itertools
import numpy as np
import sympy as sp
from ga_core import ok

# XN1 ND7 'pure GA universe = exact de Sitter' contradicts ND5, where Lambda l_P^2 has its symmetric point at 0. With EVERY number at its
#     point (masses 0, mu^2 = 0, Lambda = 0, theta = 0, Q = 0, gravity off) the symmetry is the full conformal group of Minkowski,
#     O(2,4) = versors of Cl(2,4), larger than O(4,1). Lambda is then the FIRST breaking: a vector V in R^{2,4} whose stabiliser is
#     de Sitter (one sign of V^2), anti-de Sitter (the other) or Poincare x (null V). Check: stabiliser = orthogonal complement signature.
eta = np.diag([1, 1, -1, -1, -1, -1]).astype(float)          # R^{2,4}
def complement_signature(V):
    # basis of the orthogonal complement of V, then the signature of eta restricted to it (null V: the complement contains V itself)
    A = (eta @ V).reshape(1, -1)
    _, _, vt = np.linalg.svd(A); Nn = vt[1:].T
    w = np.linalg.eigvalsh(Nn.T @ eta @ Nn)
    return (int(np.sum(w > 1e-9)), int(np.sum(w < -1e-9)), int(np.sum(np.abs(w) <= 1e-9)))
sig = {'V^2 > 0': complement_signature(np.array([1, 0, 0, 0, 0, 0.])),
       'V^2 < 0': complement_signature(np.array([0, 0, 1, 0, 0, 0.])),
       'V null': complement_signature(np.array([1, 0, 1, 0, 0, 0.]))}
dim = lambda p, q: (p + q) * (p + q - 1) // 2
ok(f"XN1 REFUTED (ND7): complement signatures {sig}: O(1,4) (de Sitter, dim {dim(1, 4)}), O(2,3) (anti-de Sitter, dim {dim(2, 3)}), null -> "
   f"Poincare x scale; all inside O(2,4) (dim {dim(2, 4)}). The pure GA universe is conformal Minkowski, the conformal algebra Cl(2,4) "
   "itself; Lambda = choosing a vector in it, its sign = the sign of V^2", sig['V^2 > 0'] == (1, 4, 0) and sig['V^2 < 0'] == (2, 3, 0) and sig['V null'][2] == 1)

# XN2 'the two fine-tuning problems are exactly the K-protected numbers' (NF4) misses the flatness problem: Omega_k has its symmetric point
#     0 (homothety), and in the matter era it is a REPELLER (eigenvalue +1, PF3). NARROWED: fine-tuning problems = numbers whose symmetric
#     point is a K/homothety point that is unstable -- quantum (mu^2, rho_Lambda: anomaly) or classical (Omega_k: repeller).
OL, Ok, Orr = sp.symbols('Omega_L Omega_k Omega_r', real=True)
Om = 1 - OL - Ok - Orr; q = Orr + Om / 2 - OL
F = sp.Matrix([2 * OL * (1 + q), 2 * Ok * q, 2 * Orr * (q - 1)])
ek = F.jacobian([OL, Ok, Orr]).subs({OL: 0, Ok: 0, Orr: 0})[1, 1]
ok(f"XN2 NARROWED: d Omega_k / dN = {ek} Omega_k at EdS (repeller): the flatness problem is a classical homothety-point instability (with Q, 56 PU3); "
   "strong CP (theta, reflection-protected, stable) is a different kind -- unexplained initial value, not instability", ek == 1)

# XN3 'alpha has the self-dual point 1/2' depends on the quantisation: Dirac e g = 2 pi n (HL) with n = 1 gives 1/2; Schwinger's dyon
#     condition e g = 4 pi n gives 1. NARROWED: the self-dual point is 1/2 or 1 (order one), we sit at 1/137 either way.
e2 = sp.symbols('e2', positive=True)
pts = [sp.solve(sp.Eq(e2, (c * sp.pi)**2 / e2), e2)[0] / (4 * sp.pi) for c in (2, 4)]
ok(f"XN3 NARROWED: self-dual alpha = {pts} for the two quantisation rules: order one, not a single value", pts == [sp.Rational(1, 2), 1])

# XN4 'dimension 4 = the K-blind point of grade-2 Hodge' presupposes that the field strength is a bivector. For a grade-k field the
#     K-blind dimension is n = 2k (k = 3 -> n = 6, the 2-form potential). NARROWED: n = 4 follows from 'forces are bivectors', not alone.
n, k = sp.symbols('n k', positive=True, integer=True)
ok(f"XN4 NARROWED: K-blind Hodge on grade k needs n = {sp.solve(n - 2 * k, n)[0]}: 4 for bivector fields, 6 for trivector fields",
   sp.solve(n - 2 * k, n) == [2 * k])

# XN5 'all small numbers with a GA point sit near it' is close to a tautology (a small number is near 0). The non-trivial content is
#     narrower: (i) the symmetry at 0 exists and is a GA one (chiral rotation by I, reflection, K, homothety) -- checked in NF1-NF3; (ii) its
#     stability decides whether the smallness is a puzzle. GA does NOT give the values (the Yukawa pattern 3e-6 ... 1 is unexplained).
yuk = np.array([0.511e-3, 0.1057, 1.777, 2.2e-3, 4.7e-3, 0.093, 1.27, 4.18, 172.7]) * np.sqrt(2) / 246.22
span = np.log10(yuk.max() / yuk.min())
ok(f"XN5 NARROWED: the Yukawas span {span:.1f} decades with one symmetric point (0) -- the point explains why they CAN be small, "
   "not their values", span > 5)
