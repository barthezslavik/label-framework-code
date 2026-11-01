# 58: strict attempts to refute the GA reading of entropy (chat 2026-10-20): (a) entropy = log of the pseudoscalar scale, conjugate to K;
# (b) entropy growth and the arrow = the same stretch K; (c) horizon entropy = pi S^2 / l_P^2 for the CGA sphere vector S. EN1-EN7.
# Run: uv run --with sympy --with numpy --with scipy python ga_entropy_falsify_check.py
import numpy as np
import sympy as sp
from ga_core import gp, add, scal, vec, ok

# EN1 (a) as mathematics: for a density on phase space mapped by a linear f, the differential entropy shifts by ln|det f|. Gaussian check.
s1, s2, lam, mu = sp.symbols('sigma1 sigma2 lambda mu', positive=True)
S_gauss = lambda C: sp.Rational(1, 2) * sp.log((2 * sp.pi * sp.E)**2 * C.det())
C0 = sp.diag(s1**2, s2**2)
f = sp.Matrix([[lam, 0], [0, mu]])
shift = sp.simplify(S_gauss(f * C0 * f.T) - S_gauss(C0))
ok(f"EN1 survives as mathematics: S(f rho) - S(rho) = {shift} = ln det f; Hamiltonian (det 1) maps keep it", sp.simplify(shift - sp.log(lam * mu)) == 0)

# EN2 (b) 'entropy growth = the K part': free streaming x -> x + p t is in SL (det 1), yet the COARSE-GRAINED entropy of a cold stream grows
#     (filamentation). Fine-grained entropy is constant (EN1, det 1). REFUTED: thermodynamic entropy grows by volume-preserving mixing plus
#     coarse-graining, with no K at all.
rng = np.random.default_rng(5)
N = 200000
x0 = rng.uniform(0.0, 0.2, N); p0 = rng.normal(0.0, 1.0, N)
def coarse_S(x, p, nb=40):
    H, _, _ = np.histogram2d(x % 1.0, np.clip(p, -4, 4), bins=nb, range=[[0, 1], [-4, 4]])
    q = H.ravel() / H.sum(); q = q[q > 0]
    return -np.sum(q * np.log(q))
S_t = [coarse_S(x0 + p0 * t, p0) for t in (0.0, 1.0, 10.0)]
ok(f"EN2 REFUTED: free streaming (det 1, no K) raises the coarse-grained entropy {S_t[0]:.2f} -> {S_t[1]:.2f} -> {S_t[2]:.2f}",
   S_t[2] > S_t[0] + 1.0)

# EN3 (b) cosmic stretch: free particles in an expanding universe, x -> a x, p -> p/a. The phase-space map has det 1: it is the squeeze
#     rotor e^{K theta} with p = -1 of the stretch group (35 S5), INSIDE SDiff. Entropy per comoving volume is conserved. REFUTED: the cosmic
#     stretch-time does not raise entropy; 'stretch' in space is 'squeeze' in phase space.
a = sp.symbols('a', positive=True)
M = sp.Matrix([[a, 0], [0, 1 / a]])
ok(f"EN3 REFUTED: det of (x, p) -> (a x, p/a) = {M.det()}: expansion is a phase-space squeeze, entropy of free matter is constant",
   M.det() == 1)

# EN4 (c) 'horizon entropy = pi S^2/l_P^2' for Kerr: A/4 = pi (r_+^2 + a_J^2) != pi r_+^2. With the areal radius r_A = sqrt(A/4 pi) it holds
#     by definition. NARROWED: an identity for spherical horizons with the areal radius, not a property of the sphere vector.
G_, Mm, aj = sp.symbols('G M a_J', positive=True)
rp = Mm + sp.sqrt(Mm**2 - aj**2)                                          # G = c = 1
A = 4 * sp.pi * (rp**2 + aj**2)
diff = sp.simplify(A / 4 - sp.pi * rp**2)
ok(f"EN4 NARROWED: Kerr A/4 - pi r_+^2 = {diff}: holds only for a_J = 0 or with the areal radius (then by definition)", diff == sp.pi * aj**2)

# EN5 (c) the CGA sphere S = P(c) - r^2/2 n_inf has S^2 = r^2 for EVERY sphere, horizon or not; entropy belongs to horizons. Check in Cl(4,1).
sig = {0: 1, 1: 1, 2: 1, 3: 1, 4: -1}                                   # e1 e2 e3 (indices 0-2), e+ (3), e- (4)
ninf = {(3,): 1, (4,): 1}; n0 = {(4,): sp.Rational(1, 2), (3,): -sp.Rational(1, 2)}
cx, cy, cz, r = sp.symbols('c_x c_y c_z r', real=True)
c = vec([cx, cy, cz])
P = add(c, scal((cx**2 + cy**2 + cz**2) / 2, ninf), n0)
S = add(P, scal(-r**2 / 2, ninf))
S2 = sp.simplify(gp(S, S, sig).get((), 0))
ok(f"EN5 NARROWED: S^2 = {S2} for any sphere: 'entropy = pi S^2' is not a property of the vector, only of a horizon", sp.simplify(S2 - r**2) == 0)

# EN6 de Sitter: S_dS = pi l^2 / l_P^2 = 3 pi / (Lambda l_P^2) ~ 3e122; same order as the Sorkin count (45 N1). Survives (known).
LlP2 = 2.85e-122
S_dS = 3 * np.pi / LlP2
ok(f"EN6 survives (known): S_dS = 3 pi/(Lambda l_P^2) = {S_dS:.1e}", 1e122 < S_dS < 1e123)

# EN7 (a) 'entropy = log of volume' depends on units: rescaling x -> lam x shifts S by ln lam (EN1). An absolute entropy needs a phase-space
#     cell, h^N (Sackur-Tetrode). NARROWED: log-volume entropy is absolute only in units of hbar -- the constant hbar is what makes it a number.
ok("EN7 NARROWED: differential entropy shifts by ln det under a change of units; absolute only per cell h^N (Sackur-Tetrode)", True)
