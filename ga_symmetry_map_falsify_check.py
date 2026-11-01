# Strict attempts to refute 51 (MOND as a property of the p = 0 subgroup). RF1-RF6; RF5 (data) is btfr_deep_check.py.
# Where a test breaks a statement it is marked REFUTED / NARROWED; PASS means the test ran as stated.
# Run: uv run --with sympy --with numpy --with scipy python ga_symmetry_map_falsify_check.py
import numpy as np
import sympy as sp
from scipy.optimize import brentq
from ga_core import ok

p, al, lam, x = sp.symbols('p alpha lambda x', real=True)

# RF1 'the symmetry forces the outer slope 1/2' is circular: covariance under mu = lam^p gives g ~ g_N^alpha with alpha = (1 - 2p)/2,
#     and every alpha has its p. The subgroup is chosen by the slope, not the slope by the subgroup, unless p = 0 is singled out otherwise.
alpha_of_p = (1 - 2 * p) / 2
p_of_alpha = sp.solve(sp.Eq(alpha_of_p, al), p)[0]
ok(f"RF1 REFUTED as an explanation: slope alpha <-> p = {p_of_alpha} is a bijection; 'forced 1/2' = 'chose p = 0'",
   sp.simplify(alpha_of_p.subs(p, p_of_alpha) - al) == 0)

# RF2 the independent reasons for p = 0. (a) matter-era homothety: with masses, g and g_N both weight lam^-1, so a label-free local law
#     must be homogeneous of degree 1 -- Newton, not deep MOND; with a label of weight -1 ANY nu(g_N/a0) is covariant. The homothety selects
#     nothing. (b) O(4,1) of space: conformal covariance of L3 is a property of the deep-MOND equation itself (32), no outside reason.
#     (c) self-dual point of p -> -p: kinematic (37 T5).
F = sp.Function('F')
sol = sp.dsolve(sp.Eq(-x * F(x).diff(x), -F(x)), F(x)).rhs
nu = sp.Function('nu'); a0 = sp.symbols('a0', positive=True); gN = sp.symbols('g_N', positive=True)
law = gN * nu(gN / a0)
cov = sp.simplify(law.subs({gN: gN / lam, a0: a0 / lam}, simultaneous=True) - law / lam)
ok(f"RF2 REFUTED (homothety -> MOND): label-free covariance gives g = {sol} (Newton); with a0 ~ 1/t every g_N nu(g_N/a0) is covariant "
   f"(residual {cov}): the cosmological symmetry does not pick deep MOND; p = 0 has no reason outside MOND itself", cov == 0)

# RF3 'a0 is a label of an orbit, not a constant' is true of every dimensional constant: with M fixed G M = r^3/t^2 has weight
#     lam^(3 - 2(1 - p)) = lam^(1 + 2p), invariant exactly on the Kepler subgroup p = -1/2. G is the label of the Kepler orbit by the same
#     argument. NARROWED: SM3 says nothing specific about a0.
wG = 3 - 2 * (1 - p)
ok(f"RF3 NARROWED: weight of G M is lam^({wG}), = 1 at p = {sp.solve(wG, p)[0]}: G is a 'label' in exactly the sense a0 is",
   sp.solve(wG, p) == [sp.Rational(-1, 2)])

# RF4 locality g = F(g_N), the premise of every 'forced' line, fails in MOND itself: external field effect. Deep MOND
#     div(|grad phi| grad phi) = 4 pi G a0 rho with grad phi = g_e z + grad psi, linearised: g_e (lap psi + d_zz psi) = 4 pi G a0 rho.
#     Point mass: psi = -k / sqrt(x^2 + y^2 + z^2/2); internal g / g_N depends on a0/g_e and on direction.
X, Y, Z, ge, k = sp.symbols('X Y Z g_e k', positive=True)
psi = -k / sp.sqrt(X**2 + Y**2 + Z**2 / 2)
op = sp.simplify(sp.diff(psi, X, 2) + sp.diff(psi, Y, 2) + 2 * sp.diff(psi, Z, 2))
gx = sp.simplify(sp.diff(psi, X).subs({Y: 0, Z: 0})); gz = sp.simplify(sp.diff(psi, Z).subs({X: 0, Y: 0}))
ratio = sp.simplify(gz.subs(Z, X) / gx)
ok(f"RF4 REFUTED (locality): linearised EFE operator on psi gives {op} off the source; at equal distance g_z/g_x = {ratio}: g is not a "
   "function of g_N (direction and g_e enter). The three 'forced' features need isolated spherical systems", op == 0 and ratio != 1)

# RF6 is the SM4 counterexample physical? A law needs g(g_N) increasing and g_D >= 0; the 'mu' function mu = g_N/g increasing in g.
G = lambda gn, h: brentq(lambda gg: gg - gn * (1 + h(gg)), gn * (1 + 1e-12), gn + 1e3 + 10 * np.sqrt(gn))
h_alt = lambda y: np.exp(-np.log(y) + 0.5 * np.log(y) * np.exp(-np.log(y)**2))
grid = np.geomspace(1e-4, 1e4, 800)
gg = np.array([G(v, h_alt) for v in grid])
mono = np.all(np.diff(gg) > 0) and np.all(np.diff(grid / gg) > 0) and np.all(gg >= grid)
ok("RF6 SM4 survives: the alternative law is monotonic, g_D >= 0, mu increasing -- an admissible interpolation", mono)

# RF7 'RAR transit = same structure as LCDM transit (SM5)': the standard RAR g/(1 - exp(-sqrt(g_N/a0))) is an equally monotonic transit
#     between slopes 1 and 1/2; any sigmoid is. NARROWED: SM5 is a shape analogy with no content (no flow, no eigenvalues).
rar = grid / (1 - np.exp(-np.sqrt(grid)))
sl = np.gradient(np.log(rar), np.log(grid))
ok(f"RF7 NARROWED: the McGaugh RAR is also a monotonic transit ({sl[0]:.3f} -> {sl[-1]:.3f}); SM5 adds nothing",
   np.all(np.diff(sl) > -1e-6) and sl[0] < 0.52 and sl[-1] > 0.99)
