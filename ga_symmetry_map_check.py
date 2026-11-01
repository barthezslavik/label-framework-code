# 51: MOND read as a property of symmetries, not as a law. Which MOND features follow from a symmetry alone and which are input. SM1-SM6.
# Literature first: deep MOND = space-time scale invariance is Milgrom's (ApJ 698, 1630, 2009); the gap computed here is the split
# 'forced by symmetry / label of an orbit / not fixed' inside the stretch group of 35.
# Run: uv run --with sympy --with numpy --with scipy python ga_symmetry_map_check.py
import numpy as np
import sympy as sp
from scipy.optimize import brentq
from ga_core import ok

x, lam, tau, C, c = sp.symbols('x lambda tau C c', positive=True)
F = sp.Function('F')

# SM1 locality g = F(g_N) plus one scaling with M fixed, r -> lam r, t -> lam^(1/2 - p) ... written as t -> lam/mu t, mu = lam^p (35):
#     g = r/t^2 has weight lam^(2p-1), g_N = GM/r^2 weight lam^-2. Covariance F(lam^-2 x) = lam^(2p-1) F(x), infinitesimally -2 x F' = (2p-1) F.
def law(p):
    sol = sp.dsolve(sp.Eq(-2 * x * F(x).diff(x), (2 * p - 1) * F(x)), F(x)).rhs
    return sp.simplify(sol)
kep, deep = law(sp.Rational(-1, 2)), law(0)
ok(f"SM1 locality + Kepler scaling (p = -1/2, t ~ r^3/2) forces g = {kep} (Newton); + Milgrom scaling (p = 0, t ~ r) forces g = {deep}: "
   "flat curves, v^4 = C G M with exponent exactly 4 and outer RAR slope 1/2 all follow from the p = 0 subgroup alone",
   sp.simplify(kep / x).free_symbols <= {sp.Symbol('C1')} and sp.simplify(deep**2 / x).free_symbols <= {sp.Symbol('C1')})

# SM2 no single law is covariant under both subgroups: c x = sqrt(C x) for all x is impossible. Any curve joining the two regimes breaks
#     both, so it carries a scale; there is exactly one number: where the two invariant curves cross, x* = C / c^2.
xs = sp.solve(sp.Eq(c * x, sp.sqrt(C * x)), x)
ok(f"SM2 Newton line and deep curve cross at g_N = {xs}: a joining law needs exactly one scale, the crossing point", xs == [C / c**2])

# SM3 that scale is a LABEL, not a parameter of the symmetry: the p = 0 subgroup maps every curve g^2 = C g_N into itself for any C.
#     Under a general (lam, tau) with M fixed C -> (lam/tau)^4 C; under the matter-era homothety (tau = lam, M -> lam M) C -> C/lam.
gN, g = sp.symbols('g_N g', positive=True)
Cmap = lambda wg, wgN: sp.simplify((g**2 * wg**2) / (gN * wgN) / (g**2 / gN))
w_general = Cmap(lam / tau**2, lam**-2)
w_homot = Cmap(lam / lam**2, lam / lam**2)
ok(f"SM3 label weight: general scaling {w_general}, Milgrom subgroup {w_general.subs(tau, lam)}, matter homothety {w_homot}: a0 is the "
   "label of the invariant orbit (a charge of the p = 0 subgroup), and it is a clock (a0 ~ 1/t) exactly when the symmetry is the homothety",
   w_general.subs(tau, lam) == 1 and w_homot == 1 / lam)

# SM4 the discrete exchange 'baryons <-> dark' (37): write g_D/g_N = h(g/a0); exchange g -> a0^2/g swaps the roles iff h(1/y) = 1/h(y).
#     Simple-mu (h = 1/y) has it. REFUTATION of 'the symmetries fix the knee': h = exp(-u + eps u e^{-u^2}), u = ln y, also satisfies the
#     exchange and both limits (Newton, deep with the same a0), but gives a different RAR.
def g_of(gn, h):
    return brentq(lambda gg: gg - gn * (1 + h(gg)), gn * (1 + 1e-12), gn + 1e3 + 10 * np.sqrt(gn))
h_simple = lambda y: 1 / y
eps = 0.5
h_alt = lambda y: np.exp(-np.log(y) + eps * np.log(y) * np.exp(-np.log(y)**2))
ys = np.geomspace(1e-3, 1e3, 7)
inv_ok = all(abs(h_alt(1 / y) * h_alt(y) - 1) < 1e-12 for y in ys)
grid = np.geomspace(1e-4, 1e4, 400)
d = max(abs(np.log10(g_of(v, h_alt) / g_of(v, h_simple))) for v in grid)
lim = abs(np.log10(g_of(1e-6, h_alt) / np.sqrt(1e-6))) < 1e-3 and abs(g_of(1e6, h_alt) / 1e6 - 1) < 1e-3
ok(f"SM4 NARROWED: a second law with the exchange symmetry and the same two limits differs from simple-mu by up to {d:.3f} dex: the "
   "knee shape is NOT fixed by the symmetries (simple-mu is only the lowest-degree algebraic choice)", inv_ok and lim and d > 0.03)

# SM5 the same structure as cosmology (50): LCDM runs between two self-similar points, the RAR between two scaling subgroups. The log-slope
#     d ln g / d ln g_N of simple-mu runs monotonically from 1 (Kepler orbit) to 1/2 (deep orbit), crossing 3/4 at g_N = a0 * (const).
lg = np.log([g_of(v, h_simple) for v in grid]); lgn = np.log(grid)
slope = np.gradient(lg, lgn)
ok(f"SM5 RAR = transit between the two subgroups: slope {slope[0]:.3f} -> {slope[-1]:.3f}, monotonic", slope[0] < 0.51 and slope[-1] > 0.99
   and np.all(np.diff(slope) > -1e-6))

# SM6 ledger of MOND features. forced by the p = 0 subgroup + locality: flat curves, BTF exponent 4, outer slope 1/2. label of the orbit:
#     the value of a0. not fixed by any symmetry here: knee shape (SM4), universality of the label between galaxies, locality itself.
feat = {'flat curves': 'symmetry', 'BTF exponent 4': 'symmetry', 'outer RAR slope 1/2': 'symmetry', 'value of a0': 'label',
        'knee shape': 'free', 'same a0 in all galaxies': 'free', 'g depends on g_N only': 'assumption'}
for k, v in feat.items():
    print(f"   {k:26s} {v}")
ok("SM6 ledger: 3 features forced, 1 label, 3 not fixed by symmetry", list(feat.values()).count('symmetry') == 3)
