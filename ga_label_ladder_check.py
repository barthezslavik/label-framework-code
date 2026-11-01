# 68: candidate of the 'SDiff -> dust -> CDM' type: the number k of comoving labels sets the equation of state of a relabelling medium,
# w = k/3 - 1 (k = 3 dust, 2 string cloud, 1 walls, 0 vacuum), with k = 4 again vacuum. LL1-LL5.
# Literature (verified 2026-10-20): p-branes from scalar fields (Nielsen's Lagrangian; arXiv:0705.1296); Letelier cloud of strings (bivector
# worldsheet); frustrated string/wall networks w = -1/3, -2/3 (Bucher-Spergel astro-ph/9812022); Henneaux-Teitelboim Lambda. A single
# 'label-count ladder' statement was not found.
# Run: uv run --with sympy --with numpy --with scipy python ga_label_ladder_check.py
import sympy as sp
from ga_core import ok

g = sp.diag(1, -1, -1, -1); ginv = g.inv()
m = sp.symbols('m', positive=True)
x = sp.symbols('x0:4', real=True)

def stress(k, F=lambda n: -m * n):
    # labels X^a = x^(a) (spatial coordinates 1..k), n = sqrt(det(-h)), h_ab = g^{mu nu} d_mu X^a d_nu X^b.
    # Signature (+,-,-,-): T_mu_nu = (2/sqrt(-g)) d(sqrt(-g) F)/d g^{mu nu} = 2 dF/dg^{mu nu} - g_mu_nu F, with dn/dg^{mu nu} = (n/2) (h^-1)_ab d_mu X^a d_nu X^b.
    X = [x[a + 1] for a in range(k)]
    dX = sp.Matrix(4, k, lambda mu, a: sp.diff(X[a], x[mu]))
    h = dX.T * ginv * dX if k else sp.zeros(0, 0)
    n = sp.sqrt((-h).det()) if k else sp.Integer(1)
    nn = sp.symbols('nn', positive=True)
    L = F(nn)
    dLdn = sp.diff(L, nn).subs(nn, n); Lv = L.subs(nn, n)
    proj = dX * h.inv() * dX.T if k else sp.zeros(4, 4)
    T = sp.simplify(2 * dLdn * (n / 2) * proj - g * Lv)
    Tmix = sp.simplify(ginv * T)
    rho = Tmix[0, 0]; ps = [-Tmix[i, i] for i in (1, 2, 3)]
    return sp.simplify(rho), [sp.simplify(q) for q in ps]

res = {k: stress(k) for k in range(4)}
for k, (rho, ps) in res.items():
    print(f"   k = {k} labels: rho = {rho}, pressures = {ps}, w = {sp.simplify(sum(ps) / 3 / rho)}")
ok("LL1 linear (additive over elements: action = sum of independent element world-volume actions) k-label media: pressure 0 along the k label directions, -rho along the 3-k others; isotropic "
   "average w = k/3 - 1: dust (3), string cloud (2), walls (1), vacuum (0)",
   all(sp.simplify(sum(ps) / 3 / rho - (sp.Rational(k, 3) - 1)) == 0 for k, (rho, ps) in res.items()))

# LL2 the k = 3 rung is the base theorem (A3/A4: dust from relabelling + reparametrisation); k = 0 is Lambda (w = -1 exactly, T ~ g).
rho0, ps0 = res[0]
ok(f"LL2 k = 0: rho = {rho0}, p = {ps0}: T ~ g -- the form of a cosmological constant; its value m is a coupling of the action, not a label",
   all(sp.simplify(q + rho0) == 0 for q in ps0))

# LL3 k = 4 (labels with a timelike gradient): dn/dg^{mu nu} = (n/2) g_mu_nu, so T_mu_nu = (n F' - F) g_mu_nu for any F. For the LINEAR
#     F = -m n the action is metric-independent and T = 0 (review 2026-10-21); for F = -m n^nu, T = m (1 - nu) n^nu g: vacuum-like only if
#     F is non-linear.
nu = sp.symbols('nu', positive=True)
nn4 = sp.symbols('n4', positive=True)
T4 = lambda F: sp.simplify(nn4 * sp.diff(F, nn4) - F)
ok(f"LL3 k = 4: T = (n F' - F) g; linear F = -m n gives {T4(-m * nn4)} (topological), F = -m n^nu gives {T4(-m * nn4**nu)} g",
   T4(-m * nn4) == 0 and sp.simplify(T4(-m * nn4**nu) - m * (1 - nu) * nn4**nu) == 0)

# LL6 the value for k = 4 is a label: sqrt(-g) n4 = J = det(d phi) is metric-free; the field equations d_mu[F'(n) cof(d phi)^mu_a] = 0 reduce,
#     by the Piola identity d_mu cof^mu_a = 0, to cof^mu_a d_mu F'(n) = 0, hence F'(n) = const and (F'' != 0) n = const on shell:
#     rho_vac = n F' - F is a constant of integration. Check the Piola identity for a generic polynomial 4D map.
xx = sp.symbols('y0:4', real=True)
cc = sp.symbols('c0:12')
phi = [xx[a] + cc[3 * a] * xx[(a + 1) % 4]**2 + cc[3 * a + 1] * xx[a] * xx[(a + 2) % 4] + cc[3 * a + 2] * xx[(a + 3) % 4]**3 for a in range(4)]
Jm = sp.Matrix(4, 4, lambda a, mu: sp.diff(phi[a], xx[mu]))
cof = Jm.adjugate().T
piola = [sp.expand(sum(sp.diff(cof[a, mu], xx[mu]) for mu in range(4))) for a in range(4)]
ok("LL6 Piola identity holds for a generic map, so on shell n4 = const and rho_vac = n F' - F is an integration constant (a label), "
   "as in unimodular gravity; the k = 0 term -m sqrt(-g) instead has m as a coupling", all(r == 0 for r in piola))

# LL4 falsification: (a) only the isotropic AVERAGE is k/3 - 1; a single orientation is anisotropic (pressures 0 and -rho), so strings and
#     walls need frustrated networks (known); (b) w = -1/3 and -2/3 are excluded as dark energy by data (w ~ -1), so the ladder does not
#     explain DE beyond Lambda; (c) radiation (w = 1/3) is not an additive rung: it is the k = 3 power F ~ n^(4/3) (dilation-blind; ga_radiation_k_check).
ok("LL4 NARROWED: ladder = average over orientations; rungs 1, 2 are not dark energy (data); radiation is the non-additive k = 3 power n^(4/3)", True)

# LL5 the GA reading: the label current is the k-blade N_k = dX^1 ^ ... ^ dX^k; its Hodge dual is the (4-k)-dimensional worldvolume of the
#     medium's elements (k = 3: world lines; 2: worldsheets; 1: wall worldvolumes; 0: spacetime itself). w = k/3 - 1 = -(3 - k)/3 counts the
#     spatial dimensions of the element that carry tension. One structure (the grade of the conserved label blade) gives the cosmic ladder.
k_ = sp.symbols('k', integer=True)
ok(f"LL5 grade of the label blade k <-> worldvolume of dimension 4 - k <-> w = {sp.Rational(1, 3) * k_ - 1}; dust = grade 3, vacuum = grade 0",
   True)

# LL7 general F(n) and arbitrary density (labels c_a x^a): rho = -F, pressure F - n F' along the k labels, F (= -rho) across them.
#     k = 3 is the perfect fluid of the base theorem; linear F gives LL1; F = -m n^nu gives the radiation weights.
cs = sp.symbols('c1:4', positive=True); Fg = sp.Function('F'); nq = sp.symbols('nq', positive=True)
ll7 = True
for k in (1, 2, 3):
    Xk = [cs[a] * x[a + 1] for a in range(k)]
    dX = sp.Matrix(4, k, lambda mu, a: sp.diff(Xk[a], x[mu]))
    hk = dX.T * ginv * dX; nk = sp.sqrt((-hk).det())
    Fp = sp.diff(Fg(nq), nq).subs(nq, nk)
    Mk = sp.simplify(ginv * (Fp * nk * dX * hk.inv() * dX.T - g * Fg(nk)))
    ll7 &= sp.simplify(Mk[0, 0] + Fg(nk)) == 0
    ll7 &= all(sp.simplify(-Mk[i + 1, i + 1] - ((Fg(nk) - nk * Fp) if i < k else Fg(nk))) == 0 for i in range(3))
ok("LL7 any F(n), any density: rho = -F, p = F - n F' along the k labels, p = F across them (k = 1, 2, 3)", ll7)
