# 54: links of the symmetry web that had no check yet. LK1-LK3. The other rows of the map cite their scripts (54-symmetry-web.md).
# Run: uv run --with sympy --with numpy --with scipy python ga_symmetry_links_check.py
import sympy as sp
from ga_core import gp, grade, grades, ok

lam, mu, kap, a0, G = sp.symbols('lambda mu kappa a0 G', positive=True)

# LK1 the stretch group (x, v, t) -> (lam x, mu v, lam/mu t) acting on Vlasov-Poisson: Vlasov is linear in f and needs phi ~ mu^2
#     (grad phi ~ dv/dt ~ mu^2/lam); f -> kap f. Newton: lap phi = 4 pi G int f d^3v gives mu^2/lam^2 = kap mu^3. Deep MOND:
#     div(|grad phi| grad phi) = 4 pi G a0 int f d^3v gives mu^4/lam^3 = kap mu^3 (a0 fixed). Both: a REAL two-parameter symmetry
#     (solution -> solution), with different mass weights M ~ kap mu^3 lam^3.
k_newton = sp.solve(sp.Eq(mu**2 / lam**2, kap * mu**3), kap)[0]
k_deep = sp.solve(sp.Eq(mu**4 / lam**3, kap * mu**3), kap)[0]
M_newton = sp.simplify(k_newton * mu**3 * lam**3); M_deep = sp.simplify(k_deep * mu**3 * lam**3)
ok(f"LK1 real symmetry of both: f -> {k_newton} f (Newton), {k_deep} f (deep MOND); mass weight {M_newton} vs {M_deep}: M ~ v^2 r is "
   "Newton's mass weight, M ~ v^4 is deep MOND's -- the BTFR exponent is the mass weight of the same group", M_newton == mu**2 * lam and M_deep == mu**4)

# LK2 with a0 also transformed (a0 -> alpha a0) the deep-MOND symmetry has three parameters and M ~ mu^4/alpha; fixing both M and a0
#     leaves mu = 1 for any lam: exactly the Milgrom subgroup p = 0.
al = sp.symbols('alpha', positive=True)
k3 = sp.solve(sp.Eq(mu**4 / lam**3, al * kap * mu**3), kap)[0]
M3 = sp.simplify(k3 * mu**3 * lam**3)
ok(f"LK2 with a0 -> alpha a0: M weight {M3}; M and a0 fixed => mu = 1 (p = 0, the Milgrom subgroup) for any lam", sp.solve(sp.Eq(M3.subs(al, 1), 1), mu) == [1])

# LK3 Hodge with sources: grad F = J -> grad (F I) = J I. J is grade 1, J I grade 3 (a magnetic current): the duality maps the theory with
#     electric charges to the theory with magnetic charges -- a CORRESPONDENCE; a real symmetry only in vacuum (37 T1).
S = {0: 1, 1: -1, 2: -1, 3: -1}
J = {(0,): sp.Rational(3, 2), (2,): -1}
JI = gp(J, {(0, 1, 2, 3): 1}, S)
ok(f"LK3 J I has grades {grades(JI)}: Hodge maps electric to magnetic sources (correspondence), symmetry only for J = 0", grades(JI) == [3])
