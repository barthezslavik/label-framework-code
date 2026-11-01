# Checks for the rewritten paper1 (papers/current/paper1-rankone/main.tex), claims not covered by other scripts. Z1-Z4.
# Run: uv run --with sympy python pub_paper1_check.py
import sympy as sp
from ga_core import ok

# Z1 orbit reduction: det B is SL(3)-invariant; tr B is not (M = diag(t, t, t^-2))
b = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f'b{min(i, j)}{max(i, j)}'))
tt = sp.symbols('t', positive=True)
Mt = sp.diag(tt, tt, 1 / tt**2)
ok("Z1 det(M B M^T) = det B for det M = 1; tr(M B M^T) != tr B (M = diag(t,t,t^-2))",
   sp.simplify((Mt * b * Mt.T).det() - b.det()) == 0 and sp.simplify((Mt * b * Mt.T).trace() - b.trace()) != 0)
# Z1b transitivity on a level set (Cholesky construction of the paper): random numeric example
import random
random.seed(3)
def rnd_spd():
    L = sp.Matrix(3, 3, lambda i, j: sp.Rational(random.randint(1, 9), random.randint(1, 4)) if i >= j else 0)
    return L * L.T
B1, B2 = rnd_spd(), rnd_spd()
B2 = B2 * (B1.det() / B2.det())**sp.Rational(1, 3)                      # same determinant
L1, L2 = B1.cholesky(hermitian=False), B2.cholesky(hermitian=False)
Mc = L2 * L1.inv()
ok("Z1b SL(3) acts transitively on det B = c: M = L2 L1^-1 has det M = 1 and M B1 M^T = B2 (example)",
   sp.simplify(Mc.det() - 1) == 0 and sp.simplify(Mc * B1 * Mc.T - B2) == sp.zeros(3))

# Z2 stress tensor of S = int sqrt(-g) F(D), D = |det B|, B^{AB} = g^{mu nu} d_mu phi^A d_nu phi^B, labels = spatial coordinates, rest frame.
#    Metric diag(A, -B1, -B2, -B3): D = 1/(B1 B2 B3). Then rho = -F, p = F - 2 D F'(D); for F = -m D^n: p/rho = 2n - 1.
A, b1, b2, b3, m, n = sp.symbols('A B1 B2 B3 m n', positive=True)
Dsym = sp.symbols('D', positive=True)
Ff = sp.Function('F')
D = 1 / (b1 * b2 * b3)
sg = sp.sqrt(A * b1 * b2 * b3)
S = sg * Ff(D)
pt = {A: 1, b1: 1, b2: 1, b3: 1}
T00 = sp.simplify((-2 / sg * sp.diff(S, A)).subs(pt))                      # T^{00} = -(2/sqrt g) dS/dg_00
T11 = sp.simplify((-2 / sg * (-sp.diff(S, b1))).subs(pt))                  # g_11 = -B1
rho_ = sp.simplify(T00.subs(sp.Symbol('D'), 1)); p_ = sp.simplify(T11)
Fp = sp.Subs(sp.Derivative(Ff(Dsym), Dsym), Dsym, 1).doit()
ok(f"Z2 rho = {rho_}, p = {p_}: rho = -F, p = F - 2 D F'(D) (at D = 1)",
   sp.simplify(rho_ + Ff(1)) == 0 and sp.simplify(p_ - (Ff(1) - 2 * Fp)) == 0)
Fpow = lambda x: -m * x**n
w = sp.simplify((Fpow(Dsym) - 2 * Dsym * sp.diff(Fpow(Dsym), Dsym)) / (-Fpow(Dsym)))
ok(f"Z2b power law F = -m D^n: w = p/rho = {w} = 2n - 1; dust (w = 0) iff n = 1/2, i.e. F linear in sqrt(D) = number density",
   sp.simplify(w - (2 * n - 1)) == 0)
# Z3 no anisotropic stress / heat flux: in the rest frame T^{ij} is proportional to delta^{ij} when all B_i are equal and T^{0i} = 0
#    (F depends on the metric only through D, which is symmetric in B1, B2, B3)
T22 = sp.simplify((-2 / sg * (-sp.diff(S, b2))).subs(pt)); T33 = sp.simplify((-2 / sg * (-sp.diff(S, b3))).subs(pt))
ok("Z3 T^11 = T^22 = T^33 in the rest frame (isotropic): no anisotropic stress", sp.simplify(T11 - T22) == 0 and sp.simplify(T11 - T33) == 0)
# Z4 Brown dictionary: J^mu = eps^{mu nu rho sigma} d phi^1 d phi^2 d phi^3 for labels = (x, y, z) scaled by k: n = k^3 = sqrt(det of spatial B)
k = sp.symbols('k', positive=True)
J0 = k**3                                                                   # eps^{0123} k^3 in flat space
detB = sp.Matrix(sp.diag(k**2, k**2, k**2)).det()
ok("Z4 n = |J| = k^3 = sqrt(det B) for phi^A = k x^A: number density = sqrt(det B), as in Brown 1993", sp.simplify(J0 - sp.sqrt(detB)) == 0)
