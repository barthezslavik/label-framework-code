# Inventory leftovers: paper4 'de Sitter gives N != 0' (Cartan curvature of dS), pedagogy Noether theorem for rotor fields.
# Run: uv run --with sympy python inv_misc_check.py
import sympy as sp
from ga_core import *

t, x, y, z = sp.symbols('t x y z'); X4 = (t, x, y, z)
l = sp.symbols('l', positive=True)
SIG = {0: -1, 1: 1, 2: 1, 3: 1}                     # forms only: metric not used by ^ and d
D = lambda A: ext_d(A, X4, SIG)
eta = [-1, 1, 1, 1]
a = sp.exp(t / l)
e = [{(0,): 1}, {(1,): a}, {(2,): a}, {(3,): a}]          # flat-slicing de Sitter coframe
# spin connection omega^a_b (indices a up, b down): omega^i_0 = omega^0_i = e^i / l, others zero
w = [[{} for _ in range(4)] for _ in range(4)]
for i in (1, 2, 3):
    w[i][0] = scal(1 / l, e[i]); w[0][i] = scal(1 / l, e[i])
T = [add(D(e[A]), *[wedge(w[A][B], e[B], SIG) for B in range(4)]) for A in range(4)]
ok("M1 dS coframe with omega^i_0 = e^i/l is torsion-free", all(Ti == {} for Ti in T))
R = [[add(D(w[A][B]), *[wedge(w[A][C], w[C][B], SIG) for C in range(4)]) for B in range(4)] for A in range(4)]
# Cartan (so(1,4)) curvature F^a_b = R^a_b - (1/l^2) e^a ^ e_b  (e_b = eta_bb e^b)
Fc = [[add(R[A][B], scal(-eta[B] / l**2, wedge(e[A], e[B], SIG))) for B in range(4)] for A in range(4)]
ok("M2 de Sitter is Cartan-flat: R^a_b = (1/l^2) e^a ^ e_b, so F = R - e^e/l^2 = 0 -> any N built from F vanishes (paper4 claim refuted)",
   all(all(sp.simplify(v) == 0 for v in Fc[A][B].values()) for A in range(4) for B in range(4)))

# pedagogy: Noether for a planar rotor field R = exp(e12 theta/2), L = -<d R d R~>_0 * 2 = 1/2 (d theta)^2 (1+1 D):
# symmetry theta -> theta + eps gives J^mu = d^mu theta and d_mu J^mu = Euler-Lagrange expression
th = sp.Function('theta')(t, x)
Lag = (sp.diff(th, t)**2 - sp.diff(th, x)**2) / 2
EL = sp.diff(sp.diff(Lag, sp.diff(th, t)), t) + sp.diff(sp.diff(Lag, sp.diff(th, x)), x)
J = [sp.diff(Lag, sp.diff(th, t)), sp.diff(Lag, sp.diff(th, x))]
ok("M3 Noether: d_mu J^mu equals the Euler-Lagrange expression, so J is conserved on shell",
   sp.simplify(sp.diff(J[0], t) + sp.diff(J[1], x) - EL) == 0)
# M4 (paper4 Prop. 2.5(ii)) temperature of a uniformly accelerated observer in de Sitter: dS = {X.X = -1/H^2} in R^{1,4} (signature + - - - -).
#    Worldline X = (R sinh(t/R), R cosh(t/R), 0, 0, Z) with R^2 + Z^2 = 1/H^2 (proper time t). Its 5D acceleration A has |A| = 1/R; the
#    normal part (along X) has magnitude H, so the intrinsic dS acceleration a satisfies a^2 = 1/R^2 - H^2. The Unruh temperature of the
#    embedding hyperbola is 1/(2 pi R) = sqrt(a^2 + H^2)/(2 pi) (Narnhofer-Peter-Thirring, Deser-Levin), not a/(2 pi).
import sympy as _sp
_t, _R, _H = _sp.symbols('t R H', positive=True)
_Z = _sp.sqrt(1 / _H**2 - _R**2)
_X = _sp.Matrix([_R * _sp.sinh(_t / _R), _R * _sp.cosh(_t / _R), 0, 0, _Z])
_eta5 = _sp.diag(1, -1, -1, -1, -1)
_ip = lambda u, v: (u.T * _eta5 * v)[0]
_V = _X.diff(_t); _A = _X.diff(_t, 2)
_norm_part = _ip(_A, _X) / _ip(_X, _X) * _X                              # component of A along the normal X
_a_vec = _A - _norm_part
_a2 = _sp.simplify(-_ip(_a_vec, _a_vec))
ok(f"M4 de Sitter: <V,V> = {_sp.simplify(_ip(_V, _V))}, |A|^2 = {_sp.simplify(-_ip(_A, _A))} = 1/R^2, intrinsic a^2 = {_a2}: "
   "T = 1/(2 pi R) = sqrt(a^2 + H^2)/(2 pi); a = theta H gives T = sqrt(1+theta^2) H/(2 pi), not theta T_dS",
   _sp.simplify(_ip(_V, _V) - 1) == 0 and _sp.simplify(-_ip(_A, _A) - 1 / _R**2) == 0 and _sp.simplify(_a2 - (1 / _R**2 - _H**2)) == 0)
