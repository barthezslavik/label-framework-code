# Self-similarity strictly in GA (papers/restart/50-self-similarity.md). SS1-SS6.
# Known (Carr-Coley 1999, gr-qc/9806048): self-similarity of the first kind = a homothetic vector, L_xi g = 2c g with constant c; p = k rho is
# the only barotropic equation of state compatible; self-similar solutions are asymptotic states.
# GA form tested here: the derivative of a symmetry field, a -> a . nabla xi, splits into scalar (scale K), bivector (rotor) and symmetric
# traceless (SL shear) parts; homothety = no shear part and a constant scalar part, i.e. the generator lies in K + rotations (+ translations),
# the similarity group Sim = stabiliser of n_inf.
# Run: uv run --with sympy --with numpy --with scipy python ga_selfsimilar_check.py
import numpy as np
import sympy as sp
from ga_core import *

t, r, x, y, z = sp.symbols('t r x y z', real=True)
X4 = [t, x, y, z]
eta = sp.diag(1, -1, -1, -1)

def split(xi):
    """lowered derivative D_ab = d_a xi_b (flat metric): scalar part tr/4, antisymmetric (bivector), symmetric traceless (shear)."""
    xl = [sum(eta[b, c] * xi[c] for c in range(4)) for b in range(4)]
    D = sp.Matrix(4, 4, lambda a, b: sp.diff(xl[b], X4[a]))
    Sym = (D + D.T) / 2; Anti = (D - D.T) / 2
    trace = sp.simplify(sum(eta[a, a] * Sym[a, a] for a in range(4)))
    shear = sp.simplify(Sym - trace / 4 * eta)
    return sp.simplify(trace / 4), Anti, shear

# SS1 the three parts on four flat-space fields: dilation x (scale only), rotation (bivector only), translation (nothing), shear (traceless
#     symmetric only -> not a conformal symmetry at all).
fields = {'dilation x^mu': [t, x, y, z], 'rotation (x,y)': [0, -y, x, 0], 'translation': [1, 0, 0, 0], 'shear (x,-y)': [0, x, -y, 0]}
rep = {k: split(v) for k, v in fields.items()}
ok("SS1 " + "; ".join(f"{k}: scale {v[0]}, bivector {v[1] != sp.zeros(4, 4)}, shear {v[2] != sp.zeros(4, 4)}" for k, v in rep.items())
   + " -> homothety = scale + rotor (no shear); the SL shear is not a symmetry of the metric",
   rep['dilation x^mu'][0] == 1 and rep['dilation x^mu'][2] == sp.zeros(4, 4) and rep['rotation (x,y)'][0] == 0 and rep['rotation (x,y)'][2] == sp.zeros(4, 4)
   and rep['shear (x,-y)'][2] != sp.zeros(4, 4))

# SS2 Einstein-de Sitter is self-similar: xi = t d_t + (1/3) x d_x satisfies L_xi g = 2 g for g = diag(1, -t^(4/3), ...).
def lie_metric(g, xi, coords):
    n = len(coords)
    return sp.Matrix(n, n, lambda a, b: sp.simplify(sum(xi[c] * sp.diff(g[a, b], coords[c]) for c in range(n))
                                                    + sum(g[c, b] * sp.diff(xi[c], coords[a]) + g[a, c] * sp.diff(xi[c], coords[b]) for c in range(n))))
coords2 = [t, x]
n_ = sp.symbols('n', positive=True)
g_pl = sp.diag(1, -t**(2 * n_))
xi_pl = [t, (1 - n_) * x]
Lg = lie_metric(g_pl, xi_pl, coords2)
ok(f"SS2 power law a = t^n with xi = t d_t + (1-n) x d_x: L_xi g = {sp.simplify(Lg[0, 0] / g_pl[0, 0])} g (tt), {sp.simplify(Lg[1, 1] / g_pl[1, 1])} g (xx); "
   "EdS is n = 2/3", sp.simplify(Lg[0, 0] / g_pl[0, 0] - 2) == 0 and sp.simplify(Lg[1, 1] / g_pl[1, 1] - 2) == 0)

# SS3 LCDM has no homothety: for xi = alpha(t) d_t + beta x d_x the tt equation gives alpha = c t + d, and the xx equation needs
#     alpha a'/a + beta = c, i.e. (c t + d) a'/a constant -- true for power laws and (c = 0) for de Sitter, false for a = sinh^(2/3)(3t/2).
c_, d_ = sp.symbols('c d', real=True)
a_l = sp.sinh(sp.Rational(3, 2) * t)**sp.Rational(2, 3)
expr = (c_ * t + d_) * sp.diff(a_l, t) / a_l
dexpr = sp.simplify(sp.diff(expr, t))
pts = [{t: sp.Rational(1, 2)}, {t: 1}, {t: 2}]
sol_cd = sp.solve([dexpr.subs(p_) for p_ in pts], [c_, d_], dict=True)
ok(f"SS3 LCDM: d/dt[(c t + d) a'/a] = 0 at three times only for {sol_cd}: no homothety (only the trivial xi); power laws and de Sitter have one",
   sol_cd == [{c_: 0, d_: 0}])

# SS4 similarity reduction: spherical Euler-Poisson (pressureless) with rho = R(s)/t^2, v = V(s), s = r/t (the p = 0, velocity-invariant
#     scaling): every term of continuity and Euler carries the same power of t, so the PDEs become ODEs in s.
s = sp.symbols('s', positive=True)
R, V, m = sp.Function('R'), sp.Function('V'), sp.Function('m')
rho = R(r / t) / t**2; v = V(r / t); Mr = t * m(r / t)      # mass inside r: int 4 pi r^2 rho dr = t m(s)
cont = sp.diff(rho, t) + sp.diff(r**2 * rho * v, r) / r**2
euler = sp.diff(v, t) + v * sp.diff(v, r) + Mr / r**2        # G = 1
c3 = sp.simplify((cont * t**3).subs(r, s * t)); e1 = sp.simplify((euler * t).subs(r, s * t))
ok(f"SS4 t^3 x continuity and t x Euler depend on s = r/t only: {sp.diff(c3, t) == 0 and sp.diff(e1, t) == 0} -> self-similar collapse is an ODE in r/t",
   sp.simplify(sp.diff(c3, t)) == 0 and sp.simplify(sp.diff(e1, t)) == 0)

# SS5 stability in log time N = ln a (flat + curvature + Lambda + dust): dOmega_L/dN = 2 Omega_L (1+q), dOmega_k/dN = 2 Omega_k q,
#     q = Omega_m/2 - Omega_L. At EdS (Omega_m = 1) the perturbations grow with exponents 3 (Lambda) and 1 (curvature): a REPELLER (flatness and
#     coincidence problems); at de Sitter they decay (-3 matter, -2 curvature): the ATTRACTOR. The two fixed halves of the stretch flow.
OL, Ok = sp.symbols('Omega_L Omega_k', real=True)
Om = 1 - OL - Ok
q = Om / 2 - OL
F = sp.Matrix([2 * OL * (1 + q), 2 * Ok * q])
Jm = F.jacobian([OL, Ok])
eds = list(Jm.subs({OL: 0, Ok: 0}).eigenvals().keys())
ds = list(Jm.subs({OL: 1, Ok: 0}).eigenvals().keys())
ok(f"SS5 eigenvalues at EdS: {eds} (repeller); at de Sitter: {ds} (attractor, together with matter decaying as -3)",
   sorted(eds) == [1, 3] and all(e < 0 for e in ds))

# SS6 in conformal GA the homothety group is Sim = stabiliser of [n_inf] (32 B2): dilation and rotation rotors keep n_inf up to scale,
#     a transversion (special conformal, 1 + n_0 b/2) moves it -- self-similarity uses exactly the K + rotor part of the conformal group.
C5 = {0: 1, 1: 1, 2: 1, 3: 1, 4: -1}
ninf = {(3,): 1, (4,): 1}; n0 = {(4,): sp.Rational(1, 2), (3,): -sp.Rational(1, 2)}
bv = vec([sp.Rational(1, 3), 0, 0, 0, 0])
Tr = add({(): 1}, scal(sp.Rational(1, 2), gp(n0, bv, C5)))
rev = lambda M: {k: (vv if len(k) % 4 in (0, 1) else -vv) for k, vv in M.items()}
moved = clean({k: sp.simplify(vv) for k, vv in gp(gp(Tr, ninf, C5), rev(Tr), C5).items()})
ok(f"SS6 transversion maps n_inf to {moved} (not a multiple of n_inf): special conformal maps are not homotheties; Sim keeps n_inf",
   set(moved) - {(3,), (4,)} != set() or abs(moved.get((3,), 0) - moved.get((4,), 0)) > 0)
