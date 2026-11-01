# A more CDM-like object than SDiff of labels: SDiff (symplectic flow) of phase space on the mass shell p^2 = m^2.
# papers/restart/24-phase-space.md, checks K1-K4.
# Run: uv run --with sympy --with numpy python ga_phase_space_check.py
import numpy as np
import sympy as sp
from ga_core import *

# K1 reparametrisation symmetry -> zero pressure.
# K1a worldline action S = int L(xdot) dtau with L = (xdot.xdot)^(k/2): under tau -> sigma(tau), S picks up sigma'^(k-1): invariant only for k = 1
k, s1 = sp.symbols('k sigma1', positive=True)
xd = sp.symbols('v0:4'); vv = sum(c * x**2 for c, x in zip([1, -1, -1, -1], xd))
L = vv**(k / 2)
Lnew = (vv / s1**2)**(k / 2) * s1                                           # xdot -> xdot/sigma', dtau -> sigma' dtau
ratio = sp.simplify(Lnew / L)
ok(f"K1a reparametrisation tau -> sigma(tau) multiplies the worldline Lagrangian by {ratio}: invariant only for k = 1 (L ~ |xdot|, proper length)",
   sp.simplify(ratio.subs(k, 1)) == 1 and sp.simplify(ratio.subs(k, 2)) != 1)
# K1b matter Lagrangian L = F(n), n = |J|/sqrt(-g) with the current density J fixed. Rest frame, metric diag(A, -B1, -B2, -B3):
#     rho = -F, p = n drho/dn - rho = F - n F'(n). p = 0 <=> F = -m n (linear, the K1a case). Sanity: radiation F = -n^(4/3) gives p = rho/3.
A, B1, B2, B3, j = sp.symbols('A B1 B2 B3 j', positive=True)
nn = sp.symbols('n', positive=True)
F = sp.Function('F')
sg = sp.sqrt(A * B1 * B2 * B3)
n_ = sp.sqrt(A) * j / sg                                                     # |J| / sqrt(-g), J = (j, 0, 0, 0)
Sdens = sg * F(n_)
pt = {A: 1, B1: 1, B2: 1, B3: 1}
# T^{mu nu} = -(2/sqrt(-g)) dS/dg_{mu nu}; g_00 = A, g_11 = -B1
T00 = sp.simplify((-2 / sg * sp.diff(Sdens, A)).subs(pt))
T11 = sp.simplify((-2 / sg * (-sp.diff(Sdens, B1))).subs(pt))
rho_, p_ = sp.simplify(T00.subs(j, nn)), sp.simplify(T11.subs(j, nn))
rad = sp.simplify((p_ / rho_).subs(F(nn), -nn**sp.Rational(4, 3)).doit())
ok(f"K1b L = F(n): rho = {rho_}, p = {p_} = n rho' - rho; radiation F = -n^(4/3): p/rho = {rad}",
   sp.simplify(rho_ + F(nn)) == 0 and sp.simplify(p_ - (F(nn) - nn * sp.diff(F(nn), nn))) == 0 and rad == sp.Rational(1, 3))
solF = sp.dsolve(sp.Eq(F(nn) - nn * sp.diff(F(nn), nn), 0)).rhs
ok(f"K1c p = 0 <=> n F' = F <=> F = {solF}: linear in n, i.e. energy = mass x number (dust) — the reparametrisation-invariant case",
   sp.simplify(solF.diff(nn, 2)) == 0)

# K2 cold distribution on the mass shell: f = A delta^3(p - m u) with A = n u^0 gives N^mu = n u^mu and T^{mu nu} = m n u^mu u^nu (dust from labels)
m, nd = sp.symbols('m n_d', positive=True)
u1, u2, u3 = sp.symbols('u1:4', real=True)
p1, p2, p3 = sp.symbols('p1:4', real=True)
u0 = sp.sqrt(1 + u1**2 + u2**2 + u3**2)
pvec = [sp.sqrt(m**2 + p1**2 + p2**2 + p3**2), p1, p2, p3]
uvec = [u0, u1, u2, u3]
Af = nd * u0
def moment(expr):                                                            # int d^3p expr/p^0 * Af delta^3(p - m u)
    e = (expr / pvec[0] * Af).subs({p1: m * u1, p2: m * u2, p3: m * u3})
    return sp.simplify(sp.powdenest(sp.factor(e), force=True))        # sqrt((1+u^2)^2) = 1+u^2 (positive)
N = [moment(pvec[mu]) for mu in range(4)]
T = sp.Matrix(4, 4, lambda a_, b_: moment(pvec[a_] * pvec[b_]))
ok("K2 cold (single-velocity) phase-space distribution: N^mu = n u^mu, T^{mu nu} = m n u^mu u^nu — rank one, pressure 0 (dust)",
   all(sp.simplify(N[mu] - nd * uvec[mu]) == 0 for mu in range(4))
   and sp.simplify(T - m * nd * sp.Matrix(4, 1, uvec) * sp.Matrix(1, 4, uvec)) == sp.zeros(4))

# K3 1D collisionless toy (non-relativistic free streaming): x = q + t v0(q), v0 = -sin q, uniform initial density.
#    Before shell crossing (t < 1) every x has one stream: velocity dispersion (pressure) 0. After (t > 1) several streams: pressure > 0.
def streams(t, x, Nq=200001):
    q = np.linspace(-np.pi, np.pi, Nq)
    xs = q - t * np.sin(q) - x
    idx = np.where(np.sign(xs[:-1]) != np.sign(xs[1:]))[0]
    out = []
    for i in idx:
        qi = q[i] - xs[i] * (q[i + 1] - q[i]) / (xs[i + 1] - xs[i])
        jac = abs(1 - t * np.cos(qi))                                        # dx/dq
        out.append((1 / jac, -np.sin(qi)))                                    # (density of stream, velocity)
    return out
def pressure(t, x):
    s = streams(t, x)
    rho = sum(r for r, _ in s); vbar = sum(r * v for r, v in s) / rho
    return len(s), sum(r * (v - vbar)**2 for r, v in s)
before = [pressure(0.5, x) for x in (0.0, 0.3, 1.0)]
after = [pressure(2.0, x) for x in (0.0, 0.3)]
ok(f"K3 t = 0.5 (before crossing): streams/pressure {[(n_, round(p, 6)) for n_, p in before]}; t = 2 (after): {[(n_, round(p, 4)) for n_, p in after]}",
   all(n_ == 1 and p < 1e-12 for n_, p in before) and all(n_ >= 3 and p > 1e-3 for n_, p in after))

# K4 the law: Hamiltonian flow on phase space preserves omega = dp^dx (symplectic) and hence phase volume (SDiff of phase space).
#     K4a free streaming (x, p) -> (x + t p/m, p) in 3D and the exact harmonic flow are symplectic (J^T W J = W) with det J = 1.
t_, w_ = sp.symbols('t omega', positive=True)
Wm = sp.zeros(6); Wm[0:3, 3:6] = sp.eye(3); Wm[3:6, 0:3] = -sp.eye(3)
Jfree = sp.eye(6); Jfree[0:3, 3:6] = t_ / m * sp.eye(3)
c_, s_ = sp.cos(w_ * t_), sp.sin(w_ * t_)
Jho = sp.zeros(6)
for i in range(3):
    Jho[i, i], Jho[i, i + 3], Jho[i + 3, i], Jho[i + 3, i + 3] = c_, s_ / (m * w_), -m * w_ * s_, c_
ok("K4a free streaming and harmonic flow: J^T W J = W (omega preserved), det J = 1 (phase volume preserved = SDiff of phase space)",
   sp.simplify(Jfree.T * Wm * Jfree - Wm) == sp.zeros(6) and sp.simplify(Jho.T * Wm * Jho - Wm) == sp.zeros(6)
   and sp.simplify(Jfree.det()) == 1 and sp.simplify(Jho.det()) == 1)
# K4b a single stream is the graph p = m u(x). omega pulled back to it is m d(u_mu dx^mu) = m Omega: the graph is Lagrangian iff the flow is
#     irrotational. So SDiff of labels (one stream) = a 3-dim submanifold of phase space carried by the flow; its vorticity is the obstruction.
X3 = sp.symbols('x y z')
uf = [sp.Function(f'U{i}')(*X3) for i in range(3)]
pull = sp.Matrix(3, 3, lambda a_, b_: m * (sp.diff(uf[b_], X3[a_]) - sp.diff(uf[a_], X3[b_])))   # sum_i dp_i ^ dx^i on p_i = m U_i(x)
curl_free = {uf[0]: sp.diff(sp.Function('S')(*X3), X3[0]), uf[1]: sp.diff(sp.Function('S')(*X3), X3[1]), uf[2]: sp.diff(sp.Function('S')(*X3), X3[2])}
ok("K4b omega on the single-stream graph p = m u(x) equals m (du): zero for u = grad S (Lagrangian), nonzero with vorticity",
   pull.subs(curl_free).doit().applyfunc(sp.simplify) == sp.zeros(3) and pull != sp.zeros(3))
