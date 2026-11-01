# The dark sector from GA symmetries: what each symmetry forces, checked (papers/restart/43-dark-sector.md). K1-K6.
# Dark energy: Lorentz invariance of the vacuum (K1) and the spacetime-volume SDiff (unimodular, K2). Dark matter: reparametrisation +
# relabelling (K3), phase-space SDiff (K4, with 34 R2-R3). Background test (K5). Darkness itself (K6).
# Run: uv run --with sympy --with numpy --with scipy python ga_dark_sector_check.py
import itertools
import numpy as np
import sympy as sp
from ga_core import *

STA = {0: 1, 1: -1, 2: -1, 3: -1}

# K1 w = -1: a stress-energy map T(a) = T^mu_nu that commutes with every Lorentz rotor (T(R a R~) = R T(a) R~) must be a multiple of the
#    identity, T^mu_nu = c delta^mu_nu: rho = -p, w = -1 (the only Lorentz-invariant medium). Check: commute with all six
#    generators of so(1,3) acting on vectors (a -> B.a).
Tm = sp.Matrix(4, 4, lambda i, j: sp.Symbol(f't{i}{j}'))
def gen(i, j):                                       # matrix of a -> (e_i ^ e_j) . a in the reciprocal frame
    M = sp.zeros(4, 4)
    for k in range(4):
        B = {(i, j): 1}
        res = grade(gp(B, {(k,): 1}, STA), 1)       # B a = B.a + B^a; grade-1 part is B.a (up to sign convention)
        for (m,), c in res.items():
            M[m, k] += c
    return M
eqs = []
for i, j in itertools.combinations(range(4), 2):
    G = gen(i, j)
    eqs += list(Tm * G - G * Tm)
sol = sp.solve(eqs, list(Tm), dict=True)[0]
Tsol = Tm.subs(sol)
free = sorted(Tsol.free_symbols, key=str)
ok(f"K1 maps commuting with all of so(1,3): T = {Tsol.tolist()} -> proportional to the identity ({len(free)} free constant): T^mu_nu = c delta, w = -1",
   len(free) == 1 and Tsol == free[0] * sp.eye(4))

# K2 Lambda constant: the volume-preserving (unimodular) theory keeps only the trace-free Einstein equation. In FLRW that is
#    dH/dt = -(kappa/2)(rho + p); with conservation d rho/dt = -3H(rho + p) one gets d/dt (H^2 - kappa rho/3) = 0: H^2 = kappa rho/3 + Lambda/3
#    with Lambda an integration constant -- a conserved charge, not a parameter; the vacuum energy (rho_vac = -p_vac) drops out of dH/dt.
t, kap = sp.symbols('t kappa', positive=True)
Hf, rho, pr = sp.Function('H')(t), sp.Function('rho')(t), sp.Function('p')(t)
dH = -kap / 2 * (rho + pr); drho = -3 * Hf * (rho + pr)
expr = sp.simplify(2 * Hf * dH - kap / 3 * drho)
vac = sp.simplify(dH.subs(pr, -rho))
ok(f"K2 (flat FLRW) d/dt(H^2 - kappa rho/3) = {expr} (Lambda/3 conserved); vacuum (p = -rho) in dH/dt: {vac} (does not source the trace-free equation)",
   expr == 0 and vac == 0)

# K3 cold dark matter: a relabelling-invariant fluid has L = -F(n) with p = n F'(n) - F (here sign: rho = F, p = n F' - F); p = 0 iff F is
#    linear in n, i.e. iff the action is the sum of independent world-line actions (non-interacting elements). Reparametrisation of a single
#    line does not select this by itself: F ~ n^(4/3) respects it and has pressure (review 2026-10-21).
n, m = sp.symbols('n m', positive=True)
Fl = sp.Function('F')(n)
p_of = lambda F: sp.simplify(n * sp.diff(F, n) - F)
ok(f"K3 F = m n -> p = {p_of(m * n)}; F = m n + c n^2 -> p = {p_of(m * n + sp.Symbol('c') * n**2)} (non-zero): pressureless <=> linear F <=> additive over world lines",
   p_of(m * n) == 0 and p_of(m * n + sp.Symbol('c') * n**2) != 0)

# K4 collisionless: phase-space SDiff (Liouville) keeps every Casimir int C(f); the maximal-entropy state under it is Fermi-Dirac (34 R2),
#    whose self-gravitating sphere has a solid-body core and a flat isothermal envelope (34 R3). Here: the Liouville flow of a Hamiltonian
#    keeps phase-space area (divergence-free): div(dH/dp, -dH/dx) = 0 for any H.
x, p = sp.symbols('x p', real=True)
Hx = sp.Function('Hh')(x, p)
ok(f"K4 phase-space velocity (dH/dp, -dH/dx) has divergence {sp.simplify(sp.diff(sp.diff(Hx, p), x) + sp.diff(-sp.diff(Hx, x), p))}: SDiff of phase space",
   sp.simplify(sp.diff(sp.diff(Hx, p), x) + sp.diff(-sp.diff(Hx, x), p)) == 0)

# K5 the background that K1-K3 give (dust w = 0 + constant Lambda w = -1, flat) against DESI DR2 BAO D_M/D_H (r_d-free): one parameter Omega_m.
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
z = np.array([0.510, 0.706, 0.934, 1.321, 1.484, 2.330])
F = np.array([0.622, 0.892, 1.223, 1.948, 2.386, 4.518]); sF = np.array([0.017, 0.021, 0.019, 0.045, 0.136, 0.097])
def Fm(Om):
    E = lambda q: np.sqrt(Om * (1 + q)**3 + 1 - Om)
    return np.array([E(zz) * quad(lambda q: 1 / E(q), 0, zz)[0] for zz in z])
best = minimize_scalar(lambda Om: np.sum(((Fm(Om) - F) / sF)**2), bounds=(0.1, 0.6), method='bounded')
ok(f"K5 dust + constant Lambda: chi2 = {best.fun:.1f} for 6 DESI DR2 points (Omega_m = {best.x:.3f}): the symmetric background fits", best.fun < 12)

# K6 gauge symmetry alone does not forbid a charge: N = dX1^dX2^dX3 is closed, so A^N is gauge-invariant (d chi ^ N = d(chi N)).
#    What forbids it is the label-orientation symmetry (K7).
X = sp.symbols('x0:4', real=True)
X1, X2, X3, chi = [sp.Function(nm)(*X) for nm in ('X1', 'X2', 'X3', 'chi')]
d = lambda f: {(i,): sp.diff(f, X[i]) for i in range(4)}
N = wedge(wedge(d(X1), d(X2), STA), d(X3), STA)
dN = ext_d(N, X, STA)
lhs = ext_d(scal(chi, N), X, STA); rhs = wedge(d(chi), N, STA)
nz = lambda M: {k: sp.simplify(v) for k, v in M.items() if sp.simplify(v) != 0}
ok(f"K6 dN = {nz(dN)}; d(chi N) - dchi^N = "
   f"{nz(add(lhs, scal(-1, rhs)))}: A^N is gauge invariant -> gauge symmetry alone allows a charge (see K7)",
   all(sp.simplify(v) == 0 for v in dN.values()) and all(sp.simplify(v) == 0 for v in add(lhs, scal(-1, rhs)).values()))

# K1' (what K1 derives): the general 4x4 T^mu_nu was left free (16 unknowns) and only commutation with the six so(1,3) generators was imposed;
#    the solution space is 1-dimensional (the identity). A perfect fluid with p != -rho (T = diag(rho, -p, -p, -p) in the rest frame) fails it:
rho_s, p_s = sp.symbols('rho p_', real=True)
Tf = sp.diag(rho_s, -p_s, -p_s, -p_s)
viol = [sp.simplify(e) for i, j in itertools.combinations(range(4), 2) for e in (Tf * gen(i, j) - gen(i, j) * Tf) if sp.simplify(e) != 0]
ok(f"K1' fluid T = diag(rho, -p, -p, -p) violates boost invariance unless {sp.solve(viol[0], p_s)} (p = -rho): w = -1 is derived, not assumed",
   len(viol) > 0 and all(sp.simplify(v.subs(p_s, -rho_s)) == 0 for v in viol))

# K2' an exact solution family: Schwarzschild-de Sitter f = 1 - 2M/r - L r^2/3 solves the trace-free vacuum equation R_ab - R g_ab/4 = 0
#    for EVERY L (and M): Lambda is not fixed by the unimodular field equation -- it labels the solution (integration constant / superselection).
tt, rr, th, ph, M, L = sp.symbols('t r theta phi M L', positive=True)
coords = [tt, rr, th, ph]
fS = 1 - 2 * M / rr - L * rr**2 / 3
g = sp.diag(fS, -1 / fS, -rr**2, -rr**2 * sp.sin(th)**2)
ginv = g.inv()
Gam = [[[sum(ginv[a, d] * (sp.diff(g[d, b], coords[c]) + sp.diff(g[d, c], coords[b]) - sp.diff(g[b, c], coords[d])) for d in range(4)) / 2
         for c in range(4)] for b in range(4)] for a in range(4)]
def Ric(b, c):
    return sp.simplify(sum(sp.diff(Gam[a][b][c], coords[a]) - sp.diff(Gam[a][b][a], coords[c])
                           + sum(Gam[a][a][d] * Gam[d][b][c] - Gam[a][c][d] * Gam[d][b][a] for d in range(4)) for a in range(4)))
Rab = sp.Matrix(4, 4, lambda b, c: Ric(b, c))
Rs = sp.simplify(sum(ginv[a, b] * Rab[a, b] for a in range(4) for b in range(4)))
tf = sp.simplify(Rab - Rs * g / 4)
ok(f"K2' SdS: R = {Rs} (= -4L in signature +,-,-,-), trace-free part = 0 for all L, M: {tf == sp.zeros(4, 4)} -> Lambda is a free label of solutions",
   tf == sp.zeros(4, 4) and sp.simplify(Rs + 4 * L) == 0)

# K7 darkness FROM a symmetry: the labels X^i are names; reversing the orientation of label space (X1 -> -X1) is a symmetry of the dust action
#    F(|N|) but flips N -> -N, so the minimal coupling A^N is odd and forbidden; even couplings f(|N|) F.F (polarisability) stay allowed.
#    Baryons carry an internal phase (a rotor e^{I theta}, square -1 unit) that can be gauged; label matter has none.
Nflip = wedge(wedge(d(-X1), d(X2), STA), d(X3), STA)
ok(f"K7 X1 -> -X1 maps N -> -N: {all(sp.simplify(Nflip.get(k, 0) + N.get(k, 0)) == 0 for k in set(N) | set(Nflip))}; "
   "|N| (dust action) unchanged, A^N changes sign: the charge is forbidden by label-orientation symmetry; polarisability-type f(|N|) F.F allowed",
   all(sp.simplify(Nflip.get(k, 0) + N.get(k, 0)) == 0 for k in set(N) | set(Nflip)))

# K8 the values as superselection charges: the label charge Q = int N over a comoving volume is conserved (dN = 0, K6), baryon number too; both
#    densities go as a^-3, so Omega_c/Omega_b = (m_c Q_c)/(m_b Q_b) is constant in time -- fixed at production, not by dynamics. Lambda is the
#    unimodular charge (K2'). Given the charges, the equality epoch follows: a_eq = (Omega_m/Omega_L)^(1/3).
a = sp.symbols('a', positive=True)
Qc, Qb, mc, mb, Om, OL = sp.symbols('Q_c Q_b m_c m_b Omega_m Omega_L', positive=True)
ratio = sp.simplify((mc * Qc / a**3) / (mb * Qb / a**3))
aeq = sp.solve(sp.Eq(Om / a**3, OL), a)
ok(f"K8 Omega_c/Omega_b = {ratio} (a-independent: ratio of conserved charges); a_eq = {aeq[0]} -> {float(aeq[0].subs({Om: 0.315, OL: 0.685})):.3f} "
   "(z = 0.30): 'now' follows from the charges; why observers live near it is selection, not symmetry",
   sp.diff(ratio, a) == 0 and abs(float(aeq[0].subs({Om: 0.315, OL: 0.685})) - 0.774) < 0.01)

# K7' the label-orientation symmetry is not an extra assumption for the dust itself: the only Lorentz scalar built from the label current
#    J = *N alone is J.J (any self-action is a function of it), which is even under N -> -N. The odd coupling A.J is the lowest term that would
#    make label orientation physical; requiring all measure-preserving relabellings (|det| = 1, labels = unoriented names) as a gauge redundancy
#    removes it.
J = sp.Matrix(sp.symbols('J0:4', real=True)); Aem = sp.Matrix(sp.symbols('A0:4', real=True))
eta = sp.diag(1, -1, -1, -1)
JJ = (J.T * eta * J)[0]; AJ = (Aem.T * eta * J)[0]
ok(f"K7' J.J -> {sp.simplify(JJ.subs({J[i]: -J[i] for i in range(4)}, simultaneous=True) - JJ)} change (even); A.J -> {sp.simplify(AJ.subs({J[i]: -J[i] for i in range(4)}, simultaneous=True) + AJ)} + (-A.J) (odd)",
   sp.simplify(JJ.subs({J[i]: -J[i] for i in range(4)}, simultaneous=True) - JJ) == 0 and sp.simplify(AJ.subs({J[i]: -J[i] for i in range(4)}, simultaneous=True) + AJ) == 0)
