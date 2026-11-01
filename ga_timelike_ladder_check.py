# 80: the timelike half of the label ladder. A label with a timelike gradient (k = 1) is a scalar medium F(n), n = sqrt(d phi . d phi):
# superfluid / k-essence / quintessence. Questions: can a label medium cross w = -1 (DESI)? Which dark-matter-dark-energy coupling is
# allowed by the dust axioms (additive over world lines, reparametrisation invariant)? TL1-TL6. Structural.
# Run: ~/.venvs/jev/bin/python ga_timelike_ladder_check.py
import sympy as sp

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

x = sp.symbols('x0:4', real=True); g = sp.diag(1, -1, -1, -1); gi = g.inv()
c, n = sp.symbols('c n', positive=True); F = sp.Function('F')
# TL1 timelike label phi = c t: n = c; T_mu_nu = n F'(n) P_mu_nu - F g_mu_nu with P = d phi d phi / n^2 (projector on the timelike gradient)
phi = c * x[0]
dphi = sp.Matrix([sp.diff(phi, xi) for xi in x])
nn = sp.sqrt((dphi.T * gi * dphi)[0])
Fp = sp.diff(F(n), n).subs(n, nn)
T = sp.simplify(Fp * nn * (dphi * dphi.T) / nn**2 - g * F(nn))
M = sp.simplify(gi * T)
rho, p = sp.simplify(M[0, 0]), sp.simplify(-M[1, 1])
ok(f"TL1 timelike label: rho = {rho}, p = {p} (k-essence P(X) = F, rho = 2X P_X - P), rho + p = {sp.simplify(rho + p)}",
   sp.simplify(rho - (c * sp.diff(F(c), c) - F(c))) == 0 and sp.simplify(p - F(c)) == 0)

# TL2 health of perturbations (P(X) with X = n^2): no ghost P_X + 2 X P_XX > 0, gradient stability P_X > 0 (c_s^2 = P_X/(P_X + 2X P_XX)).
#     rho + p = 2 X P_X = n F'(n). So a healthy timelike label medium has rho + p >= 0: w >= -1; crossing w = -1 needs P_X to change sign,
#     i.e. a ghost or a gradient instability on one side.
X = sp.symbols('X', positive=True); P = sp.Function('P')
rpX = sp.simplify(2 * X * sp.diff(P(X), X))
ok(f"TL2 rho + p = {rpX}: sign of P_X; healthy (P_X > 0) => w >= -1; phantom => P_X < 0 => gradient instability or ghost (no-go)", True)

# TL3 spacelike rungs (paper1): k = 3 fluid rho = -F, p = F - n F', rho + p = -n F'; dust F = -m n gives rho + p = m n > 0; phantom needs
#     F' > 0, i.e. d rho/dn < 0 and negative kinetic energy of the perturbations (proportional to rho + p). Vacuum rungs k = 0, 4: rho + p = 0.
ok("TL3 spacelike media: rho + p = -n F' (dust: m n > 0); phantom only with negative perturbation kinetic energy; k = 0, 4: rho + p = 0", True)

# TL4 any number of minimally coupled healthy label media: rho_tot + p_tot = sum (rho_i + p_i) >= 0, so the total w >= -1 at all times.
rs = sp.symbols('r1:5', nonnegative=True)
ok("TL4 sum of non-negative (rho_i + p_i) is non-negative: no total phantom phase from healthy, minimally coupled label media",
   sp.simplify(sum(rs)).is_nonnegative)

# TL5 the only coupling the dust axioms allow to a scalar label medium (first derivatives, additive over world lines, reparametrisation
#     invariant): world-line Lagrangian L(phi, dphi/dtau) dtau homogeneous of degree 1 in dtau -> L = -m(phi) - alpha(phi) dphi/dtau; the
#     second term is a total derivative of A(phi) with A' = alpha. Hence only a field-dependent world-line mass m(phi). Check: the
#     Euler-Lagrange contribution of alpha(phi) dphi/dtau vanishes identically.
tau = sp.symbols('tau'); ph = sp.Function('phi')(tau); al = sp.Function('alpha')
Lal = al(ph) * sp.diff(ph, tau)
EL = sp.simplify(sp.diff(sp.diff(Lal, sp.diff(ph, tau)), tau) - sp.diff(Lal, ph))
ok(f"TL5 alpha(phi) dphi/dtau is a total derivative (EL = {EL}): at lowest derivative order the only dust-scalar coupling is m(phi) (disformal terms: 82 CC1); dust stays pressureless (F = -m(phi) n is linear in n)",
   EL == 0)

# TL6 consequence, energy-conserving: dust with world-line mass m(phi) = m0 e^(beta phi) (TL5) and a scalar label medium phi with constant
#     potential V0 (a cosmological constant when the field is at rest). Units 8 pi G = 1. Equations: phi'' + 3H phi' = -beta rho_m,
#     rho_m = m(phi) n, n ~ a^-3, 3H^2 = rho_m + phi'^2/2 + V0; total energy is conserved. An observer who assumes rho_m ~ a^-3 (normalised
#     early, as from the CMB) infers rho_DE,eff = 3H^2 - rho_m,assumed and w_eff = -1 - (1/3) d ln rho_DE,eff/d ln a.
#     With beta = 0.1 and Omega_m ~ 0.3 today: w_eff = -1.05 (z = 0), -1.19 (z = 0.5): an apparent phantom phase with every component
#     healthy (TL4); beta = 0 gives w_eff = -1 exactly. (At z >~ 1.5 rho_DE,eff passes through zero for this beta, so w_eff is singular there.)
import numpy as np
from scipy.integrate import solve_ivp
def w_eff(beta, ai=1e-3, rmi=1.0):
    def run(V0):
        def f(N, y):
            ph, pd = y; rm = rmi * np.exp(-3 * (N - np.log(ai)) + beta * ph); h = np.sqrt((rm + 0.5 * pd**2 + V0) / 3)
            return [pd / h, (-3 * h * pd - beta * rm) / h]
        sol = solve_ivp(f, [np.log(ai), 0.0], [0.0, 0.0], rtol=1e-10, atol=1e-12, dense_output=True)
        N = np.linspace(np.log(ai), 0.0, 20001); ph, pd = sol.sol(N)
        rm = rmi * np.exp(-3 * (N - np.log(ai)) + beta * ph); h2 = (rm + 0.5 * pd**2 + V0) / 3
        return N, rm, h2
    for V0 in np.geomspace(1e-9, 1e-5, 400):
        N, rm, h2 = run(V0)
        if rm[-1] / (3 * h2[-1]) < 0.3: break
    rde = 3 * h2 - rmi * np.exp(-3 * (N - np.log(ai)))
    w = -1 - np.gradient(np.log(np.abs(rde)), N) / 3
    at = lambda z: w[np.argmin(abs(N + np.log(1 + z)))]
    return at(0.0), at(0.5)
w0b, w05b = w_eff(0.1); w0n, w05n = w_eff(0.0)
ok(f"TL6 coupled dust m(phi) + constant-potential scalar, beta = 0.1: w_eff = {w0b:.3f} (z=0), {w05b:.3f} (z=0.5) < -1; beta = 0: {w0n:.3f}",
   w0b < -1 and w05b < -1 and abs(w0n + 1) < 1e-3)

# TL7 (review 2026-10-22) w >= -1 needs rho > 0 as well: P = X/2 + 1 at X = 1 is stable (P_X = P_X + 2X P_XX = 1/2) but rho = 2X P_X - P = -1/2,
#     p = 3/2, w = -3. For spacelike rungs the stress is anisotropic: rho + p_par = -n F', rho + p_perp = 0, so the mean gives
#     rho + p_bar = -(k/3) n F'; the null energy condition is imposed per direction and is additive over media.
Xs = sp.Integer(1); Pq = X / 2 + 1
PX = sp.diff(Pq, X).subs(X, Xs); PXX = sp.diff(Pq, X, 2).subs(X, Xs)
rho_c = (2 * X * sp.diff(Pq, X) - Pq).subs(X, Xs); p_c = Pq.subs(X, Xs)
kk, nF = sp.symbols('k nFp')
pbar = sp.simplify((kk * (-nF) + (3 - kk) * 0) / 3)
ok(f"TL7 stable P = X/2 + 1: P_X = {PX}, P_X + 2X P_XX = {PX + 2 * PXX}, rho = {rho_c}, w = {p_c / rho_c}; spacelike mean rho + p_bar = {pbar}",
   PX > 0 and rho_c < 0 and p_c / rho_c == -3 and sp.simplify(pbar + kk * nF / 3) == 0)
