# 75: what selects the level of uniformity of 74 (k = 3 inside, k = 1 outside)? Candidate inside the framework: the phase-space SDiff of
# collisionless dust (paper1: Vlasov preserves phase-space volume; the relabelling of one stream is its restriction). Mixing under this
# constraint, at maximal coarse-grained entropy, gives a uniform velocity dispersion (isothermal state). IS1-IS3.
# Run: ~/.venvs/jev/bin/python ga_isothermal_label_check.py
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

# IS1 uniform dispersion sigma: hydrostatics d(rho sigma^2)/dr = -rho G M/r^2 gives rho ~ exp(-Phi/sigma^2) (Boltzmann factor of the
#     maximally mixed state); in Emden units d/dx(x^2 dpsi/dx) = x^2 exp(-psi), rho = exp(-psi).
sol = solve_ivp(lambda x, y: [y[1], np.exp(-y[0]) - 2 * y[1] / x], [1e-6, 1e4], [0.0, 0.0], rtol=1e-10, atol=1e-12, dense_output=True)
rho = lambda x: np.exp(-sol.sol(x)[0])
x_in, x_out = 0.05, 3e3
ok(f"IS1 isothermal sphere: rho(0.05) = {rho(x_in):.4f} (core: n3 ~ const, the k = 3 uniformity); x^2 rho at x = 3e3: {x_out**2 * rho(x_out):.3f} "
   f"(-> 2: n3 ~ 1/r^2, the k = 1 uniformity)", abs(rho(x_in) - 1) < 1e-3 and abs(x_out**2 * rho(x_out) - 2) < 0.1)

# IS2 one state, both levels: the radial label psi_M = M(x) (74, LF3) has n1 = dM/dx = x^2 rho -> 0 at the centre (ball, n1 ~ x^2) and -> 2
#     (constant) outside. The two uniformities of 74 meet where n2 = c3/c1, x_c = sqrt(c1/c3) = sqrt(2); the density there is
#     rho(x_c) (between the two asymptotes).
n1 = lambda x: x**2 * rho(x)
xc = np.sqrt(2.0)
ok(f"IS2 n1 = x^2 rho: {n1(0.05):.4f} at 0.05 (~x^2), {n1(xc):.3f} at x_c = sqrt 2, {n1(3e3):.3f} at 3e3 (-> 2); rho(x_c) = {rho(xc):.3f}",
   abs(n1(0.05) / 0.05**2 - 1) < 1e-2 and abs(n1(3e3) - 2) < 0.1 and 0.3 < rho(xc) < 1)

# IS3 the selector is a uniformity of the VELOCITY measure (sigma = const), not of the density: in Emden units the core radius and the
#     plateau level are set by the same sigma and central density, so k = 3 and k = 1 uniformities are two limits of one isothermal
#     state. Symbolic check: rho = sigma^2/(2 pi G r^2) solves the hydrostatic equation with the same sigma (the outer limit).
r, G, s = sp.symbols('r G sigma', positive=True)
rs = s**2 / (2 * sp.pi * G * r**2); Ms = sp.integrate(4 * sp.pi * r**2 * rs, (r, 0, r))
ok("IS3 the outer limit rho = sigma^2/(2 pi G r^2) solves d(rho sigma^2)/dr = -rho G M/r^2: core and plateau are limits of one uniform-sigma state",
   sp.simplify(sp.diff(rs * s**2, r) + rs * G * Ms / r**2) == 0)

# IS4 (review 2026-10-22) circular speed along the isothermal solution, v^2/sigma^2 = x psi'(x): 0.55 at x_c = sqrt 2 (far below the
#     plateau), overshoot 2.51 at x = 10, -> 2 from above. The intersection of asymptotes is not the onset of the plateau.
vs = lambda x: x * sol.sol(x)[1]
ok(f"IS4 v^2/sigma^2: {vs(np.sqrt(2)):.3f} at sqrt 2, {vs(10):.3f} at 10, {vs(1e3):.3f} at 1e3",
   abs(vs(np.sqrt(2)) - 0.555) < 0.005 and abs(vs(10) - 2.51) < 0.01 and abs(vs(1e3) - 2) < 0.06)

# IS5 (review 2026-10-22) onset of the OUTER flat part: smallest x beyond which |d ln v/d ln r| < eps everywhere (to x = 3e4):
#     eps = 0.1 -> x ~ 28, v^2/sigma^2 ~ 2.12; eps = 0.05 -> x ~ 56, v^2/sigma^2 ~ 1.89.
sol5 = solve_ivp(lambda x, y: [y[1], np.exp(-y[0]) - 2 * y[1] / x], [1e-6, 1e5], [0.0, 0.0], rtol=1e-11, atol=1e-13, dense_output=True)
xs = np.logspace(-1, 4.5, 60001); v2s = xs * sol5.sol(xs)[1]; sls = np.abs(0.5 * np.gradient(np.log(v2s), np.log(xs)))
onset = {e: (xs[np.where(sls >= e)[0][-1] + 1], v2s[np.where(sls >= e)[0][-1] + 1]) for e in (0.1, 0.05)}
ok(f"IS5 outer-plateau onset: eps 0.1 -> x = {onset[0.1][0]:.1f}, v^2/s^2 = {onset[0.1][1]:.2f}; eps 0.05 -> x = {onset[0.05][0]:.1f}, {onset[0.05][1]:.2f}",
   abs(onset[0.1][0] - 28) < 1.5 and abs(onset[0.05][0] - 56) < 2.5)

# IS6 (review 2026-10-22) r_t from one fitted model instead of the data depth: cored profile rho ~ (r^2 + r_c^2)^-1 has
#     v^2/v_inf^2 = 1 - (r_c/r) atan(r/r_c); the radius at fraction f of v_inf^2 is 7.16 r_c (f = 0.8) and 15.04 r_c (f = 0.9).
from scipy.optimize import brentq
yf = {f: brentq(lambda y: 1 - np.arctan(y) / y - f, 0.5, 1e4) for f in (0.8, 0.9)}
ok(f"IS6 cored model: f = 0.8 -> r_t = {yf[0.8]:.2f} r_c, f = 0.9 -> r_t = {yf[0.9]:.2f} r_c", abs(yf[0.8] - 7.16) < 0.01 and abs(yf[0.9] - 15.04) < 0.01)
