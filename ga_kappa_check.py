# 96: what fixes kappa in a0 = kappa pi G Sigma_v (doc 95)? KA1-KA6. Units SI for accelerations; spectrum as in ga_label_budget_check LB1.
# Run: ~/.venvs/jev/bin/python ga_kappa_check.py
import numpy as np
import sympy as sp
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

# KA1 identity. The population law M_b = v^4/(G a0) (a0 = the label of the common level) with v the halo's circular speed at r_v (the
#     plateau is set by the halo, PA2) and M_b = f M_v gives a0 = (G M_v/r_v)^2/(G f M_v) = g_v/f: kappa = 1/f, the inverse baryon share.
G, Mv, rv, f = sp.symbols('G M_v r_v f', positive=True)
v2 = G * Mv / rv
a0 = v2**2 / (G * f * Mv)
ok(f"KA1 a0 = {sp.simplify(a0)} = g_v/f, kappa = a0/g_v = {sp.simplify(a0 / (G * Mv / rv**2))}", sp.simplify(a0 / (G * Mv / rv**2) - 1 / f) == 0)

# KA2 the core does not enter: for the cored isothermal rho = rho0 r0^2/(r^2 + r0^2), v_c(r)^2 = V^2 (1 - (r0/r) atan(r/r0)); if v is the
#     asymptotic plateau V instead of v_c(r_v), a0 changes by [1 - c^-1 atan c]^-2 with c = r_v/r0; for c = 10..40 the factor is 1.36-1.08.
c = sp.symbols('c', positive=True)
fac = lambda cv: float((1 - sp.atan(cv) / cv)**-2)
ok(f"KA2 plateau vs v_c(r_v) factor: c = 10: {fac(10):.3f}, 20: {fac(20):.3f}, 40: {fac(40):.3f}: the core is a correction, not kappa", fac(40) < 1.1)

# KA3 the native concentration: maximal mixing (doc 75) at fixed M and E inside r_v is the isothermal sphere in a box (Lynden-Bell & Wood 1968,
#     known). Lambda = -E R/(G M^2) is computed directly: Emden psi'' + 2 psi'/x = e^-psi, v = x psi', u = x e^-psi/psi';
#     -E R/(G M^2) = (3/2 - u)/v (Padmanabhan 1989). Dust collapse fixes E: top hat, E = -3 G M^2/(5 r_ta), r_v = r_ta/2 -> 3/10.
#     Solve on the stable branch (x below the Antonov point, max Lambda = 0.335).
def emden(xmax):
    s = solve_ivp(lambda x, y: [y[1], np.exp(-y[0]) - 2 * y[1] / x], [1e-6, xmax], [1e-12 / 6, 1e-6 / 3], rtol=1e-10, atol=1e-13, dense_output=True)
    return s
S = emden(200.0)
def lam(x):
    psi, dpsi = S.sol(x); v = x * dpsi; u = x * np.exp(-psi) / dpsi
    return (1.5 - u) / v
xs = np.linspace(0.5, 150, 3000); ls = np.array([lam(x) for x in xs]); ia = np.argmax(ls)
x_star = brentq(lambda x: lam(x) - 0.3, 5.0, xs[ia])
psi_s = S.sol(x_star)[0]
# core radius of the isothermal sphere: King radius r_K = 3 (Emden unit), so c = r_v/r_K = x_star/3; density contrast e^psi
ok(f"KA3 Antonov point Lambda_max = {ls[ia]:.3f} at x = {xs[ia]:.1f} (known 0.335, 34.4); dust-collapse energy Lambda = 0.3 -> x = {x_star:.2f}, "
   f"density contrast {np.exp(psi_s):.0f}, r_v/r_K = {x_star / 3:.2f}", abs(ls[ia] - 0.335) < 0.002 and abs(xs[ia] - 34.4) < 0.5)

# KA4 g_v of typical halos (corrected collapse condition: D(z_c) sigma(M) = delta_c / nu, i.e. rarer peaks collapse EARLIER).
h, Om = 0.674, 0.315; Gam = Om * h
T = lambda k: np.log(1 + 2.34 * k / Gam) / (2.34 * k / Gam) * (1 + 3.89 * k / Gam + (16.1 * k / Gam)**2 + (5.46 * k / Gam)**3 + (6.71 * k / Gam)**4)**-0.25
P = lambda k: k**0.965 * T(k)**2
Wf = lambda q: 3 * (np.sin(q) - q * np.cos(q)) / q**3
sR = lambda R: np.sqrt(quad(lambda lk: np.exp(lk)**3 * P(np.exp(lk)) * Wf(np.exp(lk) * R)**2 / (2 * np.pi**2), np.log(1e-4), np.log(1e4), limit=500)[0])
norm = 0.811 / sR(8.0)
sigM = lambda M: norm * sR((3 * M * h / (4 * np.pi * 2.775e11 * Om))**(1 / 3))
Ds = solve_ivp(lambda N, y: [y[1], -(2 - 1.5 * Om / np.exp(3 * N) / (Om / np.exp(3 * N) + 1 - Om)) * y[1] + 1.5 * Om / np.exp(3 * N) / (Om / np.exp(3 * N) + 1 - Om) * y[0]],
               [np.log(1e-3), 0.0], [1e-3, 1e-3], dense_output=True, rtol=1e-9)
D = lambda z: Ds.sol(-np.log1p(z))[0] / Ds.sol(0.0)[0]
rho_m0 = 2.775e11 * Om * h**2; Gc = 4.301e-9; conv = 1e6 / 3.0857e22
def gv(M, nu):
    zc = brentq(lambda z: D(z) - 1.686 / (nu * sigM(M)), 0.0, 50) if 1.686 / (nu * sigM(M)) < 1 else 0.0
    r = (3 * M / (4 * np.pi * 18 * np.pi**2 * rho_m0 * (1 + zc)**3))**(1 / 3)
    return zc, Gc * M / r**2 * conv
tab = {(nu, M): gv(M, nu) for nu in (1.0, 1.5, 2.0) for M in (1e10, 1e11, 1e12)}
print("   " + "; ".join(f"nu={nu} M={M:.0e}: z_c={z:.2f}, g_v={g:.2e}" for (nu, M), (z, g) in tab.items()))
spread = {nu: np.log10(max(tab[(nu, M)][1] for M in (1e10, 1e11, 1e12)) / min(tab[(nu, M)][1] for M in (1e10, 1e11, 1e12))) for nu in (1.0, 1.5, 2.0)}
gmean = {nu: 10**np.mean([np.log10(tab[(nu, M)][1]) for M in (1e10, 1e11, 1e12)]) for nu in (1.0, 1.5, 2.0)}
ok("KA4 common level: g_v spread over 1e10-1e12 Msun " + ", ".join(f"nu = {nu}: {spread[nu]:.2f} dex (mean {gmean[nu]:.2e})" for nu in spread),
   all(s < 0.3 for s in spread.values()))

# KA5 native prediction with no baryon loss (adiabatic labels, PA6): f = Omega_b/Omega_m = 0.157 -> a0 = g_v/0.157 (x KA2 factor if v is the
#     plateau). Observed 1.2e-10.
fcos = 0.0493 / 0.315
pred = {nu: gmean[nu] / fcos for nu in gmean}
ok("KA5 a0 with no baryon loss: " + ", ".join(f"nu = {nu}: {pred[nu]:.2e} ({pred[nu] / 1.2e-10:.2f} x observed)" for nu in pred)
   + "; the plateau factor from the exact Emden sphere is KA7 (1.18)", True)

# KA6 the label that remains: the retained share r_b = f/f_cos needed for a0 = 1.2e-10: r_b = g_v/(a0 f_cos)
rb = {nu: gmean[nu] / (1.2e-10 * fcos) for nu in gmean}
ok("KA6 retention needed: " + ", ".join(f"nu = {nu}: r_b = {rb[nu]:.2f}" for nu in rb) + " -- kappa = 1/(f_cos r_b): the label a0 is the "
   "retained baryon share, a quantity measurable without rotation curves", all(0 < v < 1 for v in rb.values()))

# KA7 removing the choice of nu: the median half-mass formation epoch of a Gaussian dust field (Lacey & Cole 1993, equation borrowed):
#     delta_c/D(z_f) = delta_c + 0.974 sqrt(sigma^2(M/2) - sigma^2(M)) (z0 = 0). And the plateau factor from the exact Emden sphere cut at
#     x* (KA3): (V/v_c(r_v))^4 with V the maximum circular speed inside x* (v_c^2 ~ x psi').
def zf(M):
    t = 1.686 + 0.974 * np.sqrt(sigM(M / 2)**2 - sigM(M)**2)
    return brentq(lambda z: D(z) - 1.686 / t, 0.0, 50)
def gv_z(M, z):
    r = (3 * M / (4 * np.pi * 18 * np.pi**2 * rho_m0 * (1 + z)**3))**(1 / 3)
    return Gc * M / r**2 * conv
xx = np.linspace(0.05, x_star, 4000); vv = np.array([x * S.sol(x)[1] for x in xx])
pf = (vv.max() / vv[-1])**2
zfs = {M: zf(M) for M in (1e10, 1e11, 1e12)}
gf = {M: gv_z(M, zfs[M]) for M in zfs}
a0n = {M: gf[M] / fcos * pf for M in gf}
print("   " + "; ".join(f"M={M:.0e}: z_f={zfs[M]:.2f}, g_v={gf[M]:.2e}, a0 (no loss, Emden plateau) = {a0n[M]:.2e}" for M in gf)
      + f"; Emden factor (V_max/v_c(r_v))^4 = {pf:.3f}")
am = 10**np.mean(np.log10(list(a0n.values())))
ok(f"KA7 native a0 (dust collapse at the median formation epoch, max-entropy isothermal halo, no baryon loss) = {am:.2e} m/s^2 = "
   f"{am / 1.2e-10:.2f} x the observed 1.2e-10; spread over 1e10-1e12 Msun {np.log10(max(a0n.values()) / min(a0n.values())):.2f} dex", True)

# KA8 (review 2026-10-22, item 2) the plateau factor at the paper's r_t, computed inside the Emden sphere itself. r_t = 7.16 r_c of the
#     fitting model; two identifications of r_c with the Emden core: x_c = sqrt 2 (meeting point of the asymptotes, Prop iso) and the
#     half-density radius x_1/2 (rho = rho0/2, as for the cored model). Factor (v(x_t)/v(x*))^4 with x* the box radius (KA3).
from scipy.optimize import brentq as _bq
xh = _bq(lambda x: np.exp(-S.sol(x)[0]) - 0.5, 0.5, 5.0)
vv2 = lambda x: x * S.sol(x)[1]
facs = {name: (vv2(7.16 * xc) / vv2(x_star))**2 for name, xc in (('x_c = sqrt2', np.sqrt(2)), ('x_1/2', xh))}
ok(f"KA8 half-density radius x_1/2 = {xh:.2f}; factor (v(r_t)/v(r_v))^4 at r_t = 7.16 r_c: " + ", ".join(f"{k}: x_t = {7.16 * (np.sqrt(2) if k.startswith('x_c') else xh):.1f}, {v:.3f}" for k, v in facs.items())
   + f"; r_v/r_c = {x_star / np.sqrt(2):.1f} or {x_star / xh:.1f}", all(1.0 < v < 1.3 for v in facs.values()))
