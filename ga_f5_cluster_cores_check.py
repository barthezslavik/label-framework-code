# 129: F5 of PREDICTIONS.md against measured cluster cores. Data (literature, not computed here): seven massive relaxed clusters with
# strong + weak lensing and BCG stellar kinematics, cored-NFW dark-matter fits <log r_core/kpc> = 1.14 +- 0.13 (random) +0.14/-0.22 (sys)
# (Newman et al. 2013, arXiv:1209.1392; M200 ~ 0.4-2e15 Msun). The axiom's baryon-free formula (doc 120: rho_c v (s/m) t_age = 1,
# r_c^2 = 9 sigma^2/(4 pi G rho_c)) is compared; r_c ~ (s/m)^(1/2) at fixed v and age gives the cross-section the formula would need.
# Known (literature): SIDM fits that include the BCG potential give ~0.1 cm^2/g on these clusters (Kaplinghat, Tulin, Yu 2016), and LCDM
# with AGN feedback produces cluster cores of ~10 kpc (Martizzi et al. 2012). FX1-FX4.
# Run: ~/.venvs/jev/bin/python ga_f5_cluster_cores_check.py
import os, contextlib, io
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
g = {'__name__': 'sc', '__file__': os.path.join(HERE, 'ga_si_cluster_cores_check.py')}
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(g['__file__']).read(), g)
ok, rows = g['ok'], g['rows']
# common definition (review r1): the radius where the density falls to half its central value. Cored NFW at r << r_s:
# rho ~ 1/(r_core + r) -> r_half = r_core. Isothermal sphere: solve Emden's equation, xi in units sqrt(sigma^2/(4 pi G rho_0)); the King
# radius r_c of doc 120 is xi = 3.
from scipy.integrate import solve_ivp
em = solve_ivp(lambda x, y: [y[1], np.exp(-y[0]) - 2 * y[1] / x], [1e-6, 20], [0.0, 0.0], dense_output=True, rtol=1e-10, atol=1e-12)
xs = np.linspace(1e-3, 20, 200001); xh = xs[np.argmin(abs(np.exp(-em.sol(xs)[0]) - 0.5))]
conv = xh / 3.0
print(f"   isothermal sphere: half central density at xi = {xh:.3f}, i.e. r_half = {conv:.3f} x King radius")
lr, er, es_p, es_m = 1.14, 0.13, 0.14, 0.22
r_obs = 10**lr; r_hi = 10**(lr + np.hypot(er, es_p)); r_lo = 10**(lr - np.hypot(er, es_m))
print(f"   Newman et al. 2013 (cored NFW, M200 ~ 0.4-2e15): r_core = {r_obs:.0f} kpc ({r_lo:.0f}-{r_hi:.0f} with random and systematic errors)")
# review r2: the clusters are observed at z = 0.19-0.31 (MS2137 0.313, A963 0.206, A383 0.189, A611 0.288, A2537 0.297, A2667 0.233,
# A2390 0.228); age = t(z_obs) - t(z_f), z_f the median half-mass epoch of a halo of mass M observed at z_obs (doc 118 rule), and
# v200 / rho_200 at z_obs
zobs = np.array([0.313, 0.206, 0.189, 0.288, 0.297, 0.233, 0.228])
from scipy.optimize import brentq
kc = g
def zf_at(M, z):
    inv = 1 / g['D'](z) + 0.974 * np.sqrt(g['sigM'](M / 2)**2 - g['sigM'](M)**2) / 1.686
    return brentq(lambda zz: g['D'](zz) - 1 / inv, z, 80)
def v200_at(M, z):
    Hz = g['H0'] * np.sqrt(g['Om'] * (1 + z)**3 + 1 - g['Om']); rho_c = 3 * Hz**2 / (8 * np.pi * g['G'])
    r = (3 * M * g['Msun'] / (4 * np.pi * 200 * rho_c))**(1 / 3); return np.sqrt(g['G'] * M * g['Msun'] / r) / 1e3
def r_half(lM, var, z):
    M = 10**lM; v = v200_at(M, z); t = g['tcos'](z) - g['tcos'](zf_at(M, z))
    sm = g['sm'](var, v) * 1e-4 / 1e-3
    rho = 1 / (v * 1e3 * sm * t)
    return conv * np.sqrt(9 * (v * 1e3)**2 / 2 / (4 * np.pi * g['G'] * rho)) / g['kpc']
vars_ = ['constant 0.10', 'constant 0.13', 'pISO trend v^-1.0', 'Burkert trend v^-0.4']
rng_ = {k: (min(r_half(np.log10(4e14), k, z) for z in zobs), max(r_half(15.3, k, z) for z in zobs)) for k in vars_}
above = {k: a > r_hi for k, (a, b) in rng_.items()}
ok("FX1 half-density radius from the baryon-free formula at the clusters' redshifts (0.19-0.31) over M200 = 4e14-2e15 Msun: "
   + "; ".join(f"{k}: {a:.0f}-{b:.0f} kpc" for k, (a, b) in rng_.items()) + f" against {r_obs:.0f} kpc ({r_lo:.0f}-{r_hi:.0f}); entirely above "
   "the 1 sigma upper end: " + ", ".join(f"{k}: {v}" for k, v in above.items()), True)
need = 0.10 * (r_obs / r_half(15.0, 'constant 0.10', 0.25))**2
ok(f"FX2 a constant cross-section would need s/m = {need:.1e} cm^2/g at cluster speeds to give {r_obs:.0f} kpc, against 0.1 that SIDM fits with "
   "the BCG potential obtain on the same clusters with another halo model (known): the missing baryon potential is a possible "
   "explanation for constant s/m, to be checked by the same formula with that potential; the v^-1 trend of the "
   "galaxies reaches the measured cores", need < 0.01)
ok("FX3 LCDM with AGN feedback gives cores of ~10 kpc in clusters (known): the presence and the size of cluster cores do not separate the "
   "axiom from LCDM with baryons", True)
ok("FX4 recorded: F5's cluster numbers are withdrawn -- a test of the self-interaction axiom on clusters needs the isothermal core in the "
   "BCG potential (not done); on galaxy scales F5 keeps its calibration (doc 107), with no distinguishing prediction left", True)
