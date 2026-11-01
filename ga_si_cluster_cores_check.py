# 120: F5 of PREDICTIONS.md -- under the self-interaction axiom of doc 107 (elastic scattering of label elements, rho_c v (s/m) t_age ~ 1),
# the cross-section the galaxies need, carried to groups and clusters, predicts the core density and the core radius there. Three variants
# from doc 107: constant s/m = 0.10-0.13 cm^2/g; trends with v measured on galaxies, s/m ~ v^-1.0 (pISO) and v^-0.4 (Burkert), anchored at the
# sample median v = 115 km/s. Halo age: median half-mass epoch from the spectrum (doc 118 rule, EPS), t_age = t(0) - t(z_f). Core radius
# of an isothermal core (King): r_c^2 = 9 sigma^2/(4 pi G rho_c), sigma = v/sqrt(2) (NC10). Velocity v is the circular speed of the halo.
# The axiom is an axiom: these are its consequences, not derivations. SC1-SC5.
# Run: ~/.venvs/jev/bin/python ga_si_cluster_cores_check.py
import os, contextlib, io
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
HERE = os.path.dirname(os.path.abspath(__file__))
kc = {'__name__': 'kc', '__file__': os.path.join(HERE, 'ga_kappa_check.py')}
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(kc['__file__']).read(), kc)
D, sigM, ok = kc['D'], kc['sigM'], kc['ok']
Om, h, dc = 0.315, 0.674, 1.686
G, Msun, pc, kpc, Gyr = 6.674e-11, 1.989e30, 3.086e16, 3.086e19, 3.156e16
H0 = 100 * h * 1e3 / (1e6 * pc)
tcos = lambda z: quad(lambda zz: 1 / ((1 + zz) * H0 * np.sqrt(Om * (1 + zz)**3 + 1 - Om)), z, np.inf)[0]
def zf(M):
    inv = 1 / D(0.0) + 0.974 * np.sqrt(sigM(M / 2)**2 - sigM(M)**2) / dc
    return brentq(lambda zz: D(zz) - 1 / inv, 0.0, 80)
def v200(M):                                                  # circular speed at r200c today, M in Msun
    rho_c = 3 * H0**2 / (8 * np.pi * G); r = (3 * M * Msun / (4 * np.pi * 200 * rho_c))**(1 / 3)
    return np.sqrt(G * M * Msun / r) / 1e3
variants = {'constant 0.10': (0.10, 0.0), 'constant 0.13': (0.13, 0.0), 'pISO trend v^-1.0': (0.10, -1.01), 'Burkert trend v^-0.4': (0.13, -0.38)}
sm = lambda var, v: variants[var][0] * (v / 115.0)**variants[var][1]           # cm^2/g
rows = []
for lM, name in ((11.5, 'galaxy'), (13.0, 'group'), (14.0, 'poor cluster'), (15.0, 'cluster')):
    M = 10**lM; v = v200(M); z_f = zf(M); t = (tcos(0) - tcos(z_f))
    for var in variants:
        s = sm(var, v) * 1e-4 / 1e-3                                             # m^2/kg
        rho = 1 / (v * 1e3 * s * t)                                              # kg/m^3
        rc = np.sqrt(9 * (v * 1e3)**2 / 2 / (4 * np.pi * G * rho)) / kpc
        rows.append((name, lM, v, z_f, t / Gyr, var, sm(var, v), rho / (Msun / pc**3), rc))
for r in rows:
    print(f"   {r[0]:12s} M = 10^{r[1]:.1f}: v200 {r[2]:5.0f} km/s, z_f {r[3]:.2f}, t_age {r[4]:.1f} Gyr | {r[5]:22s} s/m {r[6]:.3f} cm2/g "
          f"-> rho_c {r[7]:.4f} Msun/pc^3, r_c {r[8]:6.1f} kpc")
gal = [r for r in rows if r[0] == 'galaxy' and r[5] == 'constant 0.10'][0]
ok(f"SC1 calibration: at a galaxy (v {gal[2]:.0f} km/s) the constant 0.10 cm^2/g gives rho_c = {gal[7]:.3f} Msun/pc^3, r_c = {gal[8]:.1f} kpc "
   "(the order of SPARC cores, doc 105)", 0.003 < gal[7] < 0.3)
cl = {r[5]: r for r in rows if r[0] == 'cluster'}
ok("SC2 clusters (M = 1e15): " + "; ".join(f"{k}: s/m {v[6]:.3f}, r_c {v[8]:.0f} kpc" for k, v in cl.items()), True)
gr = {r[5]: r for r in rows if r[0] == 'group'}
ok("SC3 groups (M = 1e13): " + "; ".join(f"{k}: s/m {v[6]:.3f}, r_c {v[8]:.0f} kpc" for k, v in gr.items()), True)
ratio = cl['constant 0.10'][8] / cl['pISO trend v^-1.0'][8]
ok(f"SC4 the variants separate on clusters: core radius constant / v^-1 = {ratio:.1f} -- the velocity law of s/m is measured by cluster and "
   "group cores, not by galaxies (galaxies fix only the normalisation)", ratio > 2)
ok("SC5 recorded: the scaling rho_c ~ 1/(v (s/m) t_age) and r_c ~ v/sqrt(rho_c) follow from the axiom; constant s/m gives r_c ~ v^1.5 t^0.5: "
   "cores grow with mass; in this calculation (no baryonic processes) CDM without scattering gives no core; with AGN feedback LCDM can "
   "core a cluster (Martizzi et al. 2012); the scaling law is a candidate test, not yet compared with LCDM plus baryons", True)
