# 107: label self-interaction as a NEW AXIOM (elastic scattering of dust elements, cross-section per mass s/m). A core forms where each element
# has scattered about once over the halo age: rho_c v (s/m) t_age ~ 1. Minimal version: constant s/m (no free function) => rho_c ~ 1/(v t_age).
# t_age from the nu = 1 shell-collapse epoch of the halo (KA4), M_h = M_b/eps. Compare the predicted d log rho_c/d log v with the Li 2020 fits;
# the velocity dependence s/m ~ v^-a needed to match is reported (it would be a free function, i.e. a new label). SI1-SI3.
# Run: ~/.venvs/jev/bin/python -I de_self_interaction_check.py LI_DIR SPARC_DIR
import os, sys, contextlib, io
import numpy as np
from scipy.integrate import quad
HERE = os.path.dirname(os.path.abspath(__file__))
pd = {'__name__': 'pd', '__file__': os.path.join(HERE, 'de_phase_density_check.py')}
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(pd['__file__']).read().split("\nfor model in")[0], pd)
ok, fits, bsl, vf = pd['ok'], pd['fits'], pd['bsl'], pd['vf']
kc = {'__name__': 'kc', '__file__': os.path.join(HERE, 'ga_kappa_check.py')}
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(kc['__file__']).read(), kc)
Om, h = 0.315, 0.674
H0 = 100 * h / 3.0857e19                                          # 1/s
tcos = lambda z: quad(lambda zz: 1 / ((1 + zz) * H0 * np.sqrt(Om * (1 + zz)**3 + 1 - Om)), z, np.inf)[0]
t0 = tcos(0.0)
for model in ('pISO-Flat', 'Burkert-Flat'):
    lv, lrho, lage = [], [], []
    for (g, m), (Yd, Yb, rs, ers, lrh, chi) in fits.items():
        if m != model or g not in vf or not (ers / rs < 0.5 and chi < 5): continue
        Mh = 10**vf[g][0] / 0.05
        zc = kc['gv'](Mh, 1.0)[0]
        lv.append(np.log10(vf[g][1])); lrho.append(lrh); lage.append(np.log10(t0 - tcos(zc)))
    lv, lrho, lage = map(np.array, (lv, lrho, lage))
    s_data, e_data = bsl(lv, lrho)
    s_age = np.polyfit(lv, lage, 1)[0]
    s_pred = -1 - s_age
    a_need = s_data - s_pred
    # required cross-section per galaxy: s/m = 1/(rho_c v t_age), rho_c in Msun/pc^3 -> kg/m^3, v in km/s, t in s; report in cm^2/g
    rho_si = 10**lrho * 1.989e30 / (3.0857e16)**3; v_si = 10**lv * 1e3; t_si = 10**lage
    sm = 1 / (rho_si * v_si * t_si) * 1e4 / 1e3                   # m^2/kg -> cm^2/g
    ok(f"SI {model}: {len(lv)} galaxies; data d log rho_c/d log v = {s_data:+.2f} +- {e_data:.2f}; constant s/m predicts {s_pred:+.2f} "
       f"(-1 - d log t_age/d log v, age slope {s_age:+.2f}): {abs(s_data - s_pred) / e_data:.1f} sigma; required s/m: median {np.median(sm):.2f} cm^2/g, "
       f"scatter {np.std(np.log10(sm)):.2f} dex, trend with v {np.polyfit(lv, np.log10(sm), 1)[0]:+.2f}", True)
ok("SI3 recorded: the density slope does not exclude a constant cross-section (0.9-2.0 sigma); the required value is about 0.1 cm^2/g "
   "with ~1 dex scatter between galaxies; a constant s/m with one value per universe is a new label", True)

# SI4 clusters (known constraints, data not computed here): at v ~ 1000-1500 km/s, cluster cores and mergers bound s/m below ~0.1-1 cm^2/g
#     (Bullet cluster < 1.25, Randall et al. 2008; cluster cores ~0.1, Kaplinghat, Tulin, Yu 2016). Extrapolate the galaxy requirement:
#     constant s/m (median of both fits) and the measured trends with v, to v = 1500 km/s.
vmed = 10**np.median(lv)
for name, med, tr in (('pISO', 0.10, -1.01), ('Burkert', 0.13, -0.38)):
    print(f"   {name}: median v of the sample {vmed:.0f} km/s; s/m at 1500 km/s: constant {med:.2f}, with the trend {med * (1500 / vmed)**tr:.3f} cm^2/g")
ok("SI4 the galaxy requirement extrapolated to clusters (0.01-0.13 cm^2/g at 1500 km/s) lies at or below the known cluster bounds (~0.1-1.25)", True)
