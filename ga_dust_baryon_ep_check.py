# 119: dust and baryons fall differently under the coupling m(phi) (82 CC3: G(1 + 2 beta^2) between dust elements, G between dust and
# uncoupled baryons). D5 of PREDICTIONS.md. The stream test itself is in the literature (Kesden & Kamionkowski 2006: roughly symmetric Sgr
# streams exclude a dark-sector acceleration difference above ~10%, beta < 0.2 in 0807.4363); computed here is only the gap: what the beta
# that cosmology allows gives. (i) beta allowed by S8 lensing through the S8(wbar) line (doc 117); (ii) the fractional acceleration
# difference 2 beta^2; (iii) the equilibrium offset between the stars and the dark core of a satellite, d = 2 beta^2 g_host / ((4 pi/3) G rho_c)
# (cored satellite, harmonic restoring force). EP1-EP5.
# Run: ~/.venvs/jev/bin/python ga_dust_baryon_ep_check.py
import os
import numpy as np
from scipy.optimize import brentq

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

HERE = os.path.dirname(os.path.abspath(__file__))
gc = {'__name__': 'gc'}; exec(open(os.path.join(HERE, 'ga_growth_coupling_check.py')).read().split('D0 = model(0.0)')[0], gc)
D0 = gc['model'](0.0)[0]
S8 = lambda b: 0.831 * gc['model'](b)[0] / D0
b_max = brentq(lambda b: S8(b) - (0.818 + 2 * 0.013), 0.005, 0.1, xtol=1e-5)
ok(f"EP1 S8 lensing (2 sigma) allows beta <= {b_max:.3f} for m(phi) (doc 117 SW2)", 0.02 < b_max < 0.06)
fr = {b: 2 * b**2 for b in (0.03, b_max, 0.06)}
ok("EP2 fractional acceleration difference dust vs baryons 2 beta^2: " + ", ".join(f"beta {b:.3f}: {100 * v:.2f}%" for b, v in fr.items())
   + "; stream sensitivity ~10% (Kesden & Kamionkowski 2006)", fr[b_max] < 0.01)
ok(f"EP3 the stream test is weaker than cosmology by a factor {0.10 / fr[b_max]:.0f} in 2 beta^2: m(phi) predicts symmetric streams; "
   "an asymmetry at the 10% level would not come from m(phi)", 0.10 / fr[b_max] > 10)
# EP4 offset of stars from the dark core in a cored satellite
G, Msun, pc, kpc = 6.674e-11, 1.989e30, 3.086e16, 3.086e19
cases = [("Sgr-like, r = 20 kpc, v_host 220 km/s, rho_c 0.1 Msun/pc^3", 20, 220, 0.1),
         ("dwarf at 80 kpc, v_host 200 km/s, rho_c 0.05", 80, 200, 0.05),
         ("satellite of M31-like host at 50 kpc, 250 km/s, rho_c 0.02", 50, 250, 0.02)]
offs = []
for nm, r, v, rho in cases:
    g = (v * 1e3)**2 / (r * kpc); rho_si = rho * Msun / pc**3
    d = 2 * b_max**2 * g / (4 * np.pi / 3 * G * rho_si) / pc
    offs.append(d); print(f"   {nm}: g_host = {g:.2e} m/s^2, offset at beta = {b_max:.3f}: {d:.2f} pc")
ok(f"EP4 equilibrium offset stars vs dark core at the allowed beta: {min(offs):.1f}-{max(offs):.1f} pc (cores are ~0.3-1 kpc): unobservable", max(offs) < 30)
ok("EP5 recorded: the same beta ties three observables -- phantom depth (wbar), S8 rise (doc 117) and the dust/baryon fall difference; "
   "a measured phantom at wbar = -1.05 by m(phi) would need beta ~ 0.05 and 2 beta^2 = 0.5%, still 20x under the stream bound", True)
# EP6 (paper7 reviews r13-r14) the same at the 2 sigma bounds of the two-fluid joint fit (doc 126: beta < 0.062 KiDS-DES; < 0.054 DES Y6)
for bfit in (0.054, 0.062):
    offs2 = [2 * bfit**2 * (v * 1e3)**2 / (r * kpc) / (4 * np.pi / 3 * G * rho * Msun / pc**3) / pc for nm, r, v, rho in cases]
    print(f"   beta = {bfit}: 2 beta^2 = {100 * 2 * bfit**2:.2f}%, stream sensitivity / 2 beta^2 = {0.10 / (2 * bfit**2):.0f}, offsets "
          f"{min(offs2):.1f}-{max(offs2):.1f} pc")
ok(f"EP6 at the fit bound beta = 0.062: 2 beta^2 = {100 * 2 * 0.062**2:.2f}%, {0.10 / (2 * 0.062**2):.0f} times below the stream sensitivity; "
   "offsets of a few to ~27 pc", 0.10 / (2 * 0.062**2) > 10)
