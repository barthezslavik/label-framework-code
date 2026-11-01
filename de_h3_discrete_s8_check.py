# 126b: the second native phantom mechanism (discrete fluctuation of Lambda, doc 97: 300 realizations fitted to DESI DR2 + compressed Planck
# + Pantheon+) with the same S8 term as the coupled fit (de_h3_s8_fit_check: same sigma8 pipeline, A_s and n_s at Planck, calibration 0.811
# at Planck LCDM, omega_b = 0.0224 for the transfer function since doc 97 did not store it; doc 97 grew D with radiation in H, the
# sigma8 pipeline without it -- the LCDM ratio of the two is applied). Inputs: results/phantom_compare_300.npy and
# results/h3_s8_fit3.npy (the two-fluid coupled-model grid). Delta chi2 against LCDM with the same lensing term. HD1-HD3.
# Run: ~/.venvs/jev/bin/python -I de_h3_discrete_s8_check.py DATA_ROOT
import os, sys, contextlib, io
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
g = {'__name__': 'h3', '__file__': os.path.join(HERE, 'de_h3_s8_fit_check.py')}
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(g['__file__']).read(), g)
ok, sigma8, CAL, LENS = g['ok'], g['sigma8'], g['CAL'], g['LENS']
P = np.load(os.path.join(HERE, 'results', 'phantom_compare_300.npy'))
H = np.load(os.path.join(HERE, 'results', 'h3_s8_fit3.npy'))
good = np.isfinite(P[:, 1])
chi, D, Om, h, wbar = P[good, 1], P[good, 5], P[good, 6], P[good, 7], P[good, 8]
from scipy.integrate import solve_ivp
def rad_factor(o, hh):
    """doc 97 grows D with radiation in H from a = 1e-3; the sigma8 pipeline grows it without: ratio D(no rad)/D(rad) in LCDM"""
    orr = 4.18e-5 / hh**2
    def D1(withr):
        def f(N, y):
            a = np.exp(N); E2 = (orr if withr else 0) / a**4 + o / a**3 + 1 - o - (orr if withr else 0)
            dl = 0.5 * (-4 * (orr if withr else 0) / a**4 - 3 * o / a**3) / E2
            return [y[1], -(2 + dl) * y[1] + 1.5 * (o / a**3) / E2 * y[0]]
        return solve_ivp(f, [np.log(1e-3), 0], [1e-3, 1e-3], rtol=1e-9).y[0, -1]
    return D1(False) / D1(True)
S8 = np.array([CAL * sigma8(o, hh, 0.0224, d * rad_factor(o, hh)) * np.sqrt(o / 0.3) for o, hh, d in zip(Om, h, D)])
lcdm = {0: H[(H[:, 0] == 0) & (H[:, 1] == 0)][0, 2], 1: H[(H[:, 0] == 0) & (H[:, 1] == 1)][0, 2], 2: H[(H[:, 0] == 0) & (H[:, 1] == 2)][0, 2]}
coup = {k: H[H[:, 1] == k][:, 2].min() - lcdm[k] for k in (0, 1, 2)}
res = {}
for k, nm in ((0, 'none'), (1, 'KiDS-DES'), (2, 'DES-Y6')):
    term = 0 if k == 0 else ((S8 - LENS[nm][0]) / LENS[nm][1])**2
    dchi = chi + term - lcdm[k]
    res[nm] = (dchi.min(), np.mean(dchi < -5), np.mean(np.exp(-np.clip(dchi, -60, None) / 2)), np.median(S8[dchi < -5]) if np.any(dchi < -5) else np.nan)
print(f"   {good.sum()} realizations; S8 median {np.median(S8):.3f} (range {S8.min():.3f}-{S8.max():.3f}); LCDM chi2 with lensing: "
      + ", ".join(f"{k} {v:.2f}" for k, v in lcdm.items()))
ok("HD1 discrete fluctuation with the S8 term: " + "; ".join(f"{k}: best Delta chi2 {v[0]:+.2f}, fraction < -5 {v[1]:.3f}, mean exp(-Delta chi2/2) over realizations {v[2]:.2f}"
   for k, v in res.items()), True)
ok("HD2 coupled m(phi), profiled best (one parameter): " + "; ".join(f"{nm}: Delta chi2 {coup[k]:+.2f}" for k, nm in ((0, 'none'), (1, 'KiDS-DES'), (2, 'DES-Y6'))), True)
ok("HD3 m(phi) is not preferred at 2 sigma for one parameter (Delta chi2 > -4) with any lensing set; for the discrete fluctuation the mean of "
   "exp(-Delta chi2/2) over realizations with profiled parameters is a procedure, not an evidence integral -- no significance is assigned", all(v > -4 for v in coup.values()))

# HD4 (paper5 v2, Codex r3): the signature with the S8 of this pipeline. Column wbar of phantom_compare_300 has the flipped sign
# (ga_wbar_definition_check WD2): wbar_ln_a = -2 - stored. Correlation and slope of S8 against the depth across realizations, for
# all of them and for the good fits (Delta chi2 < -5 without lensing), against the profiled line of the coupling (slope -0.572 at wbar = -1).
wl = -2 - wbar
cc_all = np.corrcoef(wl, S8)[0, 1]; sl_all = np.polyfit(wl, S8, 1)[0]
topf = (chi - lcdm[0]) < -5
cc_top = np.corrcoef(wl[topf], S8[topf])[0, 1]; sl_top = np.polyfit(wl[topf], S8[topf], 1)[0]
ok(f"HD4 fluctuation, S8 of this pipeline vs wbar over ln a: all {good.sum()}: correlation {cc_all:+.3f}, slope {sl_all:+.3f}; "
   f"good fits ({topf.sum()}): {cc_top:+.3f}, {sl_top:+.3f}; coupling line slope -0.572: the fluctuation has no line of that size",
   abs(sl_all) < 0.572 / 2)
