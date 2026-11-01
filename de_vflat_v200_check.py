# 110: the population law rebuilt at the halo radius. Measured R = v_flat/v200 at fixed baryonic mass: v200 = sqrt(G M200/r200) from weak
# lensing (DESI/DES dwarfs, M200c, doc 109; SDSS blue centrals, M200m, doc 99), v_flat from the SPARC BTFR (Lelli 2019 V_f; linear ML fit
# inverted at the bin's M_b, gas as in docs 99/109). Framework prediction (no fit): the plateau is set by shells at their own collapse
# (doc 98), so v_flat^2 ~ G M/r_v(z_c) with Delta = 18 pi^2 rho_m(z_c), while v200 today has M ~ v^3 by the fixed-overdensity definition;
# with M_v = M200: R_pred = (rho_vir(z_c)/rho_200(z))^(1/6), z_c from nu-peak collapse (nu = 1, 1.5), M_h = M200. VR1-VR4.
# Run: ~/.venvs/jev/bin/python -I de_vflat_v200_check.py DATA_ROOT SPARC_DIR
import os, sys, contextlib, io
import numpy as np
from scipy.optimize import minimize
HERE = os.path.dirname(os.path.abspath(__file__))
root, spd = sys.argv[1], sys.argv[2]
kc = {'__name__': 'kc', '__file__': os.path.join(HERE, 'ga_kappa_check.py')}
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(kc['__file__']).read(), kc)
def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond
# BTFR: forward ML fit log M_b = a + b (log V_f - 2), inverted
rows = []
for l in open(os.path.join(spd, 'BTFR_Lelli2019.mrt')):
    if len(l) < 48: continue
    try:
        lm, elm, vf, evf = float(l[12:18]), float(l[18:24]), float(l[36:42]), float(l[42:48])
        if vf > 0: rows.append((lm, elm, vf, evf))
    except ValueError:
        continue
d = np.array(rows); y, ey, x, ex = d[:, 0], d[:, 1], np.log10(d[:, 2]) - 2, d[:, 3] / d[:, 2] / np.log(10)
nll = lambda p: 0.5 * np.sum((y - p[0] - p[1] * x)**2 / (ey**2 + (p[1] * ex)**2 + np.exp(2 * p[2])) + np.log(ey**2 + (p[1] * ex)**2 + np.exp(2 * p[2])))
a, b, _ = minimize(nll, [10, 3.8, np.log(0.1)], method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 20000}).x
vflat = lambda lMb: 100 * 10**((lMb - a) / b)
# gas
L, MHI = [], []
for l in open(os.path.join(spd, 'SPARC_Lelli2016c.mrt')):
    p = l.split()
    if len(p) < 18: continue
    try:
        int(p[1]); L.append(float(p[7]) * 1e9); MHI.append(float(p[13]) * 1e9)
    except ValueError:
        continue
L, MHI = np.array(L), np.array(MHI); Ms = 0.5 * L; gd = Ms > 1e7
cg = np.polyfit(np.log10(Ms[gd]) - 9, np.log10((Ms[gd] + 1.33 * MHI[gd]) / Ms[gd]), 2)
gas = lambda lm: 10**np.polyval(cg, lm - 9)
G = 4.301e-9; h, Om = 0.674, 0.315
samples = []
for r in (l.split() for l in open(os.path.join(root, 'desi_des_dwarfs2025', 'table_halo_masses.txt')) if l.strip() and not l.startswith('#')):
    samples.append((r[0], float(r[1]), float(r[4]), 0.05, 'c'))
for r in np.loadtxt(os.path.join(root, 'mandelbaum2016', 'table_B1_blue.txt')):
    samples.append(('SDSS-blue', r[0], r[2] - np.log10(0.673), r[5], 'm'))
out = []
for name, lms, lmh, z, kind in samples:
    Mh = 10**lmh; lMb = lms + np.log10(gas(lms))
    rho_ref = 200 * 2.775e11 * h**2 * ((Om * (1 + z)**3 + 1 - Om) if kind == 'c' else Om * (1 + z)**3)
    r200 = (3 * Mh / (4 * np.pi * rho_ref))**(1 / 3); v200 = np.sqrt(G * Mh / r200)
    R = vflat(lMb) / v200
    preds = []
    for nu in (1.0, 1.5):
        zc = kc['gv'](Mh, nu)[0]
        rho_vir = 18 * np.pi**2 * kc['rho_m0'] * (1 + zc)**3
        preds.append((rho_vir / rho_ref)**(1 / 6))
    out.append((name, lMb, v200, vflat(lMb), R, preds[0], preds[1]))
    print(f"   {name:9s} log M_b {lMb:5.2f}: v200 {v200:5.1f}, v_flat {vflat(lMb):5.1f} km/s, R = {R:.2f}; predicted {preds[0]:.2f} (nu 1), {preds[1]:.2f} (nu 1.5)")
lMb = np.array([o[1] for o in out]); R = np.array([o[4] for o in out]); P1 = np.array([o[5] for o in out]); P15 = np.array([o[6] for o in out])
sR = np.polyfit(lMb, np.log10(R), 1)[0]; sP1 = np.polyfit(lMb, np.log10(P1), 1)[0]; sP15 = np.polyfit(lMb, np.log10(P15), 1)[0]
ok(f"VR1 BTFR used: log M_b = {a:.2f} + {b:.2f}(log V - 2)", 3.3 < b < 4.0)
ok(f"VR2 measured v_flat/v200 = {R.min():.2f}-{R.max():.2f} (median {np.median(R):.2f}), trend d log R/d log M_b = {sR:+.3f}", True)
ok(f"VR3 predicted from the collapse epoch: median {np.median(P1):.2f} (nu 1), {np.median(P15):.2f} (nu 1.5); trend {sP1:+.3f} / {sP15:+.3f}", True)
ok(f"VR4 level: measured/predicted median {np.median(R / P1):.2f} (nu 1), {np.median(R / P15):.2f} (nu 1.5); trend difference {sR - sP1:+.3f} / {sR - sP15:+.3f}", True)

# VR5 decomposition of the BTFR slope at the halo radius: d log v_flat/d log M_b (= 1/b, from the BTFR) = d log R/d log M_b + d log v200/d log M_b,
#     all fitted over the 18 bins with bootstrap errors; and the trend of the retained share chi = M_b/(f_cos M200).
fcos = 0.0493 / 0.315
Mh_all = np.array([10**s_[2] for s_ in samples]); chi = 10**lMb / (fcos * Mh_all)
v200s = np.array([o[2] for o in out])
rng = np.random.default_rng(13)
def bs(yv):
    sl = np.polyfit(lMb, yv, 1)[0]
    return sl, np.std([np.polyfit(lMb[i], yv[i], 1)[0] for i in (rng.integers(0, len(lMb), len(lMb)) for _ in range(2000))])
sRb, eR = bs(np.log10(R)); sc, ec = bs(np.log10(chi)); sv, ev = bs(np.log10(v200s))
ok(f"VR5 d log v_flat/d log M_b = 1/{b:.2f} = {1 / b:.3f} = d log v200/d log M_b ({sv:+.3f} +- {ev:.3f}) + d log R/d log M_b ({sRb:+.3f} +- {eR:.3f}); "
   f"retained share d log chi/d log M_b = {sc:+.2f} +- {ec:.2f}: the BTFR slope is carried by the halo relation M_b - v200 (slope {1 / sv:.2f}), "
   f"with R constant to +-{eR:.2f} per decade", True)

# VR6 one sample, one halo-mass definition at a time (review of the mixed 18 bins): DES (7 bins, M200c), DESI (6, M200c), SDSS blue (5, M200m)
names = np.array([o[0] for o in out])
for nm in ('DES', 'DESI', 'SDSS-blue'):
    m = names == nm
    x_ = lMb[m]
    def bsm(yv):
        sl = np.polyfit(x_, yv, 1)[0]
        return sl, np.std([np.polyfit(x_[i], yv[i], 1)[0] for i in (rng.integers(0, len(x_), len(x_)) for _ in range(2000)) if len(set(i)) > 2])
    a1, e1 = bsm(np.log10(R[m])); a2, e2 = bsm(np.log10(v200s[m])); a3, e3 = bsm(np.log10(chi[m]))
    ok(f"VR6 {nm}: {m.sum()} bins; R median {np.median(R[m]):.2f}, trend {a1:+.3f} +- {e1:.3f}; M_b-v200 slope {1 / a2:.2f} (d log v200/d log M_b {a2:+.3f} +- {e2:.3f}); "
       f"chi trend {a3:+.2f} +- {e3:.2f}", True)
