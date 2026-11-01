# 110b: v_flat/v200 and the retained baryon share along ONE homogeneous weak-lensing relation (HSC centrals, arXiv:2301.04664; parametric CSMF
# fit, halo masses taken as M200m, h = 0.7 for unit conversion). For log M* (h^-2) = 9.06-11.28 (the lens bins): halo mass from inverting
# M*_c(M); z from the bin's median; v200 = sqrt(G M/r200m); v_flat from the SPARC BTFR at the same M_b (gas from SPARC). Errors by sampling
# the four SHMR parameters independently (asymmetric Gaussians; correlations ignored). VH1-VH3.
# Run: ~/.venvs/jev/bin/python -I de_hsc_shmr_check.py DATA_ROOT SPARC_DIR
import os, sys, contextlib, io
import numpy as np
from scipy.optimize import brentq
HERE = os.path.dirname(os.path.abspath(__file__))
vv = {'__name__': 'vv', '__file__': os.path.join(HERE, 'de_vflat_v200_check.py')}
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(vv['__file__']).read().split('samples = []')[0], vv)
ok, vflat, gas = vv['ok'], vv['vflat'], vv['gas']
P = {}
for l in open(os.path.join(sys.argv[1], 'hsc_2301.04664', 'shmr_params.txt')):
    if l.startswith('#') or not l.strip(): continue
    n, v, ep, em = l.split(); P[n] = (float(v), float(ep), float(em))
hh = 0.7; G = 4.301e-9; Om = 0.315
lms_h = np.array([9.06, 9.62, 10.01, 10.39, 10.76, 11.08, 11.28]); zb = np.array([0.05, 0.11, 0.14, 0.18, 0.25, 0.26, 0.25])
fcos = 0.0493 / 0.315
def run(p):
    lM0, lM1, g1, g2 = p
    out = []
    for lmh_s, z in zip(lms_h, zb):
        f = lambda lM: lM0 + g1 * (lM - lM1) - (g1 - g2) * np.log10(1 + 10**(lM - lM1)) - lmh_s
        lM = brentq(f, 9.0, 15.5)                                   # log M200m in h^-1 Msun
        Mh = 10**lM / hh; lms = lmh_s - 2 * np.log10(hh)            # Msun
        lMb = lms + np.log10(gas(lms))
        rho = 200 * 2.775e11 * hh**2 * Om * (1 + z)**3
        r = (3 * Mh / (4 * np.pi * rho))**(1 / 3); v200 = np.sqrt(G * Mh / r)
        out.append((lMb, vflat(lMb) / v200, 10**lMb / (fcos * Mh), v200))
    return np.array(out)
base = run([P['logM0'][0], P['logM1'][0], P['gamma1'][0], P['gamma2'][0]])
rng = np.random.default_rng(17)
def draw(k):
    v, ep, em = P[k]; x = rng.standard_normal(); return v + (ep if x > 0 else em) * x
sl = {'R': [], 'chi': [], 'v': []}
for _ in range(2000):
    try:
        o = run([draw('logM0'), draw('logM1'), max(draw('gamma1'), 1.01), max(draw('gamma2'), 0.02)])
    except ValueError:
        continue
    sl['R'].append(np.polyfit(o[:, 0], np.log10(o[:, 1]), 1)[0]); sl['chi'].append(np.polyfit(o[:, 0], np.log10(o[:, 2]), 1)[0])
    sl['v'].append(np.polyfit(o[:, 0], np.log10(o[:, 3]), 1)[0])
for row in base:
    print(f"   log M_b {row[0]:5.2f}: R = v_flat/v200m = {row[1]:.2f}, chi_b = {row[2]:.2f}, v200m = {row[3]:.0f} km/s")
q = lambda a: (np.median(a), np.std(a))
ok(f"VH1 HSC homogeneous relation, log M_b {base[0, 0]:.1f}-{base[-1, 0]:.1f}: R = {base[:, 1].min():.2f}-{base[:, 1].max():.2f} (median {np.median(base[:, 1]):.2f})", True)
ok(f"VH2 trends (parameter sampling): d log R/d log M_b = {q(sl['R'])[0]:+.3f} +- {q(sl['R'])[1]:.3f}; d log chi/d log M_b = {q(sl['chi'])[0]:+.2f} +- {q(sl['chi'])[1]:.2f}; "
   f"M_b-v200 slope {1 / q(sl['v'])[0]:.2f}", True)
ok("VH3 recorded: a single parametric relation; its trends follow the double power law (turnover near M1), and correlations of the "
   "parameters are ignored", True)

# VH4 the disc regime only (log M_b <= 10.8; above it group-scale, mostly red centrals dominate the all-colour relation)
slr = []
for _ in range(2000):
    try:
        o = run([draw('logM0'), draw('logM1'), max(draw('gamma1'), 1.01), max(draw('gamma2'), 0.02)])
    except ValueError:
        continue
    m = o[:, 0] <= 10.85
    slr.append(np.polyfit(o[m, 0], np.log10(o[m, 1]), 1)[0])
m = base[:, 0] <= 10.85
ok(f"VH4 disc regime (log M_b {base[m, 0].min():.1f}-{base[m, 0].max():.1f}): R = {base[m, 1].min():.2f}-{base[m, 1].max():.2f}, trend {np.median(slr):+.3f} +- {np.std(slr):.3f}; "
   f"level 1.0-1.07 here against 1.26-1.37 from DES/DESI/SDSS (doc 110): the absolute level carries the halo-mass normalisation between surveys", True)

# VH5 (review 2026-10-23) uncertainty of the share trend in the disc regime from the same parameter sampling
slc = []
for _ in range(2000):
    try:
        o = run([draw('logM0'), draw('logM1'), max(draw('gamma1'), 1.01), max(draw('gamma2'), 0.02)])
    except ValueError:
        continue
    mm = o[:, 0] <= 10.85
    slc.append(np.polyfit(o[mm, 0], np.log10(o[mm, 2]), 1)[0])
ok(f"VH5 disc-regime share trend d log chi/d log M_b = {np.median(slc):+.2f} +- {np.std(slc):.2f} (16-84%: {np.percentile(slc, 16):+.2f} to {np.percentile(slc, 84):+.2f})", True)
