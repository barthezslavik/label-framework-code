# 113: is the 0.8 dex galaxy-to-galaxy scatter of the required self-interaction cross-section (doc 107) driven by the baryons? With one
# universal s/m, a self-interacting core responds to the baryon potential (denser baryons -> denser core; known behaviour of such models),
# so the naive s/m = 1/(rho_c v t_age), computed without baryons, should anticorrelate with the baryonic surface density at fixed v.
# Data: Li 2020 fits (rho_c), SPARC SB_disk (central disk surface brightness, 3.6 um) and V_f. DV1-DV3.
# Run: ~/.venvs/jev/bin/python -I de_si_diversity_check.py LI_DIR SPARC_DIR
import os, sys, contextlib, io
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
si = {'__name__': 'si', '__file__': os.path.join(HERE, 'de_self_interaction_check.py')}
src = open(si['__file__']).read().split("\nfor model in ('pISO-Flat', 'Burkert-Flat'):")[0]
with contextlib.redirect_stdout(io.StringIO()):
    exec(src, si)
ok, fits, vf, kc, tcos, t0 = si['ok'], si['fits'], si['vf'], si['kc'], si['tcos'], si['t0']
SB = {}
for l in open(os.path.join(sys.argv[2], 'SPARC_Lelli2016c.mrt')):
    p = l.split()
    if len(p) < 18: continue
    try:
        int(p[1]); SB[p[0]] = float(p[12])                         # SBdisk, Lsun/pc^2
    except ValueError:
        continue
rng = np.random.default_rng(19)
for model in ('pISO-Flat', 'Burkert-Flat'):
    lv, lsm, lsb = [], [], []
    for (g, m), (Yd, Yb, rs, ers, lrh, chi) in fits.items():
        if m != model or g not in vf or g not in SB or SB[g] <= 0 or not (ers / rs < 0.5 and chi < 5): continue
        zc = kc['gv'](10**vf[g][0] / 0.05, 1.0)[0]
        t = t0 - tcos(zc); v = vf[g][1] * 1e3; rho = 10**lrh * 1.989e30 / (3.0857e16)**3
        lsm.append(np.log10(1 / (rho * v * t) * 10)); lv.append(np.log10(vf[g][1])); lsb.append(np.log10(SB[g]))
    lv, lsm, lsb = map(np.array, (lv, lsm, lsb))
    # partial correlation of s/m with SB at fixed v: regress both on v, correlate residuals
    rs_ = lsm - np.polyval(np.polyfit(lv, lsm, 1), lv); rb_ = lsb - np.polyval(np.polyfit(lv, lsb, 1), lv)
    r = np.corrcoef(rs_, rb_)[0, 1]
    rb = [np.corrcoef(rs_[i], rb_[i])[0, 1] for i in (rng.integers(0, len(rs_), len(rs_)) for _ in range(2000))]
    sl = np.polyfit(rb_, rs_, 1)[0]
    expl = 1 - np.var(rs_ - sl * rb_) / np.var(rs_)
    ok(f"DV {model}: {len(lv)} galaxies; partial correlation of log s/m with log SB_disk at fixed v: r = {r:+.2f} +- {np.std(rb):.2f}, "
       f"slope {sl:+.2f}; share of the s/m scatter explained {expl:.2f} (scatter {np.std(rs_):.2f} -> {np.std(rs_ - sl * rb_):.2f} dex)", True)
ok("DV3 recorded: a negative partial correlation means the naive cross-section is lower where baryons are dense, as a universal s/m with "
   "baryon-dependent cores predicts; no correlation means the scatter is intrinsic", True)

# DV4 how much of the 0.8 dex is measurement noise: the catalogue error on log rho_s (e_log(rhos), bytes 119-124 of table1.dat)
erh = {}
for l in open(os.path.join(sys.argv[1], 'table1.dat')):
    if len(l) < 136: continue
    try:
        erh[(l[0:11].strip(), l[12:24].strip())] = float(l[118:124])
    except ValueError:
        continue
for model in ('pISO-Flat', 'Burkert-Flat'):
    e = np.array([erh[(g, m)] for (g, m), (Yd, Yb, rs, ers, lrh, chi) in fits.items()
                  if m == model and g in vf and g in SB and SB[g] > 0 and ers / rs < 0.5 and chi < 5])
    tot = 0.80
    intr = np.sqrt(max(tot**2 - np.median(e)**2, 0))
    ok(f"DV4 {model}: median error of log rho_s = {np.median(e):.2f} dex (mean {np.mean(e):.2f}); intrinsic scatter of the required s/m "
       f"~ sqrt(0.80^2 - err^2) = {intr:.2f} dex", True)
