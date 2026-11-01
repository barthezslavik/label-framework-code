# 102 (sixth review, 2026-10-23): do the premises P1 and P2 hold in the paper's own fits? For the SPARC galaxies whose data reach r_t = 7.16 r_c
# (same fits as de_btfr_rt_check): f_t = M_dark(r_t)/M_bar (P1 needs it common), the baryonic mass actually inside r_t, the relation
# log M(r_t) vs log r_t (P2 needs M/r^k constant, k ~ 2.1-2.3), and r_t/r_v with M_v = M_bar/eps and nu-peak collapse epochs (the Emden-box
# placement assumed r_t/r_v = 0.57 or 0.92). PF1-PF5. Run: ~/.venvs/jev/bin/python -I de_p1p2_fits_check.py SPARC_DIR
import os, sys, contextlib, io
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
rt = {'__name__': 'rt', '__file__': os.path.join(HERE, 'de_btfr_rt_check.py')}
sys.argv = [sys.argv[0], sys.argv[1], '0']
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(rt['__file__']).read().split('# bootstrap with refits')[0], rt)
ok, gals, base, cat, sel = rt['ok'], rt['gals'], rt['base'], rt['cat'], rt['sel']
G = 4.301e-6                                                     # kpc (km/s)^2 / Msun
rows = []
for g, R, V, eV, vb2 in gals:
    if g not in sel: continue
    vinf, rc = np.exp(base[g][1]); rtt = 7.16 * rc
    vd2 = vinf**2 * (1 - (rc / rtt) * np.arctan(rtt / rc))
    Md = vd2 * rtt / G
    Mbt = max(np.interp(rtt, R, vb2), 0) * rtt / G                 # baryons inside r_t (spherical estimate from v_bar)
    Mb = 10**cat[g][0]
    rows.append((g, rtt, Md, Mbt, Mb))
rtt = np.array([r[1] for r in rows]); Md = np.array([r[2] for r in rows]); Mbt = np.array([r[3] for r in rows]); Mb = np.array([r[4] for r in rows])
ft = Md / Mb; ftin = Md / Mbt
rng = np.random.default_rng(5)
def bslope(x, y):
    b = [np.polyfit(x[i], y[i], 1)[0] for i in (rng.integers(0, len(x), len(x)) for _ in range(1000))]
    return np.polyfit(x, y, 1)[0], np.std(b)
s1, e1 = bslope(np.log10(Mb), np.log10(ft)); s1b, e1b = bslope(np.log10(Mb), np.log10(ftin))
ok(f"PF1 f_t = M_dark(r_t)/M_bar: 10/50/90% = {np.percentile(ft, 10):.2f}/{np.median(ft):.2f}/{np.percentile(ft, 90):.2f}; d log f_t/d log M_bar = {s1:+.2f} +- {e1:.2f}", True)
ok(f"PF2 with the baryons inside r_t: f_t,in 10/50/90% = {np.percentile(ftin, 10):.2f}/{np.median(ftin):.2f}/{np.percentile(ftin, 90):.2f}; slope {s1b:+.2f} +- {e1b:.2f}; "
   f"M_bar(<r_t)/M_bar median {np.median(Mbt / Mb):.2f}", True)
Mt = Md + Mbt
s2, e2 = bslope(np.log10(rtt), np.log10(Mt))
ok(f"PF3 P2 at r_t: d log M(r_t)/d log r_t = {s2:.2f} +- {e2:.2f} (P2 with k = 2.1-2.3 needs this slope = k); scatter of log M/r_t^2.2 = {np.std(np.log10(Mt / rtt**2.2)):.2f} dex", True)
# PF4 r_t/r_v with M_v = M_bar/eps and nu-peak collapse
kc = {'__name__': 'kc', '__file__': os.path.join(HERE, 'ga_kappa_check.py')}
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(kc['__file__']).read(), kc)
def rv_kpc(Mv, nu):
    zc = kc['gv'](Mv, nu)[0]
    return (3 * Mv / (4 * np.pi * 18 * np.pi**2 * kc['rho_m0'] * (1 + zc)**3))**(1 / 3) * 1e3
out = {}
for eps in (0.02, 0.1):
    for nu in (1.0, 1.5):
        rv = np.array([rv_kpc(m / eps, nu) for m in Mb])
        out[(eps, nu)] = np.median(rtt / rv)
ok("PF4 median r_t/r_v: " + ", ".join(f"eps {e}, nu {n}: {v:.2f}" for (e, n), v in out.items()) + f"; median r_t = {np.median(rtt):.1f} kpc; "
   "the Emden-box placement assumed 0.57 or 0.92", True)
ok(f"PF5 conclusion recorded: P1 common f_t {'fails' if abs(s1) > 2 * e1 else 'holds'} in the fits at r_t; P2 slope {s2:.2f} vs k ~ 2.2; "
   f"r_t/r_v {min(out.values()):.2f}-{max(out.values()):.2f} vs 0.57-0.92", True)
