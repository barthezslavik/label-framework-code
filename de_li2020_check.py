# 104b: the trends of doc 103 on the Bayesian SPARC halo fits of Li et al. 2020 (ApJS 247, 31; CDS J/ApJS/247/31 table1.dat, MCMC with free
# M/L, distance, inclination). pISO-Flat and Burkert-Flat (cored). Sigma = rho_s r_s; the volume density rho_s; the dark/baryon mass ratio
# inside r_s (baryons from the SPARC mass models with the fitted Upsilon). Well-constrained fits only (e_rs/rs < 0.5, chi2_nu < 5). LI1-LI4.
# Run: ~/.venvs/jev/bin/python -I de_li2020_check.py LI_DIR SPARC_DIR
import os, sys
import numpy as np

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

li, sp = sys.argv[1], sys.argv[2]
fits = {}
for l in open(os.path.join(li, 'table1.dat')):
    if len(l) < 136: continue
    g, model = l[0:11].strip(), l[12:24].strip()
    try:
        Yd, Yb, rs, ers, lrho = float(l[25:29]), float(l[35:39]), float(l[96:102]), float(l[103:111]), float(l[112:117])
        chi = float(l[147:152]) if l[147:152].strip() else np.inf
    except ValueError:
        continue
    fits[(g, model)] = (Yd, Yb, rs, ers, lrho, chi)
L, MHI = {}, {}
for l in open(os.path.join(sp, 'SPARC_Lelli2016c.mrt')):
    p = l.split()
    if len(p) < 18: continue
    try:
        int(p[1]); L[p[0]] = float(p[7]) * 1e9; MHI[p[0]] = float(p[13]) * 1e9
    except ValueError:
        continue
mm = {}
for l in open(os.path.join(sp, 'MassModels_Lelli2016c.mrt')):
    p = l.split()
    if len(p) != 10: continue
    try:
        mm.setdefault(p[0], []).append([float(x) for x in p[1:]])
    except ValueError:
        continue
G = 4.301e-6
def mdark(model, rho, r0, r):                                     # rho in Msun/kpc^3, r in kpc
    x = r / r0
    if model.startswith('pISO'): return 4 * np.pi * rho * r0**3 * (x - np.arctan(x))
    return np.pi * rho * r0**3 * (np.log((1 + x)**2 * (1 + x**2)) - 2 * np.arctan(x))
rng = np.random.default_rng(11)
def bsl(x, y):
    return np.polyfit(x, y, 1)[0], np.std([np.polyfit(x[i], y[i], 1)[0] for i in (rng.integers(0, len(x), len(x)) for _ in range(1000))])
for model in ('pISO-Flat', 'Burkert-Flat'):
    lMb, lSig, lrho, lrat = [], [], [], []
    for (g, m), (Yd, Yb, rs, ers, lr, chi) in fits.items():
        if m != model or g not in L or g not in mm or not (ers / rs < 0.5 and chi < 5): continue
        a = np.array(mm[g]); R = a[:, 1]
        if not (R.min() < rs < R.max()): continue
        vb2 = np.sign(a[:, 3]) * a[:, 3]**2 + Yd * a[:, 4]**2 + Yb * a[:, 5]**2
        Mbin = max(np.interp(rs, R, vb2), 1e-3) * rs / G
        rho = 10**lr * 1e9                                       # Msun/kpc^3
        lMb.append(np.log10(Yd * L[g] + 1.33 * MHI[g])); lSig.append(lr + np.log10(rs * 1e3)); lrho.append(lr)
        lrat.append(np.log10(mdark(model, rho, rs, rs) / Mbin))
    lMb, lSig, lrho, lrat = map(np.array, (lMb, lSig, lrho, lrat))
    s1, e1 = bsl(lMb, lSig); s2, e2 = bsl(lMb, lrho); s3, e3 = bsl(lMb, lrat)
    ok(f"LI {model}: {len(lMb)} galaxies; Sigma = rho_s r_s median {10**np.median(lSig):.0f} Msun/pc^2, scatter {np.std(lSig):.2f} dex, d log Sigma/d log M_b = {s1:+.2f} +- {e1:.2f}; "
       f"d log rho_s/d log M_b = {s2:+.2f} +- {e2:.2f} (scatter {np.std(lrho):.2f}); inner dark/baryon ratio at r_s: median {10**np.median(lrat):.2f}, slope {s3:+.2f} +- {e3:.2f}", len(lMb) > 30)
