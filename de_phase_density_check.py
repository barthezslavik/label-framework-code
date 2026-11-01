# 106: does the fine-grained phase density eta (a label of the dust medium; Liouville bounds the coarse-grained density by it) set the
# cores? If cores saturate the bound, Q0 = rho0/sigma^3 is common, so rho0 ~ sigma^3 ~ v^3 and (isothermal core rho0 = 9 sigma^2/(4 pi G r_c^2))
# r_c ~ sigma^-1/2. Test on the Li et al. 2020 Bayesian fits (pISO, Burkert; rho_s, r_s) with sigma = V_f/sqrt 2 (Lelli 2019). PD1-PD3.
# Run: ~/.venvs/jev/bin/python -I de_phase_density_check.py LI_DIR SPARC_DIR
import os, sys, contextlib, io
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
lc = {'__name__': 'lc', '__file__': os.path.join(HERE, 'de_li2020_check.py')}
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(lc['__file__']).read().split("for model in ('pISO-Flat', 'Burkert-Flat'):")[0], lc)
ok, fits, bsl = lc['ok'], lc['fits'], lc['bsl']
vf = {}
for l in open(os.path.join(sys.argv[2], 'BTFR_Lelli2019.mrt')):
    if len(l) < 48: continue
    try:
        v = float(l[36:42])
        if v > 0: vf[l[:12].strip()] = (float(l[12:18]), v)
    except ValueError:
        continue
for model in ('pISO-Flat', 'Burkert-Flat'):
    lv, lMb, lrho, lr = [], [], [], []
    for (g, m), (Yd, Yb, rs, ers, lrh, chi) in fits.items():
        if m != model or g not in vf or not (ers / rs < 0.5 and chi < 5): continue
        lMb.append(vf[g][0]); lv.append(np.log10(vf[g][1] / np.sqrt(2))); lrho.append(lrh); lr.append(np.log10(rs))
    lv, lMb, lrho, lr = map(np.array, (lv, lMb, lrho, lr))
    lQ = lrho - 3 * lv
    sQ, eQ = bsl(lv, lQ); srho, erho = bsl(lv, lrho); sr, er = bsl(lv, lr)
    ok(f"PD {model}: {len(lv)} galaxies over sigma = {10**lv.min():.0f}-{10**lv.max():.0f} km/s: d log Q0/d log sigma = {sQ:+.2f} +- {eQ:.2f} "
       f"(saturation needs 0; scatter {np.std(lQ):.2f} dex); d log rho0/d log sigma = {srho:+.2f} +- {erho:.2f} (needs +3); "
       f"d log r_c/d log sigma = {sr:+.2f} +- {er:.2f} (needs -0.5)", True)
ok("PD3 recorded: saturation of a common phase-density bound is the Tremaine-Gunn-type core law (known); its exponents are compared above", True)
