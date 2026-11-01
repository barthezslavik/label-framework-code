# 105: core scalings from the spectrum against the Bayesian SPARC fits of Li et al. 2020. Hierarchical formation (Prop spectrum of paper3):
# rho ~ M^-(n+3)/2, r ~ M^(n+5)/6, Sigma = rho r ~ M^-(n+2)/3 at the halo mass M, n = n_eff(M) of the CDM spectrum. Assumption: the core
# scales with the virial quantities (fixed density contrast, as the isothermal box of doc 96 gives) and M_h = M_b/eps with eps common
# (d log M_h = d log M_b). Data: pISO-Flat and Burkert-Flat fits (well constrained), slopes against log M_b. No fit. CS1-CS3.
# Run: ~/.venvs/jev/bin/python -I de_core_scaling_check.py LI_DIR SPARC_DIR
import os, sys, contextlib, io
import numpy as np
from scipy.integrate import quad
HERE = os.path.dirname(os.path.abspath(__file__))
lc = {'__name__': 'lc', '__file__': os.path.join(HERE, 'de_li2020_check.py')}
src = open(lc['__file__']).read().split("for model in ('pISO-Flat', 'Burkert-Flat'):")[0]
with contextlib.redirect_stdout(io.StringIO()):
    exec(src, lc)
ok, fits, L, MHI, mm, bsl = lc['ok'], lc['fits'], lc['L'], lc['MHI'], lc['mm'], lc['bsl']
h, Om = 0.674, 0.315; Gam = Om * h
T = lambda k: np.log(1 + 2.34 * k / Gam) / (2.34 * k / Gam) * (1 + 3.89 * k / Gam + (16.1 * k / Gam)**2 + (5.46 * k / Gam)**3 + (6.71 * k / Gam)**4)**-0.25
P = lambda k: k**0.965 * T(k)**2
rho_m = 2.775e11 * Om
Wf = lambda q: 3 * (np.sin(q) - q * np.cos(q)) / q**3
sig2 = lambda Mm: quad(lambda lk: np.exp(lk)**3 * P(np.exp(lk)) * Wf(np.exp(lk) * (3 * Mm / (4 * np.pi * rho_m))**(1 / 3))**2, np.log(1e-4), np.log(1e4), limit=500)[0]
neff = lambda Mm, e=1.05: -3 - 6 * (np.log(sig2(Mm * e)) - np.log(sig2(Mm / e))) / 2 / (2 * np.log(e))
for model in ('pISO-Flat', 'Burkert-Flat'):
    lMb, lrho, lr = [], [], []
    for (g, m), (Yd, Yb, rs, ers, lrh, chi) in fits.items():
        if m != model or g not in L or not (ers / rs < 0.5 and chi < 5): continue
        lMb.append(np.log10(Yd * L[g] + 1.33 * MHI[g])); lrho.append(lrh); lr.append(np.log10(rs))
    lMb, lrho, lr = map(np.array, (lMb, lrho, lr)); lSig = lrho + lr
    d = {k: bsl(lMb, y) for k, y in (('rho', lrho), ('r', lr), ('Sigma', lSig))}
    Mb_med = 10**np.median(lMb)
    preds = {}
    for eps in (0.02, 0.1):
        n = neff(Mb_med / eps * h)
        preds[eps] = {'rho': -(n + 3) / 2, 'r': (n + 5) / 6, 'Sigma': -(n + 2) / 3, 'n': n}
    for q in ('rho', 'r', 'Sigma'):
        lo, hi = sorted([preds[0.02][q], preds[0.1][q]])
        dev = min(abs(d[q][0] - lo), abs(d[q][0] - hi)) / d[q][1] if not (lo <= d[q][0] <= hi) else 0.0
        print(f"   {model:12s} {q:5s}: data slope {d[q][0]:+.2f} +- {d[q][1]:.2f}; spectrum {lo:+.2f} .. {hi:+.2f} (n_eff {preds[0.1]['n']:.2f} .. {preds[0.02]['n']:.2f}); {dev:.1f} sigma outside")
    ok(f"CS {model}: {len(lMb)} galaxies; rho {d['rho'][0]:+.2f}, r {d['r'][0]:+.2f}, Sigma {d['Sigma'][0]:+.2f} vs spectrum "
       f"{preds[0.1]['rho']:+.2f}/{preds[0.1]['r']:+.2f}/{preds[0.1]['Sigma']:+.2f} (eps 0.1) .. {preds[0.02]['rho']:+.2f}/{preds[0.02]['r']:+.2f}/{preds[0.02]['Sigma']:+.2f} (eps 0.02)", True)
