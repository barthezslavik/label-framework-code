# Figure for paper3: the baryonic Tully--Fisher relation of SPARC (Lelli et al. 2019, flat velocities V_f) with the population slope
# 12/(1-n_eff) = 3.68-3.79 predicted at the centre of the sample (doc 93 BR1), drawn through the centre of the linear fit of the data
# (forward maximum likelihood, as in de_vflat_v200_check). Input data are untrusted: run with python -I.
# Run: ~/.venvs/jev/bin/python -I fig_paper3_btfr.py SPARC_DIR OUT.pdf
import os, sys
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
rows = []
for l in open(os.path.join(sys.argv[1], 'BTFR_Lelli2019.mrt'), encoding='latin-1'):
    if len(l) < 48: continue
    try:
        lm, elm, vf, evf = float(l[12:18]), float(l[18:24]), float(l[36:42]), float(l[42:48])
        if vf > 0: rows.append((lm, elm, vf, evf))
    except ValueError:
        continue
d = np.array(rows); y, ey, x, ex = d[:, 0], d[:, 1], np.log10(d[:, 2]), d[:, 3] / d[:, 2] / np.log(10)
nll = lambda p: 0.5 * np.sum((y - p[0] - p[1] * (x - 2))**2 / (ey**2 + (p[1] * ex)**2 + np.exp(2 * p[2])) + np.log(ey**2 + (p[1] * ex)**2 + np.exp(2 * p[2])))
a, b, _ = minimize(nll, [10, 3.8, np.log(0.1)], method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 20000}).x
plt.rcParams.update({'font.size': 9, 'font.family': 'serif'})
fig, ax = plt.subplots(figsize=(4.2, 3.2))
ax.errorbar(x, y, xerr=ex, yerr=ey, fmt='o', ms=2.5, color='0.45', elinewidth=0.5, alpha=0.8, label=f'SPARC, $V_f$ ({len(x)} galaxies)')
xx = np.linspace(x.min() - 0.05, x.max() + 0.05, 50)
ax.fill_between(xx, a + 3.68 * (xx - 2), a + 3.79 * (xx - 2), color='C3', alpha=0.35, label=r'predicted $s=12/(1-n_{\rm eff})=3.68$--$3.79$')
ax.plot(xx, a + 4 * (xx - 2), 'k:', lw=1, label='slope 4 (deep MOND)')
ax.set_xlabel(r'$\log_{10} V_f$ [km s$^{-1}$]'); ax.set_ylabel(r'$\log_{10} M_{\rm bar}$ [$M_\odot$]')
ax.legend(fontsize=6.5, loc='upper left'); fig.tight_layout(); fig.savefig(sys.argv[2], metadata={'CreationDate': None, 'ModDate': None} if sys.argv[2].endswith('.pdf') else None)
print('saved', sys.argv[2], f'N = {len(x)}, centre log M_b = {a:.3f} at log V = 2, forward ML slope {b:.3f}')
