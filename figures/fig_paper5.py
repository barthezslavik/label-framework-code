# Figure for paper5: (a) w(z) of the CPL best fits and of the fits restricted to w >= -1 (doc 70, 86b; de_desi_fit_check,
# de_nec_fit_check); (b) the S8(wbar) line of the coupling from the joint fit (SW8, doc 117/126) with the lensing values.
# Numbers are those of the paper's tables; no refit here.
# Run: ~/.venvs/jev/bin/python fig_paper5.py OUT.pdf
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
z = np.linspace(0, 2.5, 300); a = 1 / (1 + z)
cpl = [('DESI + CMB', -0.44, -1.67, 'C0'), ('+ Pantheon+', -0.85, -0.52, 'C1'), ('+ DES-SN5YR', -0.82, -0.64, 'C2')]
nec = [('+ Pantheon+, $w\\geq-1$', -0.96, -0.04, 'C1'), ('+ DES-SN5YR, $w\\geq-1$', -0.95, -0.05, 'C2')]
plt.rcParams.update({'font.size': 9, 'font.family': 'serif'})
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
for lab, w0, wa, c in cpl: ax[0].plot(z, w0 + wa * (1 - a), color=c, lw=1.3, label=lab)
for lab, w0, wa, c in nec: ax[0].plot(z, w0 + wa * (1 - a), color=c, lw=1.0, ls='--', label=lab)
ax[0].axhline(-1, color='k', lw=0.8, ls=':'); ax[0].set_xlabel('redshift $z$'); ax[0].set_ylabel('$w(z)$')
ax[0].set_ylim(-2.2, -0.3); ax[0].legend(fontsize=6.5, loc='lower left'); ax[0].set_title('(a) CPL fits, free and without phantom', fontsize=9)
wb = np.linspace(-1.12, -1.0, 200); x = wb + 1
ax[1].plot(wb, 0.805 - 0.572 * x - 0.37 * x**2, color='C3', lw=1.5, label='coupling $m(\\phi)$, joint fit')
ax[1].axhspan(0.818 - 0.013, 0.818 + 0.013, color='C0', alpha=0.15, label='KiDS--DES $0.818\\pm0.013$')
ax[1].axhspan(0.789 - 0.012, 0.789 + 0.012, color='C2', alpha=0.15, label='DES Y6 $0.789\\pm0.012$')
ax[1].axvline(-1.078, color='k', lw=0.8, ls=':'); ax[1].text(-1.077, 0.866, '$3\\sigma$ threshold\n(assumed errors)', fontsize=6.5)
ax[1].set_xlabel('phantom depth $\\bar w$ (four-point mean)'); ax[1].set_ylabel('$S_8$'); ax[1].set_ylim(0.77, 0.88)
ax[1].set_xticks([-1.0,-1.025,-1.05,-1.075,-1.1]); ax[1].set_xticklabels(['-1.00','','-1.05','','-1.10']); ax[1].invert_xaxis(); ax[1].legend(fontsize=6.5, loc='upper left'); ax[1].set_title('(b) the coupling ties $S_8$ to the phantom depth', fontsize=9)
fig.tight_layout(); fig.savefig(sys.argv[1], metadata={'CreationDate': None, 'ModDate': None} if sys.argv[1].endswith('.pdf') else None); print('saved', sys.argv[1])
