# Figure for paper8: one area coefficient for a plane and a sphere (RE3, RE4) and the cylinder against its flat control (RE8).
# Inputs: rindler_eta_2026-11-01.out (output of ga_rindler_eta_check, 100-site lattice sphere entropies) and cyl_convergence_eta.out.
# Run: ~/.venvs/jev/bin/python fig_paper8_eta.py OUT.pdf
import os, re, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.dirname(HERE)
txt = open(os.path.join(D, 'rindler_eta_2026-11-01.out')).read()
rows = re.findall(r'sphere n = +(\d+): S = +([\d.]+)', txt)
n = np.array([int(a) for a, _ in rows]); S = np.array([float(b) for _, b in rows]); R = n + 0.5
ratio = S / (4 * np.pi * R**2)
eta_plane, fits_lo, fits_hi = 0.023510, 0.023492, 0.023504
cyl = re.findall(r'R=([\d.]+) Ns=(\d+) GL(\d+):(?: flat [\d.]+ cyl [\d.]+)? rel ([-\d.e+]+)', open(os.path.join(D, 'cyl_convergence_eta.out')).read())

plt.rcParams.update({'font.size': 9, 'font.family': 'serif'})
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
ax[0].plot(R, ratio, 'o', ms=4, color='C0', label=r'sphere, $S/4\pi R^2$ (100 sites)')
ax[0].axhspan(fits_lo, fits_hi, color='C0', alpha=0.15, label='sphere, area coefficient (fits)')
ax[0].axhline(eta_plane, color='C3', lw=1.2, label=r'plane, $\eta_{\rm plane}=0.02351$')
ax[0].set_xlabel(r'radius $R/a$'); ax[0].set_ylabel(r'entanglement per area $\times a^2$')
ax[0].set_ylim(0.02340, 0.02356); ax[0].legend(fontsize=7, loc='lower right'); ax[0].set_title('(a) plane and sphere', fontsize=9)
for Ns, mk in (('150', 'o'), ('250', 's')):
    pts = sorted((float(r_), abs(float(rel))) for r_, ns, gl, rel in cyl if ns == Ns)
    if pts:
        xs, ys = zip(*pts); ax[1].plot(xs, ys, mk + '-', ms=4, label=f'lattice {Ns} sites')
ax[1].set_yscale('log'); ax[1].set_xlabel(r'cylinder radius $R/a$'); ax[1].set_ylabel('|cylinder / flat control $-1$|')
ax[1].legend(fontsize=7); ax[1].set_title('(b) curvature enters subleading terms', fontsize=9)
fig.tight_layout(); fig.savefig(sys.argv[1], metadata={'CreationDate': None, 'ModDate': None} if sys.argv[1].endswith('.pdf') else None); print('saved', sys.argv[1], 'sphere points', len(R))
