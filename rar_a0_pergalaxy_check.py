# Is a0 a constant or a population regression? Per-galaxy fits on SPARC (papers/restart/34-sdiff-rotation.md, data section). A1-A4.
# For each galaxy: simple-mu RAR with its own a0, Upsilon_disk (lognormal prior 0.5, 0.1 dex), Upsilon_bulge (0.7, 0.1 dex), distance
# (Gaussian prior e_D) and inclination (Gaussian prior e_Inc) as nuisance parameters, as in Li+ 2018. Then: is the spread of log a0_i
# consistent with the errors (constant), and does a0_i correlate with galaxy properties (regression)?
# Data (not in git): SPARC_Lelli2016c.mrt and Rotmod_LTG/ from https://astroweb.case.edu/SPARC/
# Run: uv run --with numpy --with scipy python -I rar_a0_pergalaxy_check.py /path/SPARC_Lelli2016c.mrt /path/rotmod_dir
import sys, os
import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    assert cond

table, rdir = sys.argv[1], sys.argv[2]
cut = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
func = sys.argv[4] if len(sys.argv) > 4 else 'simple'                # 'simple' or 'rar' (McGaugh form)
pw = float(sys.argv[5]) if len(sys.argv) > 5 else 0.1                  # width of the Upsilon priors, dex           # optional: drop points with R < cut * R_max (beam smearing, non-circular motions)
meta = {}
for line in open(table, encoding='latin-1'):
    f = line.split()
    if len(f) == 19 and f[1].isdigit():
        meta[f[0]] = dict(D=float(f[2]), eD=float(f[3]), inc=float(f[5]), einc=float(f[6]), L=float(f[7]), SBeff=float(f[10]),
                          MHI=float(f[13]), Vf=float(f[15]), Q=int(f[17]))
kms2 = 1e6 / 3.0857e19
simple = lambda g, a: (g + np.sqrt(g**2 + 4 * g * a)) / 2
rarf = lambda g, a: g / (1 - np.exp(-np.sqrt(g / a)))
rows = []
for fn in sorted(os.listdir(rdir)):
    name = fn.replace('_rotmod.dat', '')
    m_ = meta.get(name)
    if m_ is None or m_['Q'] >= 3 or m_['inc'] < 30:
        continue
    d = np.loadtxt(os.path.join(rdir, fn), comments='#')
    R, Vo, eV, Vg, Vd, Vb = d[:, 0], d[:, 1], d[:, 2], d[:, 3], d[:, 4], d[:, 5]
    k = (R > 0) & (Vo > 0) & (eV > 0) & (R >= cut * R.max())
    if k.sum() < 8:
        continue
    R, Vo, eV, Vg, Vd, Vb = R[k], Vo[k], eV[k], Vg[k], Vd[k], Vb[k]
    hasb = np.any(Vb > 0)
    i0 = np.radians(m_['inc']); ei = np.radians(max(m_['einc'], 1.0))
    def model(p):
        la, lud, lub, D, inc = p
        gbar = (np.abs(Vg) * Vg + 10**lud * Vd**2 + 10**lub * Vb**2) / R * kms2          # independent of D (V^2 ~ D, R ~ D)
        gobs = Vo**2 / R * kms2 * (m_['D'] / D) * (np.sin(i0) / np.sin(inc))**2
        return gbar, gobs
    def chi2(p):
        la, lud, lub, D, inc = p
        if D <= 0 or not (0.1 < inc < np.pi / 2):
            return 1e9
        gbar, gobs = model(p)
        if np.any(gbar <= 0):
            return 1e9
        s = 2 * eV / Vo / np.log(10)
        c = np.sum(((np.log10(gobs) - np.log10((rarf if func == 'rar' else simple)(gbar, 10**la))) / s)**2)
        c += ((lud - np.log10(0.5)) / pw)**2 + (((lub - np.log10(0.7)) / pw)**2 if hasb else (lub - np.log10(0.7))**2 * 1e4)
        c += ((D - m_['D']) / m_['eD'])**2 + ((inc - i0) / ei)**2
        return c
    best = None
    for la0 in (-10.5, -10.0, -9.5):
        r = minimize(chi2, x0=[la0, np.log10(0.5), np.log10(0.7), m_['D'], i0], method='Nelder-Mead',
                     options={'xatol': 1e-5, 'fatol': 1e-6, 'maxiter': 6000, 'maxfev': 12000})
        if best is None or r.fun < best.fun:
            best = r
    # profile error on log a0: scan and find delta chi2 = 1 (other parameters re-minimised)
    la_b = best.x[0]
    def prof(la):
        return minimize(lambda q: chi2([la] + list(q)), x0=best.x[1:], method='Nelder-Mead',
                        options={'xatol': 1e-5, 'fatol': 1e-6, 'maxiter': 3000}).fun
    grid = la_b + np.linspace(-0.8, 0.8, 17)
    pr = np.array([prof(g) for g in grid]) - best.fun
    inside = grid[pr < 1.0]
    err = (inside.max() - inside.min()) / 2 if len(inside) > 1 else 0.05
    err = max(err, 0.02)
    fgas = 1.33 * m_['MHI'] / (1.33 * m_['MHI'] + 0.5 * m_['L'])
    rows.append((name, la_b, err, best.fun / max(len(R) - 5, 1), m_['SBeff'], m_['Vf'], fgas, len(R), inside.min() <= grid[0] or inside.max() >= grid[-1]))
names = [r[0] for r in rows]
la = np.array([r[1] for r in rows]); er = np.array([r[2] for r in rows]); rchi = np.array([r[3] for r in rows])
SB = np.array([r[4] for r in rows]); Vf = np.array([r[5] for r in rows]); fg = np.array([r[6] for r in rows]); edge = np.array([r[8] for r in rows])
use = (~edge) & (er < 0.5)
er_inf = er * np.sqrt(np.maximum(rchi, 1.0))                               # inflate by sqrt(reduced chi2) where the fit is poor
w = 1 / er[use]**2
mean = np.sum(w * la[use]) / np.sum(w)
chi = np.sum(w * (la[use] - mean)**2); dof = use.sum() - 1
print(f"   func {func}, prior {pw} dex, cut R >= {cut} R_max; galaxies fitted {len(rows)}, with bounded a0 {use.sum()}; weighted mean log a0 = {mean:.3f} (a0 = {10**mean:.3e}); "
      f"chi2/dof of a0_i about the mean = {chi / dof:.2f}; raw scatter {la[use].std():.3f} dex, median error {np.median(er[use]):.3f} dex")
wi = 1 / er_inf[use]**2
mean_i = np.sum(wi * la[use]) / np.sum(wi); chi_i = np.sum(wi * (la[use] - mean_i)**2)
print(f"   with errors inflated by sqrt(reduced chi2) (median {np.median(rchi[use]):.2f}): chi2/dof = {chi_i / dof:.2f}, median error {np.median(er_inf[use]):.3f} dex")
intr = np.sqrt(max(la[use].var() - np.median(er_inf[use])**2, 0))
print(f"   intrinsic scatter of log a0 (raw var - median err^2) ~ {intr:.3f} dex")
cors = {}
for lab, q in (('SBeff', np.log10(np.where(SB > 0, SB, np.nan))), ('Vflat', np.log10(np.where(Vf > 0, Vf, np.nan))), ('gas fraction', fg)):
    mk = use & np.isfinite(q)
    rho, p = spearmanr(q[mk], la[mk])
    cors[lab] = (rho, p, mk.sum())
    print(f"   Spearman log a0_i vs {lab:12s}: rho = {rho:+.3f}, p = {p:.2e}, N = {mk.sum()}")

ok(f"A1 per-galaxy fits done for {len(rows)} galaxies ({use.sum()} with bounded a0)", use.sum() > 50)
ok(f"A2 a constant a0 is not consistent with the per-galaxy fits: chi2/dof = {chi / dof:.1f} (raw errors), {chi_i / dof:.1f} (inflated); "
   f"intrinsic scatter ~ {intr:.2f} dex", chi_i / dof > 2)
if cut == 0:
    ok(f"A3 (all points) a0_i correlates with surface brightness (rho = {cors['SBeff'][0]:+.2f}, p = {cors['SBeff'][1]:.1e}), not with Vflat "
       f"(p = {cors['Vflat'][1]:.2f})", cors['SBeff'][1] < 0.01 and cors['Vflat'][1] > 0.05)
else:   # with inner points removed the SB correlation is not significant (34: likely an inner-region systematic); the spread stays
    print(f"   (R >= {cut} R_max, {func}, prior {pw} dex) SB correlation rho = {cors['SBeff'][0]:+.2f}, p = {cors['SBeff'][1]:.2f} -- unstable across settings (34)")
    ok(f"A3' (R >= {cut} R_max) the spread remains: chi2/dof = {chi_i / dof:.1f}, intrinsic {intr:.2f} dex", chi_i / dof > 2)
