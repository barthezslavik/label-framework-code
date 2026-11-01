# RAR from SPARC rotation curves (Lelli+ 2016): corpus interpolation formulas vs the empirical RAR and vs a formula-free
# binned regression. Data (not in git): Rotmod_LTG/*.dat and SPARC_Lelli2016c.mrt from https://astroweb.case.edu/SPARC/
# Run: uv run --with numpy --with scipy python -I rar_check.py /path/to/SPARC_Lelli2016c.mrt /path/to/rotmod_dir
import sys, os
import numpy as np
from scipy.optimize import minimize_scalar

table, rdir = sys.argv[1], sys.argv[2]
good = set()
for line in open(table, encoding='latin-1'):
    f = line.split()
    if len(f) == 19 and f[1].isdigit() and int(f[17]) < 3 and float(f[5]) >= 30:
        good.add(f[0])
kms2 = 1e6 / 3.0857e19                          # (km/s)^2/kpc -> m/s^2
gb, go, gid = [], [], []
for fn in os.listdir(rdir):
    name = fn.replace('_rotmod.dat', '')
    if name not in good:
        continue
    d = np.loadtxt(os.path.join(rdir, fn), comments='#')
    R, Vo, eV, Vg, Vd, Vb = d[:, 0], d[:, 1], d[:, 2], d[:, 3], d[:, 4], d[:, 5]
    m = (R > 0) & (Vo > 0) & (eV / Vo < 0.1)
    gbar = (np.abs(Vg) * Vg + 0.5 * Vd**2 + 0.7 * Vb**2) / R * kms2
    gobs = Vo**2 / R * kms2
    m &= gbar > 0
    gb += list(gbar[m]); go += list(gobs[m]); gid += [name] * int(m.sum())
gb, go, gid = np.array(gb), np.array(go), np.array(gid)
x, y = np.log10(gb), np.log10(go)
print(f"galaxies {len(good)}, points {len(gb)}")

models = {
    'RAR (McGaugh): g/(1-exp(-sqrt(g/a0)))': lambda g, a: g / (1 - np.exp(-np.sqrt(g / a))),
    'simple mu: (g+sqrt(g^2+4 g a0))/2':      lambda g, a: (g + np.sqrt(g**2 + 4 * g * a)) / 2,
    'C404: g + sqrt(g a0)':                    lambda g, a: g + np.sqrt(g * a),
    'paper3 nu=sqrt(1+1/y): sqrt(g^2+g a0)':   lambda g, a: np.sqrt(g**2 + g * a),
}
res = {}
for name, fmod in models.items():
    obj = lambda la: np.mean((y - np.log10(fmod(gb, 10**la)))**2)
    r = minimize_scalar(obj, bounds=(-11, -9), method='bounded')
    rms = np.sqrt(r.fun); bias = np.mean(y - np.log10(fmod(gb, 10**r.x)))
    res[name] = (10**r.x, rms)
    print(f"{name:42s} best a0 = {10**r.x:.3e} m/s^2   rms = {rms:.4f} dex   mean resid = {bias:+.4f}")
# fixed corpus a0 values in the RAR function
for lab, a0 in (('cH0/2pi', 1.08e-10), ('3cH0/16', 1.275e-10), ('cH0', 6.80e-10)):
    f = models['RAR (McGaugh): g/(1-exp(-sqrt(g/a0)))']
    print(f"RAR with a0 = {lab:8s} ({a0:.2e}): rms = {np.sqrt(np.mean((y - np.log10(f(gb, a0)))**2)):.4f} dex")
# formula-free regression: binned mean of log g_obs in bins of log g_bar
bins = np.linspace(x.min(), x.max(), 25)
idx = np.digitize(x, bins)
mean = np.array([y[idx == k].mean() if np.any(idx == k) else np.nan for k in range(1, len(bins) + 1)])
pred = mean[idx - 1]
print(f"formula-free binned regression (24 bins): rms = {np.sqrt(np.nanmean((y - pred)**2)):.4f} dex")

# galaxy-level bootstrap (points within a galaxy are correlated): refit a0 for each model in each resample
rng = np.random.default_rng(0)
names = np.unique(gid); groups = {n: np.where(gid == n)[0] for n in names}
boot = {k: [] for k in models}; bmse = {k: [] for k in models}
for _ in range(300):
    pick = np.concatenate([groups[n] for n in rng.choice(names, len(names))])
    for k, fmod in models.items():
        o = lambda la: np.mean((y[pick] - np.log10(fmod(gb[pick], 10**la)))**2)
        r = minimize_scalar(o, bounds=(-11, -9), method='bounded')
        boot[k].append(r.x); bmse[k].append(r.fun)
ref = 'RAR (McGaugh): g/(1-exp(-sqrt(g/a0)))'
for k in models:
    la = np.array(boot[k]); d = np.array(bmse[k]) - np.array(bmse[ref])
    print(f"{k:42s} a0 = {10**la.mean():.2e} (+{10**(la.mean()+la.std())-10**la.mean():.1e}/-{10**la.mean()-10**(la.mean()-la.std()):.1e})"
          f"   d(MSE) vs RAR = {d.mean():+.5f} +- {d.std():.5f}")

# analysis/sparc/COMPLETE_THEORY_SUMMARY.md: single power law g_obs = g_bar^(4/7) a0^(3/7), claimed slope 0.5708 +- 0.0092 on SPARC
s_all = np.polyfit(x, y, 1)[0]
pl = lambda la, s: s * x + (1 - s) * la
best = minimize_scalar(lambda la: np.mean((y - pl(la, 4 / 7))**2), bounds=(-11, -9), method='bounded')
print(f"single power law: free OLS slope over all points = {s_all:.3f}; with slope fixed at 4/7: a0 = {10**best.x:.2e}, rms = {np.sqrt(best.fun):.4f} dex "
      f"(RAR function: 0.1327)")
for lo_, hi_ in ((-12.5, -11), (-11, -10), (-10, -9), (-9, -8)):
    m = (x > lo_) & (x <= hi_)
    if m.sum() > 20:
        print(f"   local slope for log g_bar in ({lo_},{hi_}]: {np.polyfit(x[m], y[m], 1)[0]:.2f}  (n={m.sum()})")
