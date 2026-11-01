# 51 falsification RF5: the p = 0 symmetry forces BTFR slope exactly 4 only for galaxies on the deep orbit. Escape route of 51: the
# measured 3.62 comes from galaxies outside the deep regime. Test: fit the slope on subsamples by g_N at R_HI (outermost HI radius).
# Data: SPARC table 1 (Lelli, McGaugh, Schombert 2016), untrusted, passed as a path. Same likelihood as btfr_full_check.py.
# Run: uv run --with numpy --with scipy python -I btfr_deep_check.py /path/to/SPARC_Lelli2016c.mrt
import sys
import numpy as np
from scipy.optimize import minimize

rows = []
for line in open(sys.argv[1], encoding='latin-1'):
    f = line.split()
    if len(f) != 19 or not f[1].isdigit():
        continue
    try:
        rows.append(tuple(float(f[k]) for k in (2, 3, 5, 7, 8, 13, 14, 15, 16, 17)))
    except ValueError:
        continue
D, eD, inc, L, eL, MHI, RHI, V, eV, Q = np.array(rows).T
LN = np.log(10); G = 4.30091e-6; A0 = 1.2e-10 * 3.0857e19 / 1e6          # (km/s)^2/kpc

def fit(x, y, sx, sy):
    def nll(p):
        a, s, lsi = p
        var = sy**2 + s**2 * sx**2 + np.exp(2 * lsi)
        return 0.5 * np.sum((y - a - s * x)**2 / var + np.log(var))
    s0 = np.polyfit(x, y, 1)
    return minimize(nll, [s0[1], s0[0], np.log(0.1)], method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 20000}).x

base = (V > 0) & (Q <= 2) & (inc >= 30) & (RHI > 0)
Ms, Mg = 0.5 * L * 1e9, 1.33 * MHI * 1e9; Mb = Ms + Mg
gN = G * Mb / np.where(RHI > 0, RHI, np.inf)**2 / A0
x, y = np.log10(np.where(V > 0, V, 1)), np.log10(np.where(Mb > 0, Mb, 1))
sx = np.where(V > 0, eV / np.where(V > 0, V, 1) / LN, 0)
rel = np.sqrt((Ms / Mb)**2 * ((eL / np.where(L > 0, L, 1))**2 + (0.1 * LN)**2) + (Mg / Mb)**2 * 0.1**2)
sy = np.sqrt((rel / LN)**2 + (2 * eD / D / LN)**2)
rng = np.random.default_rng(0)
print("subsample                 N   slope  +- boot  (slope-4)/err   median g_N(R_HI)/a0")
out = {}
for name, cut in (('all', base), ('g_N < 0.3 a0', base & (gN < 0.3)), ('g_N < 0.1 a0', base & (gN < 0.1)),
                  ('g_N < 0.05 a0', base & (gN < 0.05)), ('g_N > 0.1 a0', base & (gN > 0.1)), ('gas > 50%', base & (Mg > Ms))):
    i0 = np.where(cut)[0]
    s = fit(x[i0], y[i0], sx[i0], sy[i0])[1]
    bs = [fit(x[j], y[j], sx[j], sy[j])[1] for j in (rng.choice(i0, len(i0)) for _ in range(200))]
    e = np.std(bs); out[name] = (s, e, len(i0))
    print(f"{name:22s} {len(i0):4d}  {s:5.2f}  {e:5.2f}   {(s - 4) / e:+5.1f}          {np.median(gN[i0]):.3f}")
s, e, n = out['g_N < 0.1 a0']
print(("SURVIVES " if abs(s - 4) < 2 * e else "REFUTED ") + f"RF5 deep subsample (g_N(R_HI) < 0.1 a0, N = {n}): slope {s:.2f} +- {e:.2f} -- "
      + ("consistent with the forced 4" if abs(s - 4) < 2 * e else "below 4 before correcting the selection bias of the cut (-0.20, btfr_selection_bias_check)"))
