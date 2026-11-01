# papers/restart/18-observations.md O7, full error treatment: BTFR slope with errors on both axes and intrinsic scatter.
# Prediction fixed before the comparison (18 §1, commit 2eb8d90): slope exactly 4.
# Error budget per galaxy (SPARC table, Lelli, McGaugh & Schombert 2016):
#   logV: e_Vflat;  logM: e_L, Upsilon* scatter 0.1 dex on the stellar part, 10 % on M_HI, distance e_D (enters M as D^2).
# Model: logM = a + s logV, Gaussian likelihood with variance sM^2 + s^2 sV^2 + sig_int^2 (maximised over a, s, sig_int).
# Data: https://astroweb.case.edu/SPARC/SPARC_Lelli2016c.mrt (not stored in git).
# Run: uv run --with numpy --with scipy python -I btfr_full_check.py /path/to/SPARC_Lelli2016c.mrt
import sys
import numpy as np
from scipy.optimize import minimize

rows = []
for line in open(sys.argv[1], encoding='latin-1'):
    f = line.split()
    if len(f) != 19 or not f[1].isdigit():
        continue                      # Galaxy T D e_D f_D Inc e_Inc L e_L Reff SBeff Rdisk SBdisk MHI RHI Vflat e_Vflat Q Ref
    try:
        rows.append(tuple(float(f[k]) for k in (2, 3, 5, 7, 8, 13, 15, 16, 17)))
    except ValueError:
        continue
D, eD, inc, L, eL, MHI, V, eV, Q = np.array(rows).T
LN = np.log(10)

def data(ups, qmax, imin):
    s = (V > 0) & (Q <= qmax) & (inc >= imin)
    Ms, Mg = ups * L[s] * 1e9, 1.33 * MHI[s] * 1e9
    Mb = Ms + Mg
    x, y = np.log10(V[s]), np.log10(Mb)
    sx = eV[s] / (V[s] * LN)
    rel = np.sqrt((Ms / Mb)**2 * ((eL[s] / L[s])**2 + (0.1 * LN)**2) + (Mg / Mb)**2 * 0.1**2)
    sy = np.sqrt((rel / LN)**2 + (2 * eD[s] / (D[s] * LN))**2)
    return x, y, sx, sy

def fit(x, y, sx, sy, scatter_in='M'):
    def nll(p):
        a, s, lsi = p
        si2 = np.exp(2 * lsi) * (1 if scatter_in == 'M' else s**2)     # intrinsic scatter along logM, or along logV (scaled by s)
        var = sy**2 + s**2 * sx**2 + si2
        return 0.5 * np.sum((y - a - s * x)**2 / var + np.log(var))
    s0 = np.polyfit(x, y, 1)
    r = minimize(nll, [s0[1], s0[0], np.log(0.1)], method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 20000})
    return r.x

rng = np.random.default_rng(0)
print("variant                              N   slope  +- boot   sig_int  (slope-4)/err")
results = {}
for sc in ('M', 'V'):
    for ups in (0.4, 0.5, 0.6):
        for qmax, imin in ((2, 30), (1, 30), (2, 45)):
            x, y, sx, sy = data(ups, qmax, imin)
            a, s, lsi = fit(x, y, sx, sy, sc)
            bs = []
            for _ in range(200):
                i = rng.integers(0, len(x), len(x))
                bs.append(fit(x[i], y[i], sx[i], sy[i], sc)[1])
            e = np.std(bs)
            results[(sc, ups, qmax, imin)] = (s, e)
            print(f"scatter in log{sc} Ups={ups} Q<={qmax} i>={imin:2d}  {len(x):3d}  {s:5.2f}  {e:5.2f}   {np.exp(lsi):.3f}    {(s - 4) / e:+.1f}")
s = np.mean([results[(sc, 0.5, 2, 30)][0] for sc in ('M', 'V')])
e = np.mean([results[(sc, 0.5, 2, 30)][1] for sc in ('M', 'V')])
print(f"min / max over all variants: {min(v[0] for v in results.values()):.2f} / {max(v[0] for v in results.values()):.2f}")
spread = max(v[0] for v in results.values()) - min(v[0] for v in results.values())
print(f"baseline slope = {s:.2f} +- {e:.2f} (stat) +- {spread / 2:.2f} (systematic: Upsilon*, quality, inclination cuts)")
print(f"slope 4: {(s - 4) / np.hypot(e, spread / 2):+.1f} sigma (stat + syst combined)")
