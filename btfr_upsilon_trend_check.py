# 60 RH3: hypothesis 'the true BTFR slope is 4 and the measured 3.6 comes from a stellar mass-to-light ratio that varies with luminosity'.
# Upsilon_disk = 0.5 (L / 1e10 Lsun)^gamma; find gamma that gives slope 4 (full sample, same likelihood as btfr_full_check.py) and the
# implied Upsilon range. Data untrusted, passed as a path.
# Run: uv run --with numpy --with scipy python -I btfr_upsilon_trend_check.py /path/SPARC_Lelli2016c.mrt
import sys
import numpy as np
from scipy.optimize import minimize, brentq

rows = []
for line in open(sys.argv[1], encoding='latin-1'):
    f = line.split()
    if len(f) != 19 or not f[1].isdigit():
        continue
    try:
        rows.append(tuple(float(f[k]) for k in (2, 3, 5, 7, 8, 13, 15, 16, 17)))
    except ValueError:
        continue
D, eD, inc, L, eL, MHI, V, eV, Q = np.array(rows).T
s = (V > 0) & (Q <= 2) & (inc >= 30) & (L > 0)
D, eD, L, eL, MHI, V, eV = D[s], eD[s], L[s], eL[s], MHI[s], V[s], eV[s]
LN = np.log(10)

def slope(gamma):
    ups = 0.5 * (L / 10.0)**gamma                       # L in 1e9 Lsun in the table
    Ms, Mg = ups * L * 1e9, 1.33 * MHI * 1e9; Mb = Ms + Mg
    x, y = np.log10(V), np.log10(Mb); sx = eV / V / LN
    rel = np.sqrt((Ms / Mb)**2 * ((eL / L)**2 + (0.1 * LN)**2) + (Mg / Mb)**2 * 0.1**2)
    sy = np.sqrt((rel / LN)**2 + (2 * eD / D / LN)**2)
    def nll(p):
        a, b, ls = p
        var = sy**2 + b**2 * sx**2 + np.exp(2 * ls)
        return 0.5 * np.sum((y - a - b * x)**2 / var + np.log(var))
    s0 = np.polyfit(x, y, 1)
    return minimize(nll, [s0[1], s0[0], np.log(0.1)], method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 20000}).x[1]

for g in (0.0, 0.1, 0.2, 0.3):
    print(f"   gamma = {g:.1f}: slope {slope(g):.2f}")
g4 = brentq(lambda g: slope(g) - 4.0, 0.0, 1.0)
span = np.log10(L.max() / L.min())
print(f"RESULT slope 4 needs Upsilon ~ L^{g4:.2f}: over {span:.1f} decades of L, Upsilon changes by {g4 * span:.2f} dex "
      f"(x{10**(g4 * span):.0f}); population synthesis at 3.6 um allows ~0.1-0.2 dex")
