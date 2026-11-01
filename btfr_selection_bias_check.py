# 56 PU7: is the deep-regime BTFR slope (51 RF5, 53: 3.60 +- 0.11 for g_N(R_HI) < 0.1 a0) biased by selecting on g_N = G M_b / R_HI^2,
# a function of the dependent variable M_b? Mocks: real Vflat, R_HI and errors; M_b drawn from an EXACT slope-4 BTFR with the fitted
# intrinsic scatter; the same cut and the same likelihood as btfr_deep_check.py. Data untrusted, passed as a path.
# Run: uv run --with numpy --with scipy python -I btfr_selection_bias_check.py /path/SPARC_Lelli2016c.mrt [intrinsic scatter in dex]
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
LN = np.log(10); G = 4.30091e-6; A0 = 1.2e-10 * 3.0857e19 / 1e6
s = (V > 0) & (Q <= 2) & (inc >= 30) & (RHI > 0)
D, eD, L, eL, MHI, RHI, V, eV = D[s], eD[s], L[s], eL[s], MHI[s], RHI[s], V[s], eV[s]
Ms, Mg = 0.5 * L * 1e9, 1.33 * MHI * 1e9; Mb = Ms + Mg
x = np.log10(V); y = np.log10(Mb); sx = eV / V / LN
rel = np.sqrt((Ms / Mb)**2 * ((eL / L)**2 + (0.1 * LN)**2) + (Mg / Mb)**2 * 0.1**2)
sy = np.sqrt((rel / LN)**2 + (2 * eD / D / LN)**2)

def fit(x, y, sx, sy):
    def nll(p):
        a, b, ls = p
        var = sy**2 + b**2 * sx**2 + np.exp(2 * ls)
        return 0.5 * np.sum((y - a - b * x)**2 / var + np.log(var))
    s0 = np.polyfit(x, y, 1)
    return minimize(nll, [s0[1], s0[0], np.log(0.1)], method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 20000}).x

a_all, b_all, ls_all = fit(x, y, sx, sy)
a4 = np.median(y - 4 * x)                                   # slope-4 normalisation through the data
SIG = float(sys.argv[2]) if len(sys.argv) > 2 else np.exp(ls_all)
print(f"   intrinsic scatter used: {SIG:.3f} dex (fitted {np.exp(ls_all):.3f})")
rng = np.random.default_rng(11)
res = {'all': [], 'cut': []}
for _ in range(200):
    xt = x + rng.normal(0, sx)                               # true log V scatter around the measured one (keeps the V distribution)
    yt = a4 + 4 * xt + rng.normal(0, SIG, len(x))
    yo = yt + rng.normal(0, sy); xo = xt + rng.normal(0, sx)
    gN = G * 10**yo / RHI**2 / A0
    res['all'].append(fit(xo, yo, sx, sy)[1])
    k = gN < 0.1
    res['cut'].append(fit(xo[k], yo[k], sx[k], sy[k])[1])
for k_, v_ in res.items():
    print(f"   mock with true slope 4, {k_:4s}: recovered slope {np.mean(v_):.2f} +- {np.std(v_):.2f}")
bias = np.mean(res['cut']) - 4
print(f"RESULT selection bias of the deep cut: {bias:+.2f}; real deep slope 3.60 corrected -> {3.60 - bias:.2f}; "
      f"full-sample slope in mocks {np.mean(res['all']):.2f} (real {b_all:.2f})")
