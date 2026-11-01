# paper5 v2 (Codex r1): the signature "S8 tied to the phantom depth under m(phi), not under the discrete fluctuation" was computed in doc 97
# with two definitions of the depth: m(phi) used wbar4 = [w(0) + 2 w(0.5) + w(1)]/4, the fluctuation the mean over ln a from
# rho_DE(z=1)/rho_DE(0). Checking the second against the first found a sign error in doc 97: rho(z=1)/rho(0) = 2^(3(1+wbar)), so
# wbarR = -1 + ln[rho(1)/rho(0)] / (3 ln 2), while de_phantom_compare_check (and its stored output phantom_compare_300.txt) used
# -1 - ln(.)/(3 ln 2). Here: both definitions for m(phi) (model of ga_growth_coupling_check, one fluid, S8 = 0.831 x growth ratio, as in
# 97 PC4), and the corrected PC2/PC3 numbers of the fluctuation from the stored realizations. WD1-WD4.
# Run: ~/.venvs/jev/bin/python ga_wbar_definition_check.py
import os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

src = open(os.path.join(HERE, 'ga_growth_coupling_check.py')).read().split('D0 = model(0.0)')[0]
old = "    return d[-1], at(0.0), at(0.5), at(1.0)"
assert src.count(old) == 1
src = src.replace(old, "    rz = lambda z: rde[np.argmin(abs(N + np.log(1 + z)))]\n"
                       "    return d[-1], at(0.0), at(0.5), at(1.0), -1 + np.log(rz(1.0) / rz(0.0)) / (3 * np.log(2.0))")
gc = {'__name__': 'gc'}; exec(src, gc)
D0 = gc['model'](0.0)[0]
rows = []
for b in (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06):
    D, w0, w05, w1, wR = gc['model'](b)
    rows.append((b, (w0 + 2 * w05 + w1) / 4, wR, 0.831 * D / D0))
rows = np.array(rows)
print("   m(phi): " + "; ".join(f"beta {r[0]:.2f}: wbar4 {r[1]:.3f}, wbarR {r[2]:.3f}, S8 {r[3]:.3f}" for r in rows))
s4 = np.polyfit(rows[1:, 1], rows[1:, 3], 1)[0]; sR = np.polyfit(rows[1:, 2], rows[1:, 3], 1)[0]
ok(f"WD1 m(phi), beta = 0.01..0.06: both definitions phantom (wbarR {rows[-1, 2]:.3f}, wbar4 {rows[-1, 1]:.3f} at beta = 0.06); "
   f"dS8/dwbar = {s4:+.3f} with wbar4 (97 PC4: -0.540), {sR:+.3f} with wbarR",
   abs(s4 + 0.540) < 0.01 and sR < -0.3 and np.all(rows[1:, 2] < -1) and abs(rows[0, 2] + 1) < 1e-3)

res = np.loadtxt(os.path.join(HERE, 'phantom_compare_300.txt'))
res = res[np.isfinite(res[:, 1])]
stored = res[:, 8]; wR = -1 + np.log(res[:, 10]) / (3 * np.log(2.0))   # column 10 = rho_L(z=1)/rho_L(0)
ok(f"WD2 the stored wbar of doc 97 is the flipped one: max |stored - (-2 - wbarR)| = {np.max(abs(stored + 2 + wR)):.1e}",
   np.max(abs(stored + 2 + wR)) < 1e-4)

dchi = res[:, 1] - 1421.22; top = dchi < -5                             # 97: Delta chi2 against the constant-Lambda fit, 1421.22
S8rel = res[:, 5] * np.sqrt(res[:, 6])                                  # S8 up to the common factor 0.831 / (D0 sqrt(Om0))
cc = np.corrcoef(wR, S8rel)[0, 1]
ok(f"WD3 corrected 97 PC2: realizations with Delta chi2 < -5: {top.sum()}, mean wbar {np.mean(wR[top]):.3f} (all: {np.mean(wR):.3f}), "
   f"phantom-like past (wbar < -1) in {np.mean(wR[top] < -1):.2f} of them", top.sum() == 25)
# the two-fluid coupling of the joint fit (doc 126; model2 of ga_s8_wbar_forecast_check, SW7): phi on dark dust only, baryons uncoupled
src2 = open(os.path.join(HERE, 'ga_s8_wbar_forecast_check.py')).read()
src2 = src2[src2.index('def model2('):src2.index('D0b = model2(0.0)[0]')]
old2 = "    return (rc0 * yy[2] + rb0 * yy[4]) / (rc0 + rb0) * np.sqrt(Omt / Om), at(0.0), at(0.5), at(1.0)"
assert src2.count(old2) == 1
src2 = src2.replace(old2, "    rz = lambda z: rde[np.argmin(abs(N + np.log(1 + z)))]\n" + old2 + ", -1 + np.log(rz(1.0) / rz(0.0)) / (3 * np.log(2.0))")
from scipy.integrate import solve_ivp as _ivp
from scipy.optimize import brentq as _bq
m2 = {'np': np, '_ivp': _ivp, '_bq': _bq}; exec(src2, m2)
D0b = m2['model2'](0.0)[0]
rows2 = []
for b in (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06):
    D, w0, w05, w1, wR2 = m2['model2'](b)
    rows2.append((b, (w0 + 2 * w05 + w1) / 4, wR2, 0.831 * D / D0b))
rows2 = np.array(rows2)
print("   two fluids: " + "; ".join(f"beta {r[0]:.2f}: wbar4 {r[1]:.3f}, wbarR {r[2]:.3f}, S8 {r[3]:.3f}" for r in rows2))
s42 = np.polyfit(rows2[1:, 1], rows2[1:, 3], 1)[0]; sR2 = np.polyfit(rows2[1:, 2], rows2[1:, 3], 1)[0]
ok(f"WD4 the signature with one definition (wbarR), in the configuration of the fit (two fluids): m(phi) dS8/dwbarR = {sR2:+.3f} "
   f"(wbar4: {s42:+.3f}; one fluid: {sR:+.3f}); fluctuation correlation S8-wbarR {cc:+.2f} (sign of 97 reversed): S8 is tied to the phantom "
   f"depth only under m(phi)", sR2 < -0.3 and abs(cc) < 0.15 and np.all(rows2[1:, 2] < -1))
