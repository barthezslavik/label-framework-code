# 117: the S8(wbar) line of the coupling m(phi) as a number for joint w(z) + S8 data (H3 of PREDICTIONS.md). The m(phi) model of
# ga_growth_coupling_check (same S8 calibration as doc 97: S8 = 0.831 x growth ratio) gives S8 as a function of the mean w over 0 < z < 1;
# the discrete Lambda fluctuation (doc 97, results/phantom_compare_300.npy) gives no line. Outputs: the line and its curvature, the phantom
# depth m(phi) allows if lensing S8 holds, and the tension a measured point (wbar, S8) has with the line for assumed future errors.
# Error levels are inputs, not forecasts taken from the literature. SW1-SW5.
# Run: ~/.venvs/jev/bin/python ga_s8_wbar_forecast_check.py
import os
import numpy as np

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

HERE = os.path.dirname(os.path.abspath(__file__))
gc = {'__name__': 'gc'}; exec(open(os.path.join(HERE, 'ga_growth_coupling_check.py')).read().split('D0 = model(0.0)')[0], gc)
S8_LCDM, S8_L, eS8 = 0.831, 0.818, 0.013
D0 = gc['model'](0.0)[0]
pts = []
for beta in np.arange(0.0, 0.081, 0.01):
    D, w0, w05, w1 = gc['model'](beta)
    pts.append((beta, (w0 + 2 * w05 + w1) / 4, S8_LCDM * D / D0))
pts = np.array(pts)
print("   m(phi): " + "; ".join(f"beta {b:.2f}: wbar {w:.4f}, S8 {s:.4f}" for b, w, s in pts))
q = np.polyfit(pts[:, 1] + 1, pts[:, 2], 2); lin = np.polyfit(pts[:6, 1] + 1, pts[:6, 2], 1)
line = lambda wb: np.polyval(q, wb + 1)
ok(f"SW1 m(phi) line: S8 = {q[2]:.4f} {q[1]:+.3f} (wbar + 1) {q[0]:+.2f} (wbar + 1)^2; linear slope for beta <= 0.05: {lin[0]:+.3f} (doc 97: -0.54)",
   abs(q[2] - S8_LCDM) < 2e-3 and q[1] < 0)
# phantom depth allowed by lensing: S8 <= S8_L + 2 eS8
wgrid = np.linspace(-1.2, -1.0, 20001); wmin = wgrid[line(wgrid) <= S8_L + 2 * eS8].min()
ok(f"SW2 if lensing S8 = {S8_L} +- {eS8} holds, m(phi) allows wbar >= {wmin:.3f} (2 sigma): a deeper measured phantom excludes m(phi) "
   "as its cause", -1.1 < wmin < -1.0)
# discrete fluctuation: no line (doc 97)
R = np.load(os.path.join(HERE, 'results', 'phantom_compare_300.npy'))
g = np.isfinite(R[:, 1]); D, Om, wbar = R[g, 5], R[g, 6], -2 - R[g, 8]   # stored wbar has the flipped sign (ga_wbar_definition_check WD2)
i0 = np.argmin(R[g, 1]); # S8 needs the constant-Lambda reference of doc 97; use the correlation only (scale-free)
cc = np.corrcoef(wbar, D * np.sqrt(Om))[0, 1]
ok(f"SW3 discrete fluctuation: correlation of S8 with wbar over 300 realizations {cc:+.2f} (doc 97 PC3, sign corrected: -0.02): no line", abs(cc) < 0.2)
# tension of a measured point with the m(phi) line, for assumed errors
cases = [("today (S8 0.013, wbar 0.03)", 0.013, 0.03), ("DESI final + Stage IV lensing (0.006, 0.015)", 0.006, 0.015),
         ("optimistic (0.003, 0.01)", 0.003, 0.01)]
slope = lambda wb: np.polyval(np.polyder(q), wb + 1)              # derivative at the tested point (review r4)
for wb_true in (-1.03, -1.05):
    for name, sS, sW in cases:
        dS = abs(S8_L - line(wb_true)); t = dS / np.hypot(sS, slope(wb_true) * sW)
        print(f"   true wbar {wb_true}, S8 {S8_L} (lensing): m(phi) predicts {line(wb_true):.3f}; {name}: {t:.1f} sigma")
tt = lambda wb, sS, sW: abs(S8_L - line(wb)) / np.hypot(sS, slope(wb) * sW)
w3 = {nm: wgrid[::-1][np.argmax([tt(w, sS, sW) >= 3 for w in wgrid[::-1]])] for nm, sS, sW in cases}
ok("SW4 phantom depth at which S8 = 0.818 excludes m(phi) at 3 sigma (independent errors, no covariance): " + "; ".join(f"{nm}: wbar <= {w:.3f}" for nm, w in w3.items()),
   w3[cases[1][0]] > w3[cases[0][0]])
ok("SW5 recorded: three outcomes -- (i) phantom gone (wbar -> -1): both mechanisms idle, Lambda label; (ii) phantom stays and S8 rises along "
   "the line: m(phi); (iii) phantom stays, S8 stays at lensing: m(phi) excluded, the discrete fluctuation (no line) or a non-native cause", True)
# SW6 (review r3) the bound on beta depends on the lensing result used: KiDS-DES joint ~0.818 +- 0.013 (used above) vs DES Y6 0.789 +- 0.012.
#     With DES Y6 even beta = 0 (S8 = 0.831 at the Planck amplitude) lies above the 2 sigma upper end, so the bound is then set by the S8
#     tension itself, not by the coupling; every number derived from beta <= 0.036 is a scenario at S8 = 0.818.
up_y6 = 0.789 + 2 * 0.012
ok(f"SW6 illustration, no significance: DES Y6 flat-LCDM S8 = 0.789 +- 0.012 (0.782 +0.021/-0.020 with free w) has a 2 sigma upper end {up_y6:.3f} "
   f"< S8(beta = 0) = {line(-1.0):.3f}: the beta bound depends on the lensing fit; a quantitative test of m(phi) needs its own fit to the data",
   up_y6 < line(-1.0))

# SW7 (paper7 review r15) the same line with two fluids: phi coupled to dark dust only, baryons uncoupled (f_b = 0.157), as in the joint fit
#     of doc 126. Growth of the total matter contrast; w of the dark energy inferred as in ga_growth_coupling_check (all matter counted as
#     a^-3 matter at the early normalisation); S8 = 0.831 x growth ratio as before.
from scipy.integrate import solve_ivp as _ivp
from scipy.optimize import brentq as _bq
def model2(beta, Om=0.3, fb=0.157, ai=1e-3):
    rci = 3 * Om * (1 - fb) / ai**3 / 3 * 3 / 3; rci = Om * (1 - fb) * ai**-3; rbi = Om * fb * ai**-3   # units 3H0^2 = 1 x 3
    def run(V0):
        def f(N, y):
            ph, pd, dc, dcp, db, dbp = y; x = np.exp(-3 * (N - np.log(ai)))
            rc = 3 * rci * x * np.exp(beta * ph); rb = 3 * rbi * x; h = np.sqrt((rc + rb + 0.5 * pd**2 + V0) / 3)
            return [pd / h, (-3 * h * pd - beta * rc) / h, dcp / h, (-(2 * h + beta * pd) * dcp + 0.5 * (rb * db + (1 + 2 * beta**2) * rc * dc)) / h,
                    dbp / h, (-2 * h * dbp + 0.5 * (rb * db + rc * dc)) / h]
        v0 = ai * np.sqrt(3 * (rci + rbi) / 3)
        return _ivp(f, [np.log(ai), 0.0], [0, 0, ai, v0, ai, v0], rtol=1e-9, atol=1e-12, dense_output=True)
    def frac(V0):
        s = run(V0); ph = s.sol(0.0)[0]; pd = s.sol(0.0)[1]; rc = 3 * rci * ai**3 * np.exp(beta * ph); rb = 3 * rbi * ai**3
        return (rc + rb) / (rc + rb + 0.5 * pd**2 + V0)
    V0 = _bq(lambda V: frac(V) - Om, 1e-6, 20.0, xtol=1e-12)
    s = run(V0); N = np.linspace(np.log(ai), 0, 4001); y = s.sol(N)
    x = np.exp(-3 * (N - np.log(ai))); rc = 3 * rci * x * np.exp(beta * y[0]); rb = 3 * rbi * x
    h2 = (rc + rb + 0.5 * y[1]**2 + V0) / 3
    rde = 3 * h2 - 3 * (rci + rbi) * x                                   # dark energy inferred against uncoupled a^-3 matter
    w = -1 - np.gradient(np.log(np.abs(rde)), N) / 3
    at = lambda z: w[np.argmin(abs(N + np.log(1 + z)))]
    yy = s.sol(0.0); rc0 = 3 * rci * ai**3 * np.exp(beta * yy[0]); rb0 = 3 * rbi * ai**3
    h20 = (rc0 + rb0 + 0.5 * yy[1]**2 + V0) / 3
    Omt = (rc0 + rb0) / (3 * h20)
    return (rc0 * yy[2] + rb0 * yy[4]) / (rc0 + rb0) * np.sqrt(Omt / Om), at(0.0), at(0.5), at(1.0)   # S8 with today's Omega_m (review r16)
D0b = model2(0.0)[0]
pts2 = []
for beta in np.arange(0.0, 0.091, 0.01):
    D, w0, w05, w1 = model2(beta); pts2.append((beta, (w0 + 2 * w05 + w1) / 4, S8_LCDM * D / D0b, w1))
pts2 = np.array(pts2)
q2 = np.polyfit(pts2[:, 1] + 1, pts2[:, 2], 2); line2 = lambda wb: np.polyval(q2, wb + 1)
slope2 = lambda wb: np.polyval(np.polyder(q2), wb + 1)
wmin2 = wgrid[line2(wgrid) <= S8_L + 2 * eS8].min()
tt2 = lambda wb, sS, sW: abs(S8_L - line2(wb)) / np.hypot(sS, slope2(wb) * sW)
w32 = {nm: wgrid[::-1][np.argmax([tt2(w, sS, sW) >= 3 for w in wgrid[::-1]])] for nm, sS, sW in cases}
b11 = np.interp(-1.11, pts2[::-1, 3], pts2[::-1, 0]); S8_11 = np.interp(b11, pts2[:, 0], pts2[:, 2])
b128 = np.interp(-1.28, pts2[::-1, 3], pts2[::-1, 0]) if pts2[:, 3].min() < -1.28 else np.nan
print("   two fluids: " + "; ".join(f"beta {b:.2f}: wbar {w:.4f}, S8 {s_:.4f}, w(1) {w1:.3f}" for b, w, s_, w1 in pts2))
ok(f"SW7 two fluids: S8 = {q2[2]:.4f} {q2[1]:+.3f} (wbar + 1) {q2[0]:+.2f} (wbar + 1)^2; with S8 = 0.818 +- 0.013 wbar >= {wmin2:.3f} (2 sigma); "
   "3 sigma depths (independent errors): " + "; ".join(f"{nm}: {w:.3f}" for nm, w in w32.items())
   + f"; the CPL depth w(z = 1) = -1.11 needs beta = {b11:.3f} and gives S8 = {S8_11:.3f}", abs(q2[2] - S8_LCDM) < 2e-3 and q2[1] < 0)

# SW8 (paper7 reviews r17-r18) the forecast line from the joint fit itself: S8(beta) of the two-fluid fit to DESI DR2 + Planck + Pantheon+
#     without the lensing term (results/h3_s8_fit3.npy of de_h3_s8_fit_check, parameters profiled at each beta), mapped on wbar(beta) of
#     the same model (model2). This replaces the Planck-normalised line for thresholds; the CPL depth is taken at w(z = 1) = -1.11.
Hf = np.load(os.path.join(HERE, 'results', 'h3_s8_fit3.npy'))
fr = Hf[Hf[:, 1] == 0]; fr = fr[np.argsort(fr[:, 0])]
def wbar_fit(beta, Om, fb, ai=1e-3):
    """w(z) of the fit's own model (de_h3_s8_fit_check: early normalisation Om, H0 = 1), dark energy inferred against Om a^-3 matter"""
    rci = 3 * Om * (1 - fb) * ai**-3; rbi = 3 * Om * fb * ai**-3
    def run(V0):
        def f(N, y):
            ph, pd = y; x = np.exp(-3 * (N - np.log(ai))); rc = rci * x * np.exp(beta * ph)
            h = np.sqrt((rc + rbi * x + 0.5 * pd**2 + V0) / 3); return [pd / h, (-3 * h * pd - beta * rc) / h]
        return _ivp(f, [np.log(ai), 0.0], [0.0, 0.0], rtol=1e-10, atol=1e-13, dense_output=True)
    def H0of(V0):
        ph, pd = run(V0).sol(0.0); return np.sqrt((rci * ai**3 * np.exp(beta * ph) + rbi * ai**3 + 0.5 * pd**2 + V0) / 3)
    V0 = _bq(lambda V: H0of(V) - 1.0, 0.0, 6.0, xtol=1e-12)
    N = np.linspace(np.log(ai), 0, 4001); ph, pd = run(V0).sol(N); x = np.exp(-3 * (N - np.log(ai)))
    h2 = (rci * x * np.exp(beta * ph) + rbi * x + 0.5 * pd**2 + V0) / 3
    rde = 3 * h2 - 3 * Om * np.exp(-3 * N)                          # against Om a^-3 matter (H0 = 1)
    w = -1 - np.gradient(np.log(np.abs(rde)), N) / 3
    at = lambda z: w[np.argmin(abs(N + np.log(1 + z)))]
    return at(0.0), at(0.5), at(1.0)
wb_of = {b: wbar_fit(b, om, ob / (om * hh**2)) for b, om, hh, ob in zip(fr[:, 0], fr[:, 4], fr[:, 5], fr[:, 6])}
wbar_f = np.array([(wb_of[b][0] + 2 * wb_of[b][1] + wb_of[b][2]) / 4 for b in fr[:, 0]]); w1_f = np.array([wb_of[b][2] for b in fr[:, 0]])
qf = np.polyfit(wbar_f + 1, fr[:, 3], 2); linef = lambda wb: np.polyval(qf, wb + 1); slopef = lambda wb: np.polyval(np.polyder(qf), wb + 1)
ttf = lambda wb, sS, sW: abs(S8_L - linef(wb)) / np.hypot(sS, slopef(wb) * sW)
wg = np.linspace(-1.4, -1.0, 40001)
w33 = {}
for nm, sS, sW in cases:
    hit = [w for w in wg[::-1] if linef(w) > S8_L and ttf(w, sS, sW) >= 3]
    w33[nm] = hit[0] if hit else np.nan
bcpl = np.interp(-1.11, w1_f[::-1], fr[::-1, 0]); S8cpl = np.interp(bcpl, fr[:, 0], fr[:, 3])
ok(f"SW8 line from the joint fit (no lensing term): S8 = {qf[2]:.4f} {qf[1]:+.3f} (wbar + 1) {qf[0]:+.2f} (wbar + 1)^2; 3 sigma depths with S8 = 0.818: "
   + "; ".join(f"{nm}: {w:.3f}" for nm, w in w33.items()) + f"; at wbar = -1.03 the tension is {ttf(-1.03, 0.006, 0.015):.1f} sigma; "
   f"the CPL depth w(z = 1) = -1.11 (beta = {bcpl:.3f}) gives S8 = {S8cpl:.3f}", qf[1] < 0)
# SW9 F3: change of Omega_c/Omega_b since a = 1e-3 for two fluids (phi sourced by dark dust only), exp(beta Delta phi)
def dphi2(beta, Om=0.3, fb=0.157, ai=1e-3):
    rci = 3 * Om * (1 - fb) * ai**-3; rbi = 3 * Om * fb * ai**-3
    def run(V0):
        def f(N, y):
            ph, pd = y; x = np.exp(-3 * (N - np.log(ai))); rc = rci * x * np.exp(beta * ph)
            h = np.sqrt((rc + rbi * x + 0.5 * pd**2 + V0) / 3); return [pd / h, (-3 * h * pd - beta * rc) / h]
        return _ivp(f, [np.log(ai), 0.0], [0.0, 0.0], rtol=1e-10, atol=1e-13).y[:, -1]
    def H0of(V0):
        ph, pd = run(V0); return np.sqrt((rci * ai**3 * np.exp(beta * ph) + rbi * ai**3 + 0.5 * pd**2 + V0) / 3)
    V0 = _bq(lambda V: H0of(V) - 1.0, 0.0, 6.0, xtol=1e-12)
    return run(V0)[0]
chg = {b: abs(np.exp(b * dphi2(b)) - 1) for b in (0.036, 0.054, 0.062)}
ok("SW9 two fluids: |change of Omega_c/Omega_b| since a = 1e-3: " + ", ".join(f"beta {b}: {100 * v:.1f}%" for b, v in chg.items()), chg[0.036] < 0.016)
# SW10 (paper7 review r20) the phi-matter epoch with uncoupled baryons: kinetic fraction of phi at z = 30 in the two-fluid model against
#     2 beta^2/3 (all matter coupled, AT2) and 2 beta^2 (1 - f_b)^2/3
def omphi_z(beta, fb, z=30.0, Om=0.3, ai=1e-3):
    rci = 3 * Om * (1 - fb) * ai**-3; rbi = 3 * Om * fb * ai**-3
    def f(N, y):
        ph, pd = y; x = np.exp(-3 * (N - np.log(ai))); rc = rci * x * np.exp(beta * ph)
        h = np.sqrt((rc + rbi * x + 0.5 * pd**2) / 3); return [pd / h, (-3 * h * pd - beta * rc) / h]
    Nz = -np.log(1 + z); y = _ivp(f, [np.log(ai), Nz], [0.0, 0.0], rtol=1e-10, atol=1e-13).y[:, -1]
    x = np.exp(-3 * (Nz - np.log(ai))); rc = rci * x * np.exp(beta * y[0])
    return 0.5 * y[1]**2 / (rc + rbi * x + 0.5 * y[1]**2)
b = 0.062
o_all, o_two = omphi_z(b, 0.0), omphi_z(b, 0.157)
ok(f"SW10 Omega_phi at z = 30, beta = {b}: all matter coupled {o_all:.5f} (2 beta^2/3 = {2 * b**2 / 3:.5f}); baryons uncoupled {o_two:.5f} "
   f"(2 beta^2 (1 - f_b)^2/3 = {2 * b**2 * 0.843**2 / 3:.5f})", abs(o_two / (2 * b**2 * 0.843**2 / 3) - 1) < 0.1 and abs(o_all / (2 * b**2 / 3) - 1) < 0.1)
