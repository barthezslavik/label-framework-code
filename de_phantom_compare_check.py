# 97: the second native phantom mechanism (doc 91: Lambda label + discrete fluctuation, A = 0.015 at the BBN bound) against the first
# (dust coupling m(phi), docs 82, 86). 300 realizations with (lamc, om_m, om_b) fitted to DESI DR2 BAO + compressed Planck + Pantheon+; for
# each: Delta chi2 vs constant Lambda (same numerics, om_b free), S8 ratio at the same amplitude at a = 1e-3, the mean w over 0 < z < 1 and
# the split of the gain between BAO, CMB and SN. Distinguishing signature: in m(phi) the S8 rise is tied to the phantom depth; in the
# discrete fluctuation it is not. PC1-PC5.
# Run: ~/.venvs/jev/bin/python -I de_phantom_compare_check.py DATA_ROOT [NSEEDS] [NPROC]
import os, sys
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
HERE = os.path.dirname(os.path.abspath(__file__))
_argv = sys.argv[:]; sys.argv = [sys.argv[0], sys.argv[1], '0']
F = os.path.join(HERE, 'de_everpresent_check.py')
ev = {'__name__': 'ev', '__file__': F}; exec(open(F).read().split('# EV1')[0], ev)
sys.argv = _argv
ns, history, LNA, orad, ok = ev['ns'], ev['history'], ev['LNA'], ev['orad'], ev['ok']
A = 0.015
S8_LCDM, S8_L, eS8 = 0.831, 0.818, 0.013                          # Planck-LCDM S8 (doc 86e) and the lensing value

def parts(lamc, om, ob, g, A_=A):
    r = history(A_, om, g, lamc)
    if r is None: return None
    ev['current']['lnE'], h, _ = r
    p = (om / h**2, h, ob, -1.0, 0.0)
    if not (0.5 < h < 0.9 and 0.1 < p[0] < 0.6 and 0.019 < ob < 0.026): return None
    return ns['bao_chi2'](p), ns['cmb_chi2'](p), ns['sn_chi2'](p, 'P+'), h

def total(x, g, A_):
    q = parts(x[0], x[1], x[2], g, A_)
    return 1e9 if q is None else q[0] + q[1] + q[2]

def growth_and_shape(lamc, om, ob, g, A_):
    lnE, h, _ = history(A_, om, g, lamc)
    H = np.exp(lnE) * h; a = np.exp(LNA); lam = H**2 - orad / a**4 - om / a**3; Omm = om / a**3 / H**2
    dlnH = np.gradient(np.log(H), LNA); i0 = np.searchsorted(LNA, np.log(1e-3))
    sol = solve_ivp(lambda N, y: [y[1], -(2 + np.interp(N, LNA, dlnH)) * y[1] + 1.5 * np.interp(N, LNA, Omm) * y[0]], [LNA[i0], 0], [1e-3, 1e-3], rtol=1e-8)
    rl = lambda z: np.interp(-np.log1p(z), LNA, lam) / lam[-1]
    wbar = -1 + np.log(rl(1.0)) / (3 * np.log(2.0))               # mean w over ln a, 0 < z < 1: rho(z=1)/rho(0) = 2^(3(1+wbar))
    # (sign corrected 2026-11-01; phantom_compare_300.txt of 2026-10-22 was written with -1 - ln(.)/(3 ln 2), i.e. its column wbar is
    #  -2 - wbar; the corrected PC2/PC3 numbers are in ga_wbar_definition_check WD2-WD3)
    return sol.y[0, -1], om / h**2, h, wbar, [rl(z) for z in (0.5, 1.0, 1.5, 2.0)]

def fit(g, A_):
    best = None
    for x0 in ([0.207, 0.1425, 0.02237], [0.215, 0.1415, 0.0224]):
        r = minimize(total, x0, args=(g, A_), method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-2, 'maxiter': 400})
        if best is None or r.fun < best.fun: best = r
    return best

def worker(s):
    g = np.random.default_rng(s).standard_normal(len(LNA))
    r = fit(g, A)
    if r.fun > 1e8: return (s, np.nan,) + (np.nan,) * 11
    b, c, sn, h = parts(*r.x, g)
    D, Om, h, wbar, shp = growth_and_shape(*r.x, g, A)
    return (s, r.fun, b, c, sn, D, Om, h, wbar, *shp)

if __name__ == '__main__':
    from multiprocessing import Pool
    NS = int(sys.argv[2]) if len(sys.argv) > 2 else 300; NP = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    g0 = np.zeros(len(LNA)); r0 = fit(g0, 0.0)
    b0, c0, sn0, _ = parts(*r0.x, g0, 0.0); D0, Om0, h0, _, _ = growth_and_shape(*r0.x, g0, 0.0)
    print(f"   constant Lambda (om_b free): chi2 = {r0.fun:.2f} (BAO {b0:.2f}, CMB {c0:.2f}, SN {sn0:.2f}), Om = {Om0:.4f}, h = {h0:.4f}", flush=True)
    ok("PC0 constant Lambda with om_b free reproduces the LCDM chi2 of doc 70 within the 3000-step numerics (< 1.6)", abs(r0.fun - 1421.22) < 1.6)
    with Pool(NP) as pool:
        res = np.array(pool.map(worker, range(NS), chunksize=4))
    np.save(os.path.join(os.environ.get('EVERPRESENT_OUT', '.'), 'phantom_compare.npy'), res)
    good = np.isfinite(res[:, 1])
    s_, chi, b, c, sn, D, Om, h, wbar = [res[good, i] for i in range(9)]
    dchi = chi - r0.fun; S8 = S8_LCDM * D / D0 * np.sqrt(Om / Om0)
    BF = np.mean(np.exp(-np.clip(dchi, -60, None) / 2))
    print(f"   {good.sum()}/{NS} realizations fitted; Delta chi2: min {dchi.min():+.2f}, 5% {np.percentile(dchi, 5):+.2f}, 10% {np.percentile(dchi, 10):+.2f}, "
          f"median {np.median(dchi):+.2f}; fraction < -5: {(dchi < -5).mean():.3f}; Bayes factor {BF:.2f}", flush=True)
    ok(f"PC1 {good.sum()} realizations: fraction with Delta chi2 < -5 = {(dchi < -5).mean():.3f}, Bayes factor <exp(-Delta chi2/2)> = {BF:.2f}", good.sum() >= 0.95 * NS)
    top = dchi < -5
    print(f"   gain split for Delta chi2 < -5 (mean): BAO {np.mean(b[top] - b0):+.2f}, CMB {np.mean(c[top] - c0):+.2f}, SN {np.mean(sn[top] - sn0):+.2f}; "
          f"mean wbar(0<z<1) {np.mean(wbar[top]):.3f} (all: {np.mean(wbar):.3f}); h {np.mean(h[top]):.3f}", flush=True)
    ok(f"PC2 where the gain comes from (Delta chi2 < -5): BAO {np.mean(b[top] - b0):+.2f}, CMB {np.mean(c[top] - c0):+.2f}, SN {np.mean(sn[top] - sn0):+.2f}; "
       f"phantom-like past (wbar < -1) in {np.mean(wbar[top] < -1):.2f} of them", True)
    cc = np.corrcoef(wbar, S8)[0, 1]; slope = np.polyfit(wbar, S8, 1)[0]
    ok(f"PC3 discrete fluctuation: S8 vs wbar correlation {cc:+.2f}, slope dS8/dwbar = {slope:+.3f}; S8 of the good fits: mean {np.mean(S8[top]):.3f}, "
       f"range {S8[top].min() if top.any() else np.nan:.3f}-{S8[top].max() if top.any() else np.nan:.3f}", True)
    # m(phi): S8 tied to the phantom depth (ga_growth_coupling_check model, same S8 calibration as 86e: S8_LCDM x growth ratio)
    G = os.path.join(HERE, 'ga_growth_coupling_check.py')
    gc = {'__name__': 'gc'}; exec(open(G).read().split('D0 = model(0.0)')[0], gc)
    Dg0 = gc['model'](0.0)[0]
    mp = []
    for beta in (0.01, 0.02, 0.03, 0.04, 0.05, 0.06):
        Dg, w0, w05, w1 = gc['model'](beta)
        mp.append((beta, (w0 + 2 * w05 + w1) / 4, S8_LCDM * Dg / Dg0))
    mp = np.array(mp); slope_m = np.polyfit(mp[:, 1], mp[:, 2], 1)[0]
    print("   m(phi): " + "; ".join(f"beta {bb:.2f}: wbar {ww:.3f}, S8 {ss:.3f}" for bb, ww, ss in mp), flush=True)
    ok(f"PC4 m(phi): dS8/dwbar = {slope_m:+.3f} (S8 rises with phantom depth); discrete fluctuation: {slope:+.3f}", slope_m < 0)
    both = top & (S8 < S8_L + 2 * eS8)
    ok(f"PC5 realizations with Delta chi2 < -5 AND S8 within 2 sigma of lensing ({S8_L} +- {eS8}): {both.mean():.3f} of all "
       f"({both.sum()}/{good.sum()}); m(phi) at its best fit beta = 0.03 has S8 = {np.interp(0.03, mp[:, 0], mp[:, 2]):.3f} and gain 2.7 (86c)", True)
