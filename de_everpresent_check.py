# 91b: the discrete-label Lambda (doc 91, DL3) as a stochastic history, fitted to DESI DR2 BAO + compressed Planck + SN with the doc-70
# pipeline, plus the BBN bound. Model (Sorkin-type, derived in DL1-DL3 under axiom D): Lambda sqrt(V4) = A xi with Lambda = A W(V4)/V4,
# W a Brownian motion in the causal-past 4-volume V4 (one random +-1 per element), A = 8 pi alpha (l_P/l)^2 the single amplitude label.
# Units: H in 100 km/s/Mpc, c = 1; H^2 = om_r a^-4 + om_m a^-3 + lam, lam = Lambda/3. Each seed is one realization; (A, om_m) are fitted per
# seed, om_b fixed at 0.02237. BBN: |Omega_Lambda| < 0.04 at T = 1 MeV (a = 2.35e-10) is the imported bound Delta N_eff < 0.3.
# Conditioning a realization on its value today (Brownian bridge in V) adds a constant to W/V, i.e. a constant Lambda: the data-consistent
# version is therefore 'Lambda label lamc + discrete fluctuation of amplitude A'.
# Run: ~/.venvs/jev/bin/python -I de_everpresent_check.py DATA_ROOT [NSEEDS]
import os, sys, time
import numpy as np
from scipy.optimize import minimize
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'de_desi_fit_check.py')).read().split('# V1 validation')[0]
ns = {'__name__': 'pipeline'}; exec(src, ns)
ok = ns['ok']; orad = ns['orad']
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 120
OB = 0.02237
LNA = np.linspace(np.log(1e-11), 0.0, 3000); DL = LNA[1] - LNA[0]
A_BBN = 2.35e-10

COLL = []   # scale factors at which histories collapse (H^2 < 0)
def history(A, om, g, lamc=0.0):
    """integrate in ln a with explicit midpoint steps; returns (lnE table, h, Omega_L(BBN)) or None if H^2 < 0"""
    a = np.exp(LNA)
    t = 0.0; eta = 0.0; M = np.zeros(4); W = 0.0; Vprev = 0.0; lam = 0.0
    H = np.empty(len(a)); OLb = None
    for i, aa in enumerate(a):
        H2 = orad / aa**4 + om / aa**3 + lam
        if H2 <= 0: COLL.append(aa); return None
        Hh = np.sqrt(H2); H[i] = Hh
        if i == 0:
            t = 1 / (2 * Hh); eta = 1 / (aa * Hh)                  # radiation-era start
            V = 8 * np.pi / 105 * t**4
            W = g[0] * np.sqrt(V); Vprev = V
        else:
            dt = DL / Hh; deta = dt / aa
            t += dt; eta += deta
            M += aa**4 * eta**np.arange(4) * deta
            V = Vstart + 4 * np.pi / 3 * (eta**3 * M[0] - 3 * eta**2 * M[1] + 3 * eta * M[2] - M[3])
            dV = max(V - Vprev, 0.0); W += g[i] * np.sqrt(dV); Vprev = V
        if i == 0: Vstart = V
        lam = A * W / V / 3 + lamc
        if OLb is None and aa >= A_BBN: OLb = lam / H2
    h = H[-1]
    return np.log(H / h), h, OLb

current = {}
def Ez_t(z, Om, h, w0, wa):
    lnap = -np.log1p(np.asarray(z, float))
    tab = current['lnE']
    out = np.where(lnap >= LNA[0], np.interp(lnap, LNA, tab), tab[0] + 2 * (LNA[0] - lnap))
    return np.exp(out) if np.ndim(z) else float(np.exp(out))
ns['Ez'] = Ez_t

def chi2(A, om, lamc, g, combo='P+'):
    r = history(A, om, g, lamc)
    if r is None: return 1e9, None
    current['lnE'], h, OLb = r
    p = (om / h**2, h, OB, -1.0, 0.0)
    if not (0.5 < h < 0.9 and 0.1 < p[0] < 0.6): return 1e9, None
    return ns['bao_chi2'](p) + ns['cmb_chi2'](p) + ns['sn_chi2'](p, combo), (h, p[0], OLb)

# EV1 survival without a Lambda label (pure discrete Lambda): fraction of histories from a = 1e-11 that reach today without H^2 < 0
rng_g = lambda s: np.random.default_rng(s).standard_normal(len(LNA))
surv = {A: np.mean([history(A, 0.143, rng_g(s)) is not None for s in range(60)]) for A in (0.7, 0.3, 0.2, 0.1, 0.05)}
ok("EV1 survival to today without a Lambda label: " + ", ".join(f"A = {A}: {f:.2f}" for A, f in surv.items())
   + "; today's Lambda needs A |xi0| = Lambda sqrt(V4) = 0.73 (DL5)", surv[0.7] < 0.05 and surv[0.05] > 0.9)

# EV1b the histories at today's amplitude A = 0.7 all collapse, and they do so in the radiation era, before nucleosynthesis
COLL.clear(); _ = [history(0.7, 0.143, rng_g(s)) for s in range(60)]
ok(f"EV1b A = 0.7: {len(COLL)} of 60 histories collapse, at a = {min(COLL):.1e}..{max(COLL):.1e}, all before nucleosynthesis (a = {A_BBN:.2e})",
   len(COLL) == 60 and max(COLL) < A_BBN)

# EV2 BBN and the ratio argument (independent of A and of the process): Omega_L(radiation) = 2.725 A xi_b (DL4), today A xi_0 = 0.732;
#     |Omega_L(BBN)| < 0.04 needs |xi_b/xi_0| < 0.04/(2.725*0.732) = 0.020; for independent normal draws P = (2/pi) arctan(0.020).
#     With a label the bound reads A < 0.04/2.725 at 1 sigma.
r_ = 0.04 / (2.725 * 0.732); Pr = 2 / np.pi * np.arctan(r_)
ok(f"EV2 pure discrete Lambda: P(BBN and today) <= {Pr:.4f} even before recollapse; with a Lambda label the fluctuation amplitude A < {0.04 / 2.725:.3f}", Pr < 0.02)

# EV3 DESI test of the data-consistent version (label lamc + fluctuation A): per seed fit (lamc, om); Delta chi2 vs LCDM (1421.22).
#     A = 0.015 (the BBN bound) and A = 0.1 (ignoring BBN, to see what a large fluctuation could do).
CL = 1421.22
def fit_seed(A, s):
    g = rng_g(s)
    f = lambda x: chi2(A, x[1], x[0], g)[0]
    r = minimize(f, [0.207, 0.1425], method='Nelder-Mead', options={'xatol': 1e-5, 'fatol': 2e-2, 'maxiter': 200})
    return r.fun
t0 = time.time(); c0 = fit_seed(0.0, 0); print(f"   one seed fit: {time.time() - t0:.1f} s", flush=True)
ok(f"EV0 validation: A = 0 (constant Lambda, om_b fixed, 3000-step table) gives chi2 = {c0:.2f} vs the LCDM fit 1421.22 (om_b free); "
   "Delta chi2 below is taken against this A = 0 value (same numerics)", abs(c0 - CL) < 2.0)
CL = c0
out = {}
for A in (0.015, 0.1):
    d = np.array([fit_seed(A, s) - CL for s in range(NS)])
    out[A] = d
    print(f"   A = {A}: Delta chi2 vs LCDM over {NS} seeds: min {d.min():+.2f}, 10% {np.percentile(d, 10):+.2f}, median {np.median(d):+.2f}; "
          f"fraction with Delta chi2 < -5: {(d < -5).mean():.3f}", flush=True)
np.save(os.path.join(os.environ.get('EVERPRESENT_OUT', '.'), 'everpresent_dchi2.npy'), np.array([out[0.015], out[0.1]]))
ok(f"EV3 Delta chi2 (DESI+CMB+Pantheon+) of label + fluctuation vs LCDM: A = 0.015 median {np.median(out[0.015]):+.2f}, best {out[0.015].min():+.2f}; "
   f"A = 0.1 median {np.median(out[0.1]):+.2f}, best {out[0.1].min():+.2f} (CPL gains 7.6)", True)

# EV4 the marginal likelihood over realizations (Bayes factor vs LCDM at A = 0.015) and, for the three best seeds, the shape of the DE
#     density (rho_L(z)/rho_L(0)), Omega_L before recombination, and the growth (sigma8, S8 ratios at the same amplitude at a = 1e-3).
from scipy.integrate import solve_ivp
BF = float(np.mean(np.exp(-out[0.015] / 2)))
def diag(A, s):
    g = rng_g(s) if s is not None else np.zeros(len(LNA))
    r = minimize(lambda x: chi2(A, x[1], x[0], g)[0], [0.207, 0.1425], method='Nelder-Mead', options={'xatol': 1e-5, 'fatol': 2e-2, 'maxiter': 200})
    lamc, om = r.x; lnE, h, _ = history(A, om, g, lamc)
    H = np.exp(lnE) * h; a = np.exp(LNA); lam = H**2 - orad / a**4 - om / a**3; Omm = om / a**3 / H**2
    dlnH = np.gradient(np.log(H), LNA); i0 = np.searchsorted(LNA, np.log(1e-3))
    sol = solve_ivp(lambda N, y: [y[1], -(2 + np.interp(N, LNA, dlnH)) * y[1] + 1.5 * np.interp(N, LNA, Omm) * y[0]], [LNA[i0], 0], [1e-3, 1e-3], rtol=1e-8)
    rl = lambda z: np.interp(-np.log1p(z), LNA, lam) / lam[-1]
    return sol.y[0, -1], om / h**2, h, rl(1.0), rl(2.0), np.interp(-np.log1p(1e4), LNA, lam / H**2)
D0, Om0, h0 = diag(0.0, None)[:3]
best = np.argsort(out[0.015])[:3]
for s in best:
    D, Om_, h_, r1, r2, ol4 = diag(0.015, int(s))
    print(f"   seed {s}: Delta chi2 {out[0.015][s]:+.2f}, h = {h_:.3f}, rho_L(z=1, 2)/rho_L(0) = {r1:.3f}, {r2:.3f}, Omega_L(z=1e4) = {ol4:+.3f}, "
          f"sigma8 x{D / D0:.3f}, S8 x{D / D0 * np.sqrt(Om_ / Om0):.3f}", flush=True)
ok(f"EV4 Bayes factor <exp(-Delta chi2/2)> over {NS} realizations at A = 0.015: {BF:.1f} (carried by the best 1-3 seeds; noisy)", True)
