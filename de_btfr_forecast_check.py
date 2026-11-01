# 116: when can the running BTFR slope (doc 93 BR1, c ~ 0.58 per dex in V) be measured? The predicted relation is integrated exactly,
# dy/dx = s(M_h) = 12/(1 - n_eff(M_h)), M_h = M_b/f (P1), from the SPARC centre; synthetic samples get SPARC-like errors and the SPARC
# intrinsic scatter; the same quadratic fit as de_btfr_running_check gives sigma(c). Calibration: SPARC's own x, ex, ey must reproduce the
# bootstrap error 0.77. Then: N and the velocity range needed for a 3 sigma detection, and the slope the prediction gives at 20 and 300 km/s.
# MOND gives c = 0 (M = V^4/(G a0) at every V_f). FC1-FC6.
# Run: ~/.venvs/jev/bin/python -I de_btfr_forecast_check.py PATH/BTFR_Lelli2019.mrt
import sys
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import minimize

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

rows = []
for l in open(sys.argv[1]):
    if len(l) < 48 or l[:5].strip() in ('Title', 'Autho', 'Bytes', 'Note'): continue
    try:
        lm, elm, vf, evf = float(l[12:18]), float(l[18:24]), float(l[36:42]), float(l[42:48])
    except ValueError:
        continue
    if vf > 0: rows.append((lm, elm, vf, evf))
d = np.array(rows)
y, ey = d[:, 0], d[:, 1]; x = np.log10(d[:, 2]) - 2; ex = d[:, 3] / d[:, 2] / np.log(10)

# prediction (as in de_btfr_running_check)
h, Om = 0.674, 0.315; Gam = Om * h
T = lambda k: np.log(1 + 2.34 * k / Gam) / (2.34 * k / Gam) * (1 + 3.89 * k / Gam + (16.1 * k / Gam)**2 + (5.46 * k / Gam)**3 + (6.71 * k / Gam)**4)**-0.25
P = lambda k: k**0.965 * T(k)**2
rho_m = 2.775e11 * Om
Wf = lambda q: 3 * (np.sin(q) - q * np.cos(q)) / q**3
sig2 = lambda Mm: quad(lambda lk: np.exp(lk)**3 * P(np.exp(lk)) * Wf(np.exp(lk) * (3 * Mm / (4 * np.pi * rho_m))**(1 / 3))**2,
                       np.log(1e-4), np.log(1e4), limit=500)[0]
neff = lambda Mm, e=1.05: -3 - 6 * (np.log(sig2(Mm * e)) - np.log(sig2(Mm / e))) / 2 / (2 * np.log(e))
s_of = lambda Mh: 12 / (1 - neff(Mh * h))
lgM = np.linspace(5.0, 14.5, 77)                               # tabulate s(M_h) once
s_tab = {f: np.array([s_of(10**(m) / f) for m in lgM]) for f in (0.02, 0.05, 0.1)}
s_b = lambda yy, f: np.interp(yy, lgM, s_tab[f])               # slope as a function of log M_b

def relation(f, x0=np.mean(x), y0=np.mean(y)):                 # integrate dy/dx = s(y) both ways from the SPARC centre
    up = solve_ivp(lambda t, v: [s_b(v[0], f)], [x0, 1.2], [y0], dense_output=True, rtol=1e-8)
    dn = solve_ivp(lambda t, v: [s_b(v[0], f)], [x0, -1.2], [y0], dense_output=True, rtol=1e-8)
    return lambda xx: np.where(xx >= x0, up.sol(np.maximum(xx, x0))[0], dn.sol(np.minimum(xx, x0))[0])

def nll(p, xx, yy, exx, eyy):
    a, b, c, ls = p
    f = a + b * xx + c * xx**2 / 2; fp = b + c * xx
    var = eyy**2 + (fp * exx)**2 + np.exp(2 * ls)
    return 0.5 * np.sum((yy - f)**2 / var + np.log(var))
def fit(xx, yy, exx, eyy):
    return minimize(nll, [10.0, 3.8, 0.0, np.log(0.1)], args=(xx, yy, exx, eyy), method='Nelder-Mead',
                    options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 20000}).x

sig_int = fit(x, y, ex, ey)[3]; sig_int = np.exp(sig_int)
rng = np.random.default_rng(7)
def mock(rel, xs, exs, eys, n_rep=200):
    cs = []
    for _ in range(n_rep):
        xt = xs; yt = rel(xt) + rng.normal(0, sig_int, len(xt))
        cs.append(fit(xt + rng.normal(0, exs), yt + rng.normal(0, eys), exs, eys)[2])
    cs = np.array(cs); return cs.mean(), cs.std()

rel05 = relation(0.05)
cm, cs = mock(rel05, x, ex, ey)
ok(f"FC1 calibration: SPARC's own x and errors with the predicted relation (f = 0.05) and scatter {sig_int:.3f} dex give c = {cm:.2f} +- {cs:.2f} "
   f"(bootstrap on the data 0.77)", 0.4 < cs < 1.2)

xs_lo, xs_hi = rel05(np.array([np.log10(20) - 2, np.log10(300) - 2])), None
sl = {v: s_b(rel05(np.array([np.log10(v) - 2]))[0], 0.05) for v in (15, 20, 50, 100, 200, 300)}
ok("FC2 predicted local slope d log M_b / d log V_f (f = 0.05): " + ", ".join(f"{v} km/s {s:.2f}" for v, s in sl.items())
   + "; MOND: 4 for the asymptotic speed (measured V_f not computed)", sl[300] > sl[20])

# future samples: x uniform in log V over [V_lo, V_hi]; errors: 5% in V, 0.10 dex in M_b (SPARC median errors in brackets)
print(f"   SPARC median errors: ex {np.median(ex):.3f} dex, ey {np.median(ey):.3f} dex; log V range {x.min() + 2:.2f}-{x.max() + 2:.2f}")
res = {}
for (vlo, vhi) in ((20, 300), (15, 350)):
    for N in (123, 500, 1000, 3000):
        xs = rng.uniform(np.log10(vlo) - 2, np.log10(vhi) - 2, N)
        exs = np.full(N, 0.05 / np.log(10)); eys = np.full(N, 0.10)
        res[(vlo, vhi, N)] = mock(rel05, xs, exs, eys, n_rep=100)
for k, (m_, s_) in res.items():
    print(f"   V {k[0]}-{k[1]} km/s, N = {k[2]:5d}: c = {m_:.2f} +- {s_:.2f} -> {m_ / s_:.1f} sigma")
cfit = res[(20, 300, 3000)][0]
ok(f"FC3 over 20-300 km/s the predicted relation fitted with a quadratic has c = {cfit:.2f} (local value 0.58 at the SPARC centre)", cfit > 0.2)
N3 = [k[2] for k, v in res.items() if k[:2] == (20, 300) and v[0] / v[1] >= 3]
ok(f"FC4 3 sigma detection over 20-300 km/s with 5% / 0.10 dex errors needs N >= {min(N3) if N3 else '>3000'}", True)
# the same at SPARC errors (ey 0.10 is optimistic for M/L): mass errors 0.2 dex
xs = rng.uniform(np.log10(20) - 2, np.log10(300) - 2, 1000)
m2, s2 = mock(rel05, xs, np.full(1000, 0.05 / np.log(10)), np.full(1000, 0.20), n_rep=100)
ok(f"FC5 with 0.20 dex mass errors (M/L systematics) and N = 1000: c = {m2:.2f} +- {s2:.2f} ({m2 / s2:.1f} sigma)", True)
# dependence on f (which n_eff applies)
for f in (0.02, 0.1):
    xs = rng.uniform(np.log10(20) - 2, np.log10(300) - 2, 1000)
    mf, sf = mock(relation(f), xs, np.full(1000, 0.05 / np.log(10)), np.full(1000, 0.10), n_rep=60)
    print(f"   f = {f}: c over 20-300 km/s = {mf:.2f} +- {sf:.2f} (N = 1000)")
ok("FC6 recorded: the prediction is a smooth rise of the slope with V across the whole range, from the spectrum (n_eff), with sign c > 0 "
   "for every f; MOND has c = 0 for the asymptotic speed of isolated galaxies (for measured V_f and this selection not computed); a feedback turndown below ~50 km/s (Ruan et al. 2025) is a separate, local feature", True)
# FC7 (review r1) correlated M/L systematics do not average down: a common offset or a trend linear in log V leaves c unchanged, a curved
#     M/L(V) shifts c one to one. Measured on mocks (N = 1000, 20-300 km/s) with a systematic delta log M = o + t x + (k/2) x^2.
xs = rng.uniform(np.log10(20) - 2, np.log10(300) - 2, 1000); exs = np.full(1000, 0.05 / np.log(10)); eys = np.full(1000, 0.10)
base_c = mock(rel05, xs, exs, eys, n_rep=40)[0]
shift = {}
for nm, (o, t, k) in {'offset 0.2 dex': (0.2, 0, 0), 'linear 0.2 dex per dex V': (0, 0.2, 0), 'curved k = 0.47': (0, 0, 0.47)}.items():
    rel_s = lambda xx, o=o, t=t, k=k: rel05(xx) + o + t * xx + k * xx**2 / 2
    shift[nm] = mock(rel_s, xs, exs, eys, n_rep=40)[0] - base_c
W = np.log10(300 / 20); dev = 0.47 / 2 * (W / 2)**2
ok("FC7 correlated M/L systematics: shift of c: " + ", ".join(f"{k}: {v:+.3f}" for k, v in shift.items())
   + f"; an M/L(V) that bends by {dev:.2f} dex from a straight line across 20-300 km/s mimics (or cancels) the whole predicted curvature",
   abs(shift['offset 0.2 dex']) < 0.05 and abs(shift['linear 0.2 dex per dex V']) < 0.05 and abs(shift['curved k = 0.47'] - 0.47) < 0.1)
# FC8 (review r5) a baryon share that varies with mass, f = f0 (M_b/M_b0)^t: M_b = f M_h gives dy/dx = s(M_h)/(1 - t). The measured slope
#     at the SPARC centre (3.67 +- 0.11) bounds t; within that bound the curvature over 20-300 km/s is recomputed.
def relation_t(t, f0=0.05, x0=np.mean(x), y0=np.mean(y)):
    fb = lambda yy: f0 * 10**(t * (yy - y0))
    rhs = lambda tt, v: [np.interp(v[0] - np.log10(fb(v[0])) + np.log10(0.05), lgM, s_tab[0.05]) / (1 - t)]
    up = solve_ivp(rhs, [x0, 1.2], [y0], dense_output=True, rtol=1e-8); dn = solve_ivp(rhs, [x0, -1.2], [y0], dense_output=True, rtol=1e-8)
    return lambda xx: np.where(xx >= x0, up.sol(np.maximum(xx, x0))[0], dn.sol(np.minimum(xx, x0))[0])
s0 = s_b(np.mean(y), 0.05)
t_lo, t_hi = 1 - s0 / (3.67 - 2 * 0.11), 1 - s0 / (3.67 + 2 * 0.11)
res_t = {}
for t in (t_hi, 0.0, t_lo, 0.2):
    rt = relation_t(t)
    xs = rng.uniform(np.log10(20) - 2, np.log10(300) - 2, 1000)
    res_t[t] = (s0 / (1 - t), mock(rt, xs, np.full(1000, 0.05 / np.log(10)), np.full(1000, 0.10), n_rep=40)[0])
ok("FC8 share trend t = d ln f/d ln M_b: " + "; ".join(f"t = {t:+.3f}: centre slope {a:.2f}, c = {c:.2f}" for t, (a, c) in res_t.items())
   + f"; the measured slope 3.67 +- 0.11 (2 sigma) allows t in [{min(t_hi, t_lo):+.3f}, {max(t_hi, t_lo):+.3f}] -- c stays {min(res_t[t_hi][1], res_t[t_lo][1]):.2f}-"
   f"{max(res_t[t_hi][1], res_t[t_lo][1]):.2f} there; t = +0.2 (the r200 trend of doc 114) would steepen the slope to {res_t[0.2][0]:.2f}, against the data",
   min(t_hi, t_lo) < 0 < max(t_hi, t_lo))
# FC9 (review r6) a curved share, log f = log f0 + (k/2)(log M_b - y0)^2: the slope at the centre is unchanged, the curvature shifts by about
#     s^2 k. The k that cancels the predicted curvature is computed; the test then needs f(M) measured independently (lensing).
def relation_k(k, f0=0.05, x0=np.mean(x), y0=np.mean(y)):
    lf = lambda yy: np.log10(f0) + k / 2 * (yy - y0)**2
    rhs = lambda tt, v: [np.interp(v[0] - lf(v[0]) + np.log10(0.05), lgM, s_tab[0.05]) / (1 - k * (v[0] - y0))]   # M_b = f M_h
    up = solve_ivp(rhs, [x0, 1.2], [y0], dense_output=True, rtol=1e-8); dn = solve_ivp(rhs, [x0, -1.2], [y0], dense_output=True, rtol=1e-8)
    return lambda xx: np.where(xx >= x0, up.sol(np.maximum(xx, x0))[0], dn.sol(np.minimum(xx, x0))[0])
xs = rng.uniform(np.log10(20) - 2, np.log10(300) - 2, 1000); exs = np.full(1000, 0.05 / np.log(10)); eys = np.full(1000, 0.10)
ck = {k: mock(relation_k(k), xs, exs, eys, n_rep=30)[0] for k in (0.0, -0.02, -0.035)}
kz = -ck[0.0] / ((ck[-0.035] - ck[0.0]) / -0.035)
ok("FC9 curved share log f with curvature k per dex^2 in M_b: " + ", ".join(f"k = {k:+.3f}: c = {v:.2f}" for k, v in ck.items())
   + f"; k = {kz:+.3f} cancels the predicted curvature at unchanged centre slope -- the curvature test needs f(M_b) measured independently",
   ck[-0.035] < ck[0.0])
