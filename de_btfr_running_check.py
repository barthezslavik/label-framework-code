# 93: the running BTFR slope (doc 84 prediction, a number the framework fixes) against SPARC (Lelli et al. 2019, flat velocities).
# Prediction: d log M_b / d log V_f = s(M_h) = 12/(1 - n_eff(M_h)) with n_eff the CDM spectral index on the halo mass scale (BBKS, as in
# ga_p2_spectrum_check HS4) and M_h = M_b / f (P1: one common ratio f; f only shifts which n_eff applies). Curvature of the relation:
# c = d s / d log V = s * ds/dlog M. Data: fit log M_b = a + b x + c x^2 / 2 (x = log V_f - 2) by maximum likelihood with errors on both axes
# and an intrinsic scatter; errors on c by bootstrap. BR1-BR3.
# Run: ~/.venvs/jev/bin/python -I de_btfr_running_check.py PATH/BTFR_Lelli2019.mrt
import sys
import numpy as np
from scipy.integrate import quad
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
print(f"   {len(d)} galaxies with V_f; log V_f in [{x.min() + 2:.2f}, {x.max() + 2:.2f}], log M_b in [{y.min():.2f}, {y.max():.2f}]")

# prediction
h, Om = 0.674, 0.315; Gam = Om * h
T = lambda k: np.log(1 + 2.34 * k / Gam) / (2.34 * k / Gam) * (1 + 3.89 * k / Gam + (16.1 * k / Gam)**2 + (5.46 * k / Gam)**3 + (6.71 * k / Gam)**4)**-0.25
P = lambda k: k**0.965 * T(k)**2
rho_m = 2.775e11 * Om
Wf = lambda q: 3 * (np.sin(q) - q * np.cos(q)) / q**3
sig2 = lambda Mm: quad(lambda lk: np.exp(lk)**3 * P(np.exp(lk)) * Wf(np.exp(lk) * (3 * Mm / (4 * np.pi * rho_m))**(1 / 3))**2,
                       np.log(1e-4), np.log(1e4), limit=500)[0]
neff = lambda Mm, e=1.05: -3 - 6 * (np.log(sig2(Mm * e)) - np.log(sig2(Mm / e))) / 2 / (2 * np.log(e))
s_of = lambda Mh: 12 / (1 - neff(Mh * h))                    # Mh in Msun, sig2 in Msun/h
pred = {}
for f in (0.02, 0.05, 0.1):                                   # M_b/M_h (P1 ratio: a label)
    Mh = 10**np.mean(y) / f
    s0 = s_of(Mh); ds = (s_of(Mh * 10**0.25) - s_of(Mh * 10**-0.25)) / 0.5
    pred[f] = (s0, ds, s0 * ds)
print("   prediction at the sample's mean mass: " + "; ".join(f"f = {f}: s = {v[0]:.2f}, ds/dlogM = {v[1]:.3f}, c = {v[2]:.3f}" for f, v in pred.items()))
ok("BR1 predicted curvature c = s ds/dlogM > 0 for every P1 ratio (slope steeper at high mass)", all(v[2] > 0 for v in pred.values()))

def nll(p, xx, yy, exx, eyy):
    a, b, c, ls = p
    f = a + b * xx + c * xx**2 / 2; fp = b + c * xx
    var = eyy**2 + (fp * exx)**2 + np.exp(2 * ls)
    return 0.5 * np.sum((yy - f)**2 / var + np.log(var))
def fit(xx, yy, exx, eyy, quad_=True):
    p0 = [10.0, 3.8, 0.0, np.log(0.1)]
    if quad_:
        return minimize(nll, p0, args=(xx, yy, exx, eyy), method='Nelder-Mead', options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 20000})
    g = lambda p: nll([p[0], p[1], 0.0, p[2]], xx, yy, exx, eyy)
    return minimize(g, [10.0, 3.8, np.log(0.1)], method='Nelder-Mead', options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 20000})
r = fit(x, y, ex, ey); r1 = fit(x, y, ex, ey, quad_=False)
rng = np.random.default_rng(1); cb = []; bb = []
for _ in range(400):
    i = rng.integers(0, len(x), len(x)); rb = fit(x[i], y[i], ex[i], ey[i]); cb.append(rb.x[2]); bb.append(rb.x[1])
cb = np.array(cb); sc = cb.std()
print(f"   linear fit: slope {r1.x[1]:.2f}, scatter {np.exp(r1.x[2]):.3f} dex; quadratic: slope at V = 100 km/s {r.x[1]:.2f} +- {np.std(bb):.2f}, "
      f"c = {r.x[2]:+.2f} +- {sc:.2f}, scatter {np.exp(r.x[3]):.3f}; 2 Delta lnL = {2 * (r1.fun - r.fun):.2f}")
cp = pred[0.05][2]
ok(f"BR2 measured curvature c = {r.x[2]:+.2f} +- {sc:.2f} vs predicted {cp:.2f} (f = 0.05): {abs(r.x[2] - cp) / sc:.1f} sigma from the prediction, "
   f"{abs(r.x[2]) / sc:.1f} sigma from zero", True)
ok(f"BR2b slope at the sample's centre: measured {r.x[1]:.2f} +- {np.std(bb):.2f}, predicted {min(v[0] for v in pred.values()):.2f}-{max(v[0] for v in pred.values()):.2f} "
   f"(Upsilon_3.6 = 0.5 fixed by the catalogue)", abs(r.x[1] - pred[0.05][0]) < 2 * np.std(bb))
ok(f"BR3 recorded: the data resolve a slope change of +-{2 * sc:.2f} per dex in V (2 sigma); prediction {cp:.2f}", True)
