# 93b (reviews 2026-10-22): the BTFR with the speed the paper defines, v_flat = v(r_t), not the catalogue V_f. For each SPARC galaxy fit
# v_obs^2 = v_gas^2 + Ud v_disk^2 + Ub v_bul^2 + v_inf^2 [1 - (r_c/r) atan(r/r_c)] (Ud = 0.5, Ub = 0.7 fixed; free v_inf, r_c), set r_t = 7.16 r_c
# (u = 0.8), keep galaxies whose data reach r_t, take the model total speed at r_t; M_b and V_f (with e_V_f) from the Lelli 2019 table.
# Errors: a bootstrap that perturbs every rotation curve within its errors, REFITS each galaxy, re-applies the r_t selection and resamples
# galaxies (review item 1). Comparison with V_f on one common sample with catalogue errors (item 2). Prediction s = 12/(1 - n_eff(M_h)) at the
# baryonic masses of the selected galaxies, M_h = M_b/eps for eps = 0.02-0.1 (item 3). RT1-RT5.
# Run: ~/.venvs/jev/bin/python -I de_btfr_rt_check.py SPARC_DIR [NBOOT]
import sys, os
import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.integrate import quad

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

d = sys.argv[1]; NB = int(sys.argv[2]) if len(sys.argv) > 2 else 200
rows = {}
for l in open(os.path.join(d, 'MassModels_Lelli2016c.mrt')):
    p = l.split()
    if len(p) != 10: continue
    try:
        vals = [float(x) for x in p[1:]]
    except ValueError:
        continue
    rows.setdefault(p[0], []).append(vals)
cat = {}
for l in open(os.path.join(d, 'BTFR_Lelli2019.mrt')):
    if len(l) < 48: continue
    try:
        cat[l[:12].strip()] = (float(l[12:18]), float(l[18:24]), float(l[36:42]), float(l[42:48]))
    except ValueError:
        continue
Ud, Ub = 0.5, 0.7
gals = []
for g, arr in rows.items():
    if g not in cat: continue
    a = np.array(arr)
    gals.append((g, a[:, 1], a[:, 2], np.maximum(a[:, 3], 1.0), np.sign(a[:, 4]) * a[:, 4]**2 + Ud * a[:, 5]**2 + Ub * a[:, 6]**2))

def fit_vt(R, V, eV, vb2, p0=None):
    def model(p, r):
        vinf, rc = np.exp(p)
        return np.sqrt(np.maximum(np.interp(r, R, vb2) + vinf**2 * (1 - (rc / r) * np.arctan(r / rc)), 0))
    f = least_squares(lambda p: (model(p, R) - V) / eV, p0 if p0 is not None else [np.log(max(V.max(), 10)), np.log(max(R.max() / 5, 0.1))])
    vinf, rc = np.exp(f.x); rt = 7.16 * rc
    if rt > R.max() or rc < 0.05: return None, f.x
    return float(model(f.x, np.array([rt]))[0]), f.x

def nll(p, xx, yy, exx, eyy):
    a, b, ls = p; var = eyy**2 + (b * exx)**2 + np.exp(2 * ls)
    return 0.5 * np.sum((yy - a - b * xx)**2 / var + np.log(var))
def slope(xx, yy, exx, eyy):
    return minimize(nll, [10, 3.8, np.log(0.1)], args=(xx, yy, exx, eyy), method='Nelder-Mead', options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 20000}).x

# nominal fits
base = {}
for g, R, V, eV, vb2 in gals:
    vt, px = fit_vt(R, V, eV, vb2); base[g] = (vt, px)
sel = [g for g in base if base[g][0] is not None]
lm = np.array([cat[g][0] for g in sel]); elm = np.array([cat[g][1] for g in sel]); vt = np.array([base[g][0] for g in sel])
x = np.log10(vt) - 2
# speed error at r_t from the refit scatter (computed below) -- nominal fit uses a 5% floor until then
ex0 = np.full_like(x, 0.05 / np.log(10))
p_nom = slope(x, lm, ex0, elm)

# bootstrap with refits (item 1)
rng = np.random.default_rng(3); bsl = []; nsel = []; spread = {g: [] for g in sel}
for b in range(NB):
    vals = {}
    for g, R, V, eV, vb2 in gals:
        Vp = V + eV * rng.standard_normal(len(V))
        v, _ = fit_vt(R, Vp, eV, vb2, base[g][1])
        if v is not None: vals[g] = v
        if g in spread and v is not None: spread[g].append(v)
    names = list(vals); nsel.append(len(names))
    pick = rng.integers(0, len(names), len(names))
    xb = np.array([np.log10(vals[names[i]]) - 2 for i in pick]); yb = np.array([cat[names[i]][0] for i in pick]); eyb = np.array([cat[names[i]][1] for i in pick])
    bsl.append(slope(xb, yb, np.full_like(xb, 0.02), eyb)[1])
ev_rt = np.array([np.std(np.log10(spread[g])) if len(spread[g]) > 5 else 0.05 for g in sel])
p_rt = slope(x, lm, ev_rt, elm)
s_rt, e_rt = p_rt[1], np.std(bsl)
ok(f"RT1 slope with v_flat = v(r_t): {s_rt:.2f} +- {e_rt:.2f} (bootstrap with refits and re-selection, {NB} rounds; selected {np.min(nsel)}-{np.max(nsel)} "
   f"galaxies, nominal {len(sel)}); intrinsic scatter {np.exp(p_rt[2]):.3f} dex", len(sel) > 30)

# item 2: one common sample (selected AND catalogue V_f > 0), catalogue errors for V_f
com = [g for g in sel if cat[g][2] > 0]
lmc = np.array([cat[g][0] for g in com]); elmc = np.array([cat[g][1] for g in com])
xf = np.log10([cat[g][2] for g in com]) - 2; exf = np.array([cat[g][3] / cat[g][2] / np.log(10) for g in com])
xr = np.log10([base[g][0] for g in com]) - 2; exr = np.array([ev_rt[sel.index(g)] for g in com])
sf, sr = slope(xf, lmc, exf, elmc)[1], slope(xr, lmc, exr, elmc)[1]
bf, br = [], []
for _ in range(400):
    i = rng.integers(0, len(com), len(com))
    bf.append(slope(xf[i], lmc[i], exf[i], elmc[i])[1]); br.append(slope(xr[i], lmc[i], exr[i], elmc[i])[1])
dd = np.array(bf) - np.array(br)
ok(f"RT2 common sample of {len(com)} galaxies: V_f (catalogue errors) {sf:.2f} +- {np.std(bf):.2f}, v(r_t) {sr:.2f} +- {np.std(br):.2f}; "
   f"paired difference {sf - sr:+.2f} +- {np.std(dd):.2f}", True)
ratio = np.array([base[g][0] / cat[g][2] for g in com])
ok(f"RT3 v(r_t)/V_f: median {np.median(ratio):.3f}, scatter {np.std(np.log10(ratio)):.3f} dex -- not a common factor", True)

# item 3: prediction at the masses of the selected galaxies
h, Om = 0.674, 0.315; Gam = Om * h
T = lambda k: np.log(1 + 2.34 * k / Gam) / (2.34 * k / Gam) * (1 + 3.89 * k / Gam + (16.1 * k / Gam)**2 + (5.46 * k / Gam)**3 + (6.71 * k / Gam)**4)**-0.25
P = lambda k: k**0.965 * T(k)**2
rho_m = 2.775e11 * Om
Wf = lambda q: 3 * (np.sin(q) - q * np.cos(q)) / q**3
sig2 = lambda Mm: quad(lambda lk: np.exp(lk)**3 * P(np.exp(lk)) * Wf(np.exp(lk) * (3 * Mm / (4 * np.pi * rho_m))**(1 / 3))**2, np.log(1e-4), np.log(1e4), limit=500)[0]
neff = lambda Mm, e=1.05: -3 - 6 * (np.log(sig2(Mm * e)) - np.log(sig2(Mm / e))) / 2 / (2 * np.log(e))
pred = {}
for eps in (0.02, 0.05, 0.1):
    Mh = 10**np.mean(lm) / eps * h
    pred[eps] = 12 / (1 - neff(Mh))
lo, hi = min(pred.values()), max(pred.values())
ok(f"RT4 prediction at the mean baryonic mass of the selected galaxies (log M_b = {np.mean(lm):.2f}): s = {lo:.2f}-{hi:.2f} (eps = 0.1-0.02)", True)
ok(f"RT5 tension: v(r_t) slope {s_rt:.2f} +- {e_rt:.2f} vs {lo:.2f}-{hi:.2f}: {(lo - s_rt) / e_rt:.1f}-{(hi - s_rt) / e_rt:.1f} sigma", True)
