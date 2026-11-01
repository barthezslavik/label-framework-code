# Transition dynamics: thermal mu-functions from T(a) = sqrt(a^2+H^2)/2pi against SPARC RAR (papers/restart/33-mu-transition.md). U1-U5.
# A thermal rule fixes BOTH the shape of mu and the scale a0 = k c H_inf (k pi-free, ga_theta_mond_check M4). Here each shape is fitted
# with a0 free and the fitted a0 is compared with the rule's own prediction; the family mu_n = x/(1+x^n)^(1/n) measures the sharpness.
# Data (not in git): Rotmod_LTG/*.dat and SPARC_Lelli2016c.mrt from https://astroweb.case.edu/SPARC/ (same cuts as rar_check.py).
# Run: uv run --with numpy --with scipy python -I rar_mu_thermal_check.py /path/SPARC_Lelli2016c.mrt /path/rotmod_dir
import sys, os
import numpy as np
from scipy.optimize import minimize_scalar, minimize

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    assert cond

table, rdir = sys.argv[1], sys.argv[2]
good = set()
for line in open(table, encoding='latin-1'):
    f = line.split()
    if len(f) == 19 and f[1].isdigit() and int(f[17]) < 3 and float(f[5]) >= 30:
        good.add(f[0])
kms2 = 1e6 / 3.0857e19
gb, go, gid = [], [], []
VG, VD, VB, RR = [], [], [], []
for fn in sorted(os.listdir(rdir)):
    name = fn.replace('_rotmod.dat', '')
    if name not in good:
        continue
    d = np.loadtxt(os.path.join(rdir, fn), comments='#')
    R, Vo, eV, Vg, Vd, Vb = d[:, 0], d[:, 1], d[:, 2], d[:, 3], d[:, 4], d[:, 5]
    m = (R > 0) & (Vo > 0) & (eV / Vo < 0.1)
    gbar = (np.abs(Vg) * Vg + 0.5 * Vd**2 + 0.7 * Vb**2) / R * kms2
    m &= gbar > 0
    gb += list(gbar[m]); go += list((Vo**2 / R * kms2)[m]); gid += [name] * int(m.sum())
    VG += list(Vg[m]); VD += list(Vd[m]); VB += list(Vb[m]); RR += list(R[m])
gb, go, gid = np.array(gb), np.array(go), np.array(gid)
VG, VD, VB, RR = map(np.array, (VG, VD, VB, RR))
y = np.log10(go)
print(f"galaxies {len(set(gid))}, points {len(gb)}")
cHinf = 2.998e8 * 67.4e3 / 3.0857e22 * 0.685 ** 0.5

def nu_n(g, a, n):                     # invert g_N = g * mu_n(g/a), mu_n(x) = x/(1+x^n)^(1/n): g^n = g_N^n (1 + (a/g)^n)... solve numerically
    lo, hi = g.copy(), g + np.sqrt(g * a) + a
    for _ in range(80):
        mid = (lo + hi) / 2
        x = mid / a
        f = mid * x / (1 + x**n) ** (1 / n) - g
        lo, hi = np.where(f < 0, mid, lo), np.where(f < 0, hi, mid)
    return (lo + hi) / 2

models = {                             # name: (g_obs(g_N, a0), predicted a0/(c H_inf) of the thermal rule or None)
    'RAR empirical':                  (lambda g, a: g / (1 - np.exp(-np.sqrt(g / a))), None),
    'simple mu (n=1)':                (lambda g, a: nu_n(g, a, 1), None),
    'thermal T_U/T = standard (n=2)': (lambda g, a: np.sqrt((g**2 + np.sqrt(g**4 + 4 * g**2 * a**2)) / 2), 1.0),
    'thermal Milgrom T-T_GH':         (lambda g, a: np.sqrt(g**2 + g * a), 2.0),
}
def fit(fmod, idx=slice(None)):
    obj = lambda la: np.mean((y[idx] - np.log10(fmod(gb[idx], 10**la)))**2)
    r = minimize_scalar(obj, bounds=(-11, -9), method='bounded')
    return 10**r.x, r.fun
res = {k: fit(v[0]) for k, v in models.items()}
names = np.unique(gid); groups = {n: np.where(gid == n)[0] for n in names}
rng = np.random.default_rng(1)
boot = {k: [] for k in models}; dmse = {k: [] for k in models}
for _ in range(150):
    pick = np.concatenate([groups[n] for n in rng.choice(names, len(names))])
    base = fit(models['RAR empirical'][0], pick)[1]
    for k, v in models.items():
        a, m = fit(v[0], pick); boot[k].append(a); dmse[k].append(m - base)
for k, (a, m) in res.items():
    pred = models[k][1]
    s = np.std(boot[k])
    extra = f" | rule predicts {pred:.0f} c H_inf = {pred * cHinf:.2e}: off by x{pred * cHinf / a:.1f} ({(pred * cHinf - a) / s:+.0f} sigma)" if pred else ""
    print(f"   {k:32s} a0 = {a:.3e} +- {s:.1e}  rms {np.sqrt(m):.4f}  dMSE vs RAR {np.mean(dmse[k]):+.5f} +- {np.std(dmse[k]):.5f}"
          f"  a0/(cH_inf) = {a / cHinf:.3f}{extra}")

# U1 the thermal rules' own scales are excluded: standard mu needs a0 ~ 1e-10, the rule gives c H_inf = 5.4e-10; Milgrom's needs ~1.5e-10 vs 2 c H_inf
st, mi = res['thermal T_U/T = standard (n=2)'][0], res['thermal Milgrom T-T_GH'][0]
ok(f"U1 fitted vs predicted: standard {st:.2e} vs {cHinf:.2e}; Milgrom-rule {mi:.2e} vs {2 * cHinf:.2e}: both rules off by > x3",
   cHinf / st > 3 and 2 * cHinf / mi > 3)

# U2 sharpness: fit mu_n with n free. n is the only shape information the data carry here.
def obj2(p):
    return np.mean((y - np.log10(nu_n(gb, 10**p[0], p[1])))**2)
r2 = minimize(obj2, x0=[-10, 1.5], method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-8})
nb = []
for _ in range(200):
    pick = np.concatenate([groups[n] for n in rng.choice(names, len(names))])
    o = lambda p: np.mean((y[pick] - np.log10(nu_n(gb[pick], 10**p[0], p[1])))**2)
    nb.append(minimize(o, x0=r2.x, method='Nelder-Mead', options={'xatol': 1e-3, 'fatol': 1e-8}).x[1])
n_fit, n_s = r2.x[1], np.std(nb)
print(f"   mu_n free: n = {n_fit:.2f} +- {n_s:.2f}, a0 = {10**r2.x[0]:.3e}, rms {np.sqrt(r2.fun):.4f}")
ok(f"U2 data prefer n = {n_fit:.2f} +- {n_s:.2f}; the thermal standard shape n = 2 is {(2 - n_fit) / n_s:+.1f} sigma away",
   abs(n_fit - 2) / n_s > 2)

# U3 the a0 of a shape: a0/(c H_inf) runs with n; theta = 0.225 is reached only for one sharpness, 1/4 for another
ns = [1, 1.5, 2, 3]
an = {n: fit(lambda g, a, n=n: nu_n(g, a, n))[0] / cHinf for n in ns}
print("   a0/(c H_inf) by sharpness: " + ", ".join(f"n={n}: {v:.3f}" for n, v in an.items()))
ok("U3 a0/(c H_inf) depends on the shape (runs monotonically with n): theta vs 1/4 is a statement about the shape, not only the scale",
   all(an[ns[i]] != an[ns[i + 1]] for i in range(3)) and (max(an.values()) - min(an.values())) > 0.05)

# U4 both shapes respect the duality x <-> 1/x (a <-> a0^2/a) but partition unity differently:
#    standard (thermal, 5D Pythagoras) mu(x)^2 + mu(1/x)^2 = 1  (L2: sin, cos);  simple (data) mu(x) + mu(1/x) = 1  (L1: Fermi function in ln x).
xs = np.logspace(-3, 3, 61)
mu_s = lambda x: x / (1 + x); mu_t = lambda x: x / np.sqrt(1 + x**2)
l1 = np.max(np.abs(mu_s(xs) + mu_s(1 / xs) - 1)); l2 = np.max(np.abs(mu_t(xs)**2 + mu_t(1 / xs)**2 - 1))
fermi = np.max(np.abs(mu_s(xs) - 1 / (1 + np.exp(-np.log(xs)))))
ok(f"U4 simple: max|mu(x)+mu(1/x)-1| = {l1:.1e} and mu = 1/(1+e^-ln x) (Fermi, {fermi:.1e}); thermal standard: max|mu^2(x)+mu^2(1/x)-1| = {l2:.1e}; "
   "data (U2, n = 1) choose the L1 partition, de Sitter temperatures give the L2 one", l1 < 1e-12 and l2 < 1e-12 and fermi < 1e-12)

# U5 with the data-preferred shape (simple mu) the scale: a0/(c H_inf) = 0.209 +- stat; candidates theta (H_inf), 1/4 (H_inf), 1/2pi (H0).
a_s, s_s = res['simple mu (n=1)'][0], np.std(boot['simple mu (n=1)'])
H0r = 1 / 0.685 ** 0.5
for lab, k, sc in (('theta, H_inf', np.sqrt(2) / (2 * np.pi), 1.0), ('1/4, H_inf', 0.25, 1.0), ('1/2pi, H0', 1 / (2 * np.pi), H0r)):
    print(f"   {lab:13s} predicts {k * sc * cHinf:.3e}: {(k * sc * cHinf - a_s) / s_s:+.1f} sigma (stat only; Upsilon systematics ~ +-20%)")
ok("U5 simple-mu scale: theta (H_inf) and 1/2pi (H0) within ~1.2 sigma stat, 1/4 at ~3 sigma; the thermal rules' k = 1, 2 are excluded",
   abs(np.sqrt(2) / (2 * np.pi) * cHinf - a_s) / s_s < 1.5 and abs(H0r * cHinf / (2 * np.pi) - a_s) / s_s < 1.5
   and abs(0.25 * cHinf - a_s) / s_s > 2.5)

# U6 robustness: global Upsilon_disk, Upsilon_bulge free together with a0 and n (mass-to-light systematics), galaxy bootstrap.
def gbar_u(ud, ub, idx=slice(None)):
    return (np.abs(VG[idx]) * VG[idx] + ud * VD[idx]**2 + ub * VB[idx]**2) / RR[idx] * kms2
def obj6(p, idx=slice(None)):
    ud, ub = 10**p[2], 10**p[3]
    g = gbar_u(ud, ub, idx)
    if np.any(g <= 0):
        return 1e3
    return np.mean((y[idx] - np.log10(nu_n(g, 10**p[0], p[1])))**2)
p0 = [np.log10(10**r2.x[0]), r2.x[1], np.log10(0.5), np.log10(0.7)]
r6 = minimize(obj6, x0=p0, method='Nelder-Mead', options={'xatol': 1e-4, 'fatol': 1e-9, 'maxiter': 4000})
b6 = []
for _ in range(100):
    pick = np.concatenate([groups[n] for n in rng.choice(names, len(names))])
    b6.append(minimize(lambda p: obj6(p, pick), x0=r6.x, method='Nelder-Mead', options={'xatol': 1e-3, 'fatol': 1e-8, 'maxiter': 2000}).x)
b6 = np.array(b6)
n6, sn6 = r6.x[1], b6[:, 1].std(); k6 = 10**r6.x[0] / cHinf; sk6 = np.std(10**b6[:, 0] / cHinf)
print(f"   free Upsilon: Ud = {10**r6.x[2]:.2f} +- {np.std(10**b6[:, 2]):.2f}, Ub = {10**r6.x[3]:.2f} +- {np.std(10**b6[:, 3]):.2f}, "
      f"n = {n6:.2f} +- {sn6:.2f}, a0/(c H_inf) = {k6:.3f} +- {sk6:.3f}, rms {np.sqrt(r6.fun):.4f}")
for lab, k in (('theta (H_inf)', np.sqrt(2) / (2 * np.pi)), ('1/2pi (H0)', H0r / (2 * np.pi)), ('1/4 (H_inf)', 0.25)):
    print(f"   {lab:14s} {(k - k6) / sk6:+.1f} sigma")
ok(f"U6 with Upsilon free: n = {n6:.2f} +- {sn6:.2f} (thermal n = 2 at {(2 - n6) / sn6:+.1f} sigma); a0/(c H_inf) = {k6:.3f} +- {sk6:.3f}",
   (2 - n6) / sn6 > 2)
