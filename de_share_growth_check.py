# 114: the collapse-epoch premises seen today. Framework picture: at the collapse epoch the share eps_c = M_b/M_v and R_c = v_flat/v_v are
# common (that gives s = 12/(1 - n), doc 84); the halo stays isothermal outside r_v (M ~ r), so today v200 = v_v (R constant, as the HSC
# relation shows in the disc regime, doc 110) and M200/M_v = r200/r_v = sqrt(rho_vir(z_c)/rho_200), i.e. the present-day share
# chi_200 = chi_c M_v/M200 rises with mass because massive haloes collapsed later. Predict d log chi_200/d log M_b (no fit; nu = 1, 1.5;
# M_h from M_b with the HSC relation's mean share) and compare with the lensing shares (HSC disc regime; DES; DESI; SDSS). SG1-SG3.
# Run: ~/.venvs/jev/bin/python -I de_share_growth_check.py DATA_ROOT SPARC_DIR
import os, sys, contextlib, io
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
hs = {'__name__': 'hs', '__file__': os.path.join(HERE, 'de_hsc_shmr_check.py')}
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(hs['__file__']).read().split('rng = np.random.default_rng(17)')[0], hs)
vv = {'__name__': 'vv', '__file__': os.path.join(HERE, 'de_vflat_v200_check.py')}
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(vv['__file__']).read().split('# VR5')[0], vv)
kc = vv['kc']; ok = vv['ok']
base = hs['base']                                                   # HSC: columns lMb, R, chi, v200
m = base[:, 0] <= 10.85
s_hsc = np.polyfit(base[m, 0], np.log10(base[m, 2]), 1)[0]
Om = 0.315
def chi_pred_slope(lMb, nu, eps=0.1):
    # halo mass from M_b with a common share eps (the trend DOES depend on it: it sets the halo-mass window)
    Mh = 10**lMb / (eps * 0.157)
    zc = np.array([kc['gv'](M, nu)[0] for M in Mh])
    rho_vir = 18 * np.pi**2 * Om * (1 + zc)**3; rho_200 = 200 * Om * (1 + 0.1)**3          # both relative to rho_m0 (M200m at z ~ 0.1)
    ratio = np.sqrt(rho_vir / rho_200)                                                    # M200/M_v = r200/r_v for M ~ r
    return np.polyfit(lMb, np.log10(1 / ratio), 1)[0], ratio
lMb_grid = np.linspace(9.7, 10.8, 5)
p1, r1 = chi_pred_slope(lMb_grid, 1.0); p15, r15 = chi_pred_slope(lMb_grid, 1.5)
ok(f"SG1 predicted d log chi_200/d log M_b = {p1:+.2f} (nu 1), {p15:+.2f} (nu 1.5); M200/M_v = {r1.min():.2f}-{r1.max():.2f} (nu 1)", p1 > 0 and p15 > 0)
lMb = vv['lMb']; R = vv['R']; names = np.array([o[0] for o in vv['out']])
Mh_all = np.array([10**s_[2] for s_ in vv['samples']]); chi = 10**lMb / (0.157 * Mh_all)
obs = {'HSC disc regime': s_hsc}
for nm in ('DES', 'DESI', 'SDSS-blue'):
    mm = names == nm; obs[nm] = np.polyfit(lMb[mm], np.log10(chi[mm]), 1)[0]
ok("SG2 measured d log chi_200/d log M_b: " + ", ".join(f"{k} {v:+.2f}" for k, v in obs.items()), True)
ok(f"SG3 the BTFR slope with R constant today is 3/(1 - d log chi/d log M_b): predicted {3 / (1 - p1):.2f} (nu 1), {3 / (1 - p15):.2f} (nu 1.5); "
   f"with the HSC trend {3 / (1 - s_hsc):.2f}; catalogue slope 3.63-3.85", True)

# SG4 (review 2026-10-23) dependence on the assumed share (halo-mass window) and the implied local slope 3/(1 - trend); the slope
#     prediction of the paper (3.68-3.77) refers to 1e10-1e12 Msun haloes
for eps in (0.02, 0.05, 0.1, 0.3):
    a1 = chi_pred_slope(lMb_grid, 1.0, eps)[0]; a2 = chi_pred_slope(lMb_grid, 1.5, eps)[0]
    lo, hi = np.log10(10**lMb_grid.min() / (eps * 0.157)), np.log10(10**lMb_grid.max() / (eps * 0.157))
    print(f"   eps (retained share) {eps}: halo window 10^{lo:.1f}-10^{hi:.1f}; trend {a1:+.2f} (nu 1) / {a2:+.2f} (nu 1.5); local slope {3 / (1 - a1):.2f} / {3 / (1 - a2):.2f}")
ok("SG4 recorded: the predicted share trend and the implied local slope depend on the halo-mass window set by the share", True)

# SG5 (Codex review 2026-10-23) restrict the prediction to haloes where (P3) applies: the collapse has happened before observation
#     (nu sigma(M) > delta_c, so z_c > 0 exists; kc['gv'] otherwise returns z_c = 0) and r200 >= r_v (an outward continuation)
def chi_pred_valid(lMb, nu, eps):
    Mh = 10**lMb / (eps * 0.157)
    ok_c = np.array([nu * kc['sigM'](M) > 1.686 for M in Mh])
    zc = np.array([kc['gv'](M, nu)[0] for M in Mh])
    ratio = np.sqrt(18 * np.pi**2 * Om * (1 + zc)**3 / (200 * Om * (1 + 0.1)**3))
    v = ok_c & (ratio >= 1)
    if v.sum() < 3: return None, v.sum()
    return np.polyfit(lMb[v], np.log10(1 / ratio[v]), 1)[0], v.sum()
grid = np.linspace(9.7, 10.8, 12)
res = []
for eps in (0.02, 0.05, 0.1, 0.3):
    for nu in (1.0, 1.5):
        t, n = chi_pred_valid(grid, nu, eps)
        res.append((eps, nu, t, n))
        print(f"   eps {eps}, nu {nu}: valid points {n}/12; trend {('%+.2f' % t) if t is not None else 'n/a'}; local slope {('%.2f' % (3 / (1 - t))) if t is not None else 'n/a'}")
tv = [r[2] for r in res if r[2] is not None]
ok(f"SG5 restricted to z_c > 0 and r200 >= r_v: trend {min(tv):+.2f} to {max(tv):+.2f}, local slope {3 / (1 - min(tv)):.2f}-{3 / (1 - max(tv)):.2f}", len(tv) > 0)

# SG6 (Codex round 2) compare on one mass range: keep only the cases valid over the whole grid (log M_b 9.7-10.8); report the excluded one
full = [r for r in res if r[3] == 12]
part = [r for r in res if r[3] < 12]
ok(f"SG6 common range log M_b 9.7-10.8: trend {min(r[2] for r in full):+.2f} to {max(r[2] for r in full):+.2f}, local slope "
   f"{3 / (1 - min(r[2] for r in full)):.2f}-{3 / (1 - max(r[2] for r in full)):.2f}; excluded (partial coverage): "
   + ", ".join(f"eps {r[0]}, nu {r[1]} ({r[3]}/12 points)" for r in part), len(full) > 0)

# SG7 (Codex round 3) LCDM virial overdensity (Bryan & Norman 1998, equation borrowed): Delta_c = 18 pi^2 + 82 x - 39 x^2 relative to the
#     critical density at z_c, x = Omega_m(z_c) - 1; rho_vir/rho_m0 = Delta_c E(z_c)^2/Omega_m. Same restriction and common range as SG5/SG6.
def chi_pred_bn(lMb, nu, eps):
    Mh = 10**lMb / (eps * 0.157)
    ok_c = np.array([nu * kc['sigM'](M) > 1.686 for M in Mh])
    zc = np.array([kc['gv'](M, nu)[0] for M in Mh])
    E2 = Om * (1 + zc)**3 + 1 - Om; Omz = Om * (1 + zc)**3 / E2; xx = Omz - 1
    Dc = 18 * np.pi**2 + 82 * xx - 39 * xx**2
    rho_vir = Dc * E2                                                 # rho_vir / rho_crit0
    rho_200 = 200 * Om * (1 + 0.1)**3                                 # M200m at z = 0.1, in rho_crit0 units
    ratio = np.sqrt(rho_vir / rho_200)
    v = ok_c & (ratio >= 1)
    if v.sum() < 12: return None, v.sum()
    return np.polyfit(lMb[v], np.log10(1 / ratio[v]), 1)[0], v.sum()
bn = []
for eps in (0.02, 0.05, 0.1, 0.3):
    for nu in (1.0, 1.5):
        t, n = chi_pred_bn(grid, nu, eps)
        if t is not None: bn.append(t)
ok(f"SG7 LCDM virial overdensity, common range: trend {min(bn):+.2f} to {max(bn):+.2f}, local slope {3 / (1 - min(bn)):.2f}-{3 / (1 - max(bn)):.2f} "
   f"(EdS: SG6)", len(bn) > 0)

# SG8 (Codex round 5) the LOCAL slope at the point of the measured one (v_flat = 100 km/s, log M_b = 9.71 for the BTFR fit), by a
#     central difference of log chi_200 in log M_b for the LCDM-overdensity cases valid on the whole range (SG7)
def chi_bn_point(lMb, nu, eps):
    Mh = 10**lMb / (eps * 0.157)
    zc = kc['gv'](Mh, nu)[0]
    E2 = Om * (1 + zc)**3 + 1 - Om; xx = Om * (1 + zc)**3 / E2 - 1
    Dc = 18 * np.pi**2 + 82 * xx - 39 * xx**2
    return np.log10(1 / np.sqrt(Dc * E2 / (200 * Om * 1.1**3)))
loc = []
for eps in (0.02, 0.05, 0.1, 0.3):
    for nu in (1.0, 1.5):
        t, n = chi_pred_bn(grid, nu, eps)
        if t is None: continue
        h_ = 0.05; d = (chi_bn_point(9.71 + h_, nu, eps) - chi_bn_point(9.71 - h_, nu, eps)) / (2 * h_)
        loc.append(3 / (1 - d))
ok(f"SG8 local slope at log M_b = 9.71 (v_flat = 100 km/s), LCDM overdensity, {len(loc)} valid cases: {min(loc):.2f}-{max(loc):.2f}", len(loc) > 0)

# SG9 (Codex round 10) continuous scan of the stated intervals: nu = 1.0-1.5 (step 0.05), effective fraction 0.02-0.3 (15 log-spaced values);
#     cases valid over the whole range (collapsed and r200 >= r_v at all 12 grid masses), EdS and LCDM overdensities; local slope at 9.71
def zc_ratio(lMb, nu, eps, lcdm):
    Mh = 10**lMb / (eps * 0.157)
    zc = kc['gv'](Mh, nu)[0]
    ok_c = (nu * kc['sigM'](Mh) > 1.686) and zc >= 0.1              # collapsed before the observation epoch z = 0.1 (Codex round 11)
    if lcdm:
        E2 = Om * (1 + zc)**3 + 1 - Om; xx = Om * (1 + zc)**3 / E2 - 1
        rv = (18 * np.pi**2 + 82 * xx - 39 * xx**2) * E2
    else:
        rv = 18 * np.pi**2 * Om * (1 + zc)**3
    return ok_c, np.sqrt(rv / (200 * Om * 1.1**3))
out_scan = {False: [], True: []}
for lcdm in (False, True):
    for nu in np.arange(1.0, 1.501, 0.05):
        for eps in np.geomspace(0.02, 0.3, 15):
            vals = [zc_ratio(x, nu, eps, lcdm) for x in grid]
            if not all(c and r >= 1 for c, r in vals): continue
            tr = np.polyfit(grid, [np.log10(1 / r) for c, r in vals], 1)[0]
            h_ = 0.05; d = (np.log10(1 / zc_ratio(9.71 + h_, nu, eps, lcdm)[1]) - np.log10(1 / zc_ratio(9.71 - h_, nu, eps, lcdm)[1])) / (2 * h_)
            out_scan[lcdm].append((nu, eps, tr, 3 / (1 - d)))
for lcdm, rows in out_scan.items():
    tr = [r[2] for r in rows]; sl = [r[3] for r in rows]
    print(f"   {'LCDM' if lcdm else 'EdS '}: {len(rows)} valid cases; trend {min(tr):+.3f} to {max(tr):+.3f}; local slope at 100 km/s {min(sl):.2f}-{max(sl):.2f}")
ok("SG9 continuous scan of nu = 1-1.5 and effective fractions 0.02-0.3 recorded (bounds printed above)", all(len(v) > 0 for v in out_scan.values()))
