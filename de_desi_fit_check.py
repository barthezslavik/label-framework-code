# 70: is Lambda constant? Our own fit of DESI DR2 BAO + compressed Planck CMB + SN (Pantheon+, DES-SN5YR Dovekie): LCDM vs w0waCDM (CPL).
# Data (untrusted, downloaded 2026-10-21 into data/external/, passed as paths):
#   DESI DR2 BAO mean/cov: CobayaSampler/bao_data desi_bao_dr2 (DESI DR2 II, arXiv:2503.14738);
#   Pantheon+SH0ES.dat and STAT+SYS cov (Scolnic et al. 2022, Brout et al. 2022): cosmology cut zHD > 0.01, M marginalised;
#   DES-SN5YR Dovekie HD and inverse STAT+SYS covariance (des-science/DES-SN5YR, upper-triangular npz);
#   Planck 2018 distance priors (R, l_A, omega_b), LCDM TT,TE,EE+lowE: Chen, Huang, Wang 2019, arXiv:1808.05724, Table 1.
# Approximations (stated): flat; radiation with N_eff = 3.046, massless neutrinos; z_* and z_d from Hu-Sugiyama / Eisenstein-Hu fits;
# r_s by the sound-speed integral; compressed CMB instead of the full likelihood. Expect qualitative, not exact, agreement with DESI.
# Run: python -I de_desi_fit_check.py DATA_ROOT     (DATA_ROOT = data/external)
import sys, os
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import chi2 as chi2d, norm

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

root = sys.argv[1]
c = 299792.458
Tcmb = 2.7255
og = 2.469e-5                                  # Omega_gamma h^2
orad = og * (1 + 0.2271 * 3.046)

# ---------- cosmology ----------
def Ez(z, Om, h, w0, wa):
    Or = orad / h**2
    Ode = 1 - Om - Or
    de = (1 + z)**(3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / (1 + z))
    return np.sqrt(Om * (1 + z)**3 + Or * (1 + z)**4 + Ode * de)

def DM(z, p):
    Om, h, ob, w0, wa = p
    return c / (100 * h) * quad(lambda q: 1 / Ez(q, Om, h, w0, wa), 0, z, limit=200)[0]

def rs(z, p):
    Om, h, ob, w0, wa = p
    Rb = lambda q: 31500 * ob * (Tcmb / 2.7)**-4 / (1 + q)
    f = lambda q: c / np.sqrt(3 * (1 + Rb(q))) / (100 * h * Ez(q, Om, h, w0, wa))
    return quad(f, z, np.inf, limit=400)[0]

def zstar(ob, om):
    g1 = 0.0783 * ob**-0.238 / (1 + 39.5 * ob**0.763); g2 = 0.560 / (1 + 21.1 * ob**1.81)
    return 1048 * (1 + 0.00124 * ob**-0.738) * (1 + g1 * om**g2)

def zdrag(ob, om):
    b1 = 0.313 * om**-0.419 * (1 + 0.607 * om**0.674); b2 = 0.238 * om**0.223
    return 1291 * om**0.251 / (1 + 0.659 * om**0.828) * (1 + b1 * ob**b2)

# ---------- data ----------
bd = os.path.join(root, 'desi_dr2_bao')
rows = [l.split() for l in open(os.path.join(bd, 'desi_gaussian_bao_ALL_GCcomb_mean.txt')) if not l.startswith('#')]
bz = np.array([float(r[0]) for r in rows]); bv = np.array([float(r[1]) for r in rows]); bq = [r[2] for r in rows]
bcov = np.loadtxt(os.path.join(bd, 'desi_gaussian_bao_ALL_GCcomb_cov.txt')); bicov = np.linalg.inv(bcov)

# Calibration (stated approximation): the Eisenstein-Hu z_d fit gives r_d = 150.6 Mpc at the Planck 2018 LCDM best fit, 2.4% above the CAMB
# value 147.05 Mpc (Planck 2018 VI, best fit) -- a known mismatch of the z_d definition. BAO alone is blind to it (r_d is degenerate with h),
# BAO + CMB is not. We rescale r_d by a constant so that the Planck best fit gives 147.05 Mpc.
def _rd_raw(Om, h, ob):
    p = (Om, h, ob, -1.0, 0.0)
    return rs(zdrag(ob, Om * h**2), p)
RD_CAL = _rd_raw(0.3153, 0.6736, 0.02237) / 147.05

def bao_chi2(p):
    Om, h, ob, w0, wa = p
    rd = rs(zdrag(ob, Om * h**2), p) / RD_CAL
    th = []
    for z, q in zip(bz, bq):
        dm = DM(z, p); dh = c / (100 * h * Ez(z, Om, h, w0, wa))
        th.append(dm / rd if q == 'DM_over_rs' else dh / rd if q == 'DH_over_rs' else (z * dm**2 * dh)**(1 / 3) / rd)
    d = np.array(th) - bv
    return float(d @ bicov @ d)

cm = np.array([1.7502, 301.471, 0.02236]); cs = np.array([0.0046, 0.0895, 0.00015])
# Calibration (stated approximation): our l_A at the Planck 2018 LCDM best fit (Om 0.3153, h 0.6736, ob 0.02237) is 0.25% high (+8.5 sigma),
# a convention offset of the theta computation behind the priors; we rescale r_s(z_*) by a constant so that the best fit gives the mean.
# Dark energy changes the late-time D_M, not r_s(z_*), so the factor does not bias the w0-wa comparison at first order.
def _lA_raw(Om, h, ob):
    p = (Om, h, ob, -1.0, 0.0); zs = zstar(ob, Om * h**2)
    return np.pi * DM(zs, p) / rs(zs, p)
RS_CAL = _lA_raw(0.3153, 0.6736, 0.02237) / 301.471
cc = np.array([[1, 0.46, -0.66], [0.46, 1, -0.33], [-0.66, -0.33, 1]])
cicov = np.linalg.inv(cc * np.outer(cs, cs))
def cmb_chi2(p):
    Om, h, ob, w0, wa = p
    zs = zstar(ob, Om * h**2)
    dmz = DM(zs, p)
    R = np.sqrt(Om) * 100 * h * dmz / c
    lA = np.pi * dmz / (rs(zs, p) * RS_CAL)
    d = np.array([R, lA, ob]) - cm
    return float(d @ cicov @ d)

def marg_chi2(res, icov):
    a = res @ icov @ res; b = np.sum(res @ icov); cc_ = np.sum(icov)
    return float(a - b**2 / cc_)

pp = np.genfromtxt(os.path.join(root, 'pantheonplus', 'PantheonPlusSH0ES.dat'), names=True, dtype=None, encoding=None)
pcov = np.loadtxt(os.path.join(root, 'pantheonplus', 'PantheonPlusSH0ES_STAT_SYS.cov'), skiprows=1).reshape(len(pp), len(pp))
pm = pp['zHD'] > 0.01
pz, pzh, pmb = pp['zHD'][pm], pp['zHEL'][pm], pp['m_b_corr'][pm]
picov = np.linalg.inv(pcov[np.ix_(pm, pm)])

snl = [l.split() for l in open(os.path.join(root, 'des_sn5yr_dovekie', 'DES-Dovekie_HD.csv')) if l.startswith('SN:')]   # SNANA FITRES format
dd = {'zHD': np.array([float(r[3]) for r in snl]), 'zHEL': np.array([float(r[4]) for r in snl]), 'MU': np.array([float(r[5]) for r in snl])}
npz = np.load(os.path.join(root, 'des_sn5yr_dovekie', 'STAT+SYS.npz'))
nn = int(npz[npz.files[0]][0]); assert nn == len(snl), (nn, len(snl))
dic = np.zeros((nn, nn)); dic[np.triu_indices(nn)] = npz[npz.files[1]]
il = np.tril_indices(nn, -1); dic[il] = dic.T[il]
dm_ = dd['zHD'] > 0
dz, dzh, dmu = dd['zHD'][dm_], dd['zHEL'][dm_], dd['MU'][dm_]
dicov = np.linalg.inv(np.linalg.inv(dic)[np.ix_(dm_, dm_)])

def dm_grid(p, zs):
    zg = np.concatenate([[0], np.geomspace(1e-3, max(zs.max(), 2.5), 400)])
    Om, h, ob, w0, wa = p
    ig = 1 / Ez(zg, Om, h, w0, wa)
    cum = np.concatenate([[0], np.cumsum(np.diff(zg) * (ig[1:] + ig[:-1]) / 2)]) * c / (100 * h)
    return np.interp(zs, zg, cum)

def sn_chi2(p, which):
    if which == 'P+':
        mu = 5 * np.log10((1 + pzh) * dm_grid(p, pz)) + 25
        return marg_chi2(pmb - mu, picov)
    mu = 5 * np.log10((1 + dzh) * dm_grid(p, dz)) + 25
    return marg_chi2(dmu - mu, dicov)

# ---------- fits ----------
def fit(fun, lcdm, starts):
    best = None
    for s in starts:
        x0 = s[:3] if lcdm else s
        def f(x):
            p = (x[0], x[1], x[2], -1.0, 0.0) if lcdm else tuple(x)
            if not (0.1 < p[0] < 0.6 and 0.5 < p[1] < 0.9 and 0.015 < p[2] < 0.03 and -3 < p[3] < 1 and -5 < p[4] < 3 and p[3] + p[4] < 0):
                return 1e9
            return fun(p)
        r = minimize(f, x0, method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-4, 'maxiter': 6000, 'maxfev': 9000})
        if best is None or r.fun < best.fun:
            best = r
    return best

starts = [np.array([0.31, 0.68, 0.0224, -0.8, -0.6]), np.array([0.30, 0.67, 0.0223, -0.5, -1.5]), np.array([0.32, 0.67, 0.0224, -1.0, 0.0]),
          np.array([0.35, 0.64, 0.0223, -0.42, -1.75]), np.array([0.31, 0.68, 0.0224, -0.838, -0.62]), np.array([0.32, 0.67, 0.0224, -0.752, -0.86])]   # incl. DESI optima

# V1 validation: DESI BAO alone, flat LCDM (r_d free in DESI's analysis; here we report Omega_m with omega_b fixed by the CMB prior value)
def bao_only(p):
    return bao_chi2(p) + ((p[2] - 0.02236) / 0.00015)**2
r0 = fit(bao_only, True, starts)
Om0, h0, ob0 = r0.x
rd0 = rs(zdrag(ob0, Om0 * h0**2), (Om0, h0, ob0, -1, 0)) / RD_CAL
print(f"   calibration factors: r_s(z_*) {RS_CAL:.5f}, r_d {RD_CAL:.5f}")
print(f"   BAO-only LCDM: Omega_m = {Om0:.4f} (DESI 0.2975 +- 0.0086), h r_d = {h0 * rd0:.2f} Mpc (DESI 101.54 +- 0.73), chi2 = {r0.fun:.1f}/13")
ok("V1 the pipeline reproduces DESI's BAO-only Omega_m and h r_d", abs(Om0 - 0.2975) < 0.009 and abs(h0 * rd0 - 101.54) < 1.5)

combos = {'DESI+CMB': lambda p: bao_chi2(p) + cmb_chi2(p),
          'DESI+CMB+Pantheon+': lambda p: bao_chi2(p) + cmb_chi2(p) + sn_chi2(p, 'P+'),
          'DESI+CMB+DESY5(Dovekie)': lambda p: bao_chi2(p) + cmb_chi2(p) + sn_chi2(p, 'D5')}
published = {'DESI+CMB': (-12.5, 3.1), 'DESI+CMB+Pantheon+': (-10.7, 2.8), 'DESI+CMB+DESY5(Dovekie)': (None, 3.2)}
out = {}
for name, fun in combos.items():
    rl = fit(fun, True, starts); rw = fit(fun, False, starts)
    dchi = rw.fun - rl.fun
    sig = norm.isf(chi2d.sf(-dchi, 2) / 2) if dchi < 0 else 0.0
    out[name] = (rl, rw, dchi, sig)
    print(f"   {name:24s} LCDM chi2 {rl.fun:8.2f}; w0waCDM chi2 {rw.fun:8.2f} at w0 = {rw.x[3]:+.3f}, wa = {rw.x[4]:+.3f}, Om = {rw.x[0]:.3f}; "
          f"dchi2 = {dchi:+.2f} -> {sig:.1f} sigma (published {published[name]})", flush=True)

ok("V2 DESI+CMB: our compressed-CMB fit prefers w0 > -1, wa < 0 like DESI", out['DESI+CMB'][1].x[3] > -1 and out['DESI+CMB'][1].x[4] < 0)
ok("V3 adding SNe keeps w0 > -1, wa < 0 in both samples", all(out[k][1].x[3] > -1 and out[k][1].x[4] < 0 for k in list(out)[1:]))

# V4 our independent significances against the published ones (DESI DR2 II; DES Dovekie re-analysis 3.2 sigma per later reviews)
sig = {k: v[3] for k, v in out.items()}
# The sigma values are the frequentist conversion of delta chi2 for 2 extra parameters (as DESI). This check only records the range; it is
# not a validation of the probabilistic interpretation, and the published numbers use full (and, for Dovekie, Planck+ACT+SPT) CMB data.
ok(f"V4 recorded: constant Lambda disfavoured at {min(sig.values()):.1f}-{max(sig.values()):.1f} sigma by delta chi2 (approximate pipeline, "
   "compressed Planck priors); published 2.8-3.2 sigma use other CMB data -- direction and size compared, not numbers", 1.5 < min(sig.values()) and max(sig.values()) < 4.0)

# V5 the preferred CPL solutions cross into phantom in the past (w(a -> 0) = w0 + wa < -1): a relabelling medium cannot do that stably
#     (62 SD3: phantom label media have c_s^2 < -1). If the preference is real, it needs something beyond label media.
ok("V5 best fits have w0 + wa < -1 (phantom crossing): " + ", ".join(f"{k}: {v[1].x[3] + v[1].x[4]:+.2f}" for k, v in out.items()),
   all(v[1].x[3] + v[1].x[4] < -1 for v in out.values()))
