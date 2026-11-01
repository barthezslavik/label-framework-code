# 100 (rewritten after the fifth review, 2026-10-22): the lensing g200/eps against ONE velocity model used at all three radii r_t, r_v, r200m.
# Model: the exact Emden sphere inside r_v (box radius x* = 17.69, KA3), v^2(r) = v_v^2 [x psi'(x)]/[x* psi'(x*)] with x = x* r/r_v; outside
# r_v an assumed continuation rho ~ r^-alpha (alpha = 9/4 is the inner asymptote of self-similar secondary infall, Bertschinger 1985; 2.5,
# 2.75 for sensitivity). r_t = 7.16 r_c placed in the sphere at x_t = 10.1 (r_c = sqrt 2) or 16.3 (r_c = half-density radius). Collapse
# epoch of nu-peak shells (KA4). For each lensing bin solve for M_v such that the mean density inside r200m today is 200 rho_m(z_eff) and
# M(r200m) = M200m. Since a0 = v(r_t)^4/(G M_b) and g200/eps = v(r200m)^4/(G M_b), the prediction is (g200/eps)/a0 = (v(r200m)/v(r_t))^4.
# No fit. OU1-OU5. Run: ~/.venvs/jev/bin/python -I de_halo_outer_check.py DATA_ROOT SPARC_TABLE
import os, sys, contextlib, io
import numpy as np
from scipy.optimize import brentq
HERE = os.path.dirname(os.path.abspath(__file__))
hs = {'__name__': 'hs', '__file__': os.path.join(HERE, 'de_halo_share_check.py')}
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(hs['__file__']).read().split('# HS5')[0], hs)
ok = hs['ok']; Mh, zeff, a0t, lms, w, Mbar = hs['Mh'], hs['zeff'], hs['a0t'], hs['lms'], hs['w'], hs['Mbar']
kc = {'__name__': 'kc', '__file__': os.path.join(HERE, 'ga_kappa_check.py')}
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(kc['__file__']).read(), kc)
S, xs = kc['S'], kc['x_star']
h, Om, G, conv = 0.673, 0.315, 4.301e-9, 1e6 / 3.0857e22
rho0 = 2.775e11 * h**2 * Om
u = lambda x: x * S.sol(x)[1]                                     # v^2/sigma^2 in the Emden sphere
uv = u(xs)

def profile(r, Mv, rv, alpha):                                    # M(r), v^2(r)/v_v^2
    if r <= rv:
        q = u(xs * r / rv) / uv
        return Mv * (r / rv) * q, q
    return Mv * (r / rv)**(3 - alpha), (r / rv)**(2 - alpha)

def predict(M200, z, nu, alpha, xt):
    def rv_of(Mv):
        zc = kc['gv'](Mv, nu)[0]
        return (3 * Mv / (4 * np.pi * 18 * np.pi**2 * kc['rho_m0'] * (1 + zc)**3))**(1 / 3)
    def r200_of(Mv, rv):
        f = lambda lr: np.log(3 * profile(np.exp(lr), Mv, rv, alpha)[0] / (4 * np.pi * np.exp(lr)**3) / (200 * rho0 * (1 + z)**3))
        return np.exp(brentq(f, np.log(rv) - 3, np.log(rv) + 4))
    def resid(lMv):
        Mv = 10**lMv; rv = rv_of(Mv); r2 = r200_of(Mv, rv)
        return np.log(profile(r2, Mv, rv, alpha)[0] / M200)
    lMv = brentq(resid, np.log10(M200) - 2.0, np.log10(M200) + 1.0)
    Mv = 10**lMv; rv = rv_of(Mv); r2 = r200_of(Mv, rv)
    v200 = profile(r2, Mv, rv, alpha)[1]; vt = u(xt) / uv
    return r2 / rv, (v200 / vt)**2

t = np.loadtxt(os.path.join(sys.argv[1], 'mandelbaum2016', 'table_B1_blue.txt')); corr = t[:, 6]
rho_z = rho0 * (1 + zeff)**3
def measured(M):
    r2 = (3 * M / (4 * np.pi * 200 * rho_z))**(1 / 3)
    return (G * M / r2**2 * conv) / (Mbar / M) / 1.2e-10
meas = {'mean': measured(Mh), 'typical': measured(Mh / corr)}
def trend(y): A_ = np.vstack([np.ones_like(lms), lms - 10.8]).T; return np.linalg.lstsq(A_ * np.sqrt(w)[:, None], np.log10(y) * np.sqrt(w), rcond=None)[0][1]
def chi(m, p): return float(np.sum(w * (np.log10(m) - np.log10(p))**2))
mavg = lambda y: 10**np.average(np.log10(y), weights=w)
for k, m in meas.items():
    print(f"   measured (g200/eps)/a0, {k} masses: {', '.join(f'{x:.2f}' for x in m)}; mean {mavg(m):.2f}, trend {trend(m):+.2f}")
rows = []
for nu in (1.0, 1.5):
    for xt in (10.13, 16.3):
        for alpha in (2.25, 2.5, 2.75):
            for k, Ms in (('mean', Mh), ('typical', Mh / corr)):
                P = np.array([predict(M, z, nu, alpha, xt) for M, z in zip(Ms, zeff)])
                rows.append((nu, xt, alpha, k, P[:, 0], P[:, 1], mavg(P[:, 1]), trend(P[:, 1]), chi(meas[k], P[:, 1])))
for r in rows:
    print(f"   nu {r[0]}, x_t {r[1]:5.2f}, alpha {r[2]}, {r[3]:7s}: r200/r_v {', '.join(f'{x:.1f}' for x in r[4])}; pred {', '.join(f'{x:.2f}' for x in r[5])}; "
          f"mean {r[6]:.2f}, trend {r[7]:+.2f}, chi2 {r[8]:.1f}")
sel = lambda nu, xt, al, k: next(r for r in rows if r[0] == nu and r[1] == xt and r[2] == al and r[3] == k)
r94 = [sel(nu, xt, 2.25, 'typical') for nu in (1.0, 1.5) for xt in (10.13, 16.3)]
ok(f"OU1 one velocity model, alpha = 9/4, typical masses: predicted mean {min(r[6] for r in r94):.2f}-{max(r[6] for r in r94):.2f} "
   f"(nu = 1-1.5, both r_c identifications) vs measured {mavg(meas['typical']):.2f}; chi2 {min(r[8] for r in r94):.1f}-{max(r[8] for r in r94):.1f}", True)
ok(f"OU2 predicted trend {min(r[7] for r in r94):+.2f}-{max(r[7] for r in r94):+.2f} vs measured {trend(meas['typical']):+.2f} (typical), "
   f"{trend(meas['mean']):+.2f} (mean masses)", all(r[7] > 0 for r in r94))
best = min(rows, key=lambda r: r[8])
ok(f"OU3 best of the grid (not a fit, a scan of the assumptions): nu {best[0]}, x_t {best[1]}, alpha {best[2]}, {best[3]} masses, chi2 {best[8]:.1f} for 5 points", True)
ok("OU4 sensitivity to the assumed outer slope: " + "; ".join(f"alpha {al}: mean {sel(1.5, 10.13, al, 'typical')[6]:.2f}, chi2 {sel(1.5, 10.13, al, 'typical')[8]:.1f}" for al in (2.25, 2.5, 2.75)), True)
flat = [sel(nu, xt, 2.25, 'typical')[4] for nu in (1.0, 1.5) for xt in (10.13, 16.3)]
ok(f"OU5 r200m/r_v ranges {min(f.min() for f in flat):.1f}-{max(f.max() for f in flat):.1f}: bins with r200m < r_v use the Emden interior", True)
