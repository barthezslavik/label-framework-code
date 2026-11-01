# 126: H3 of PREDICTIONS.md as a fit, not a scenario. The coupled model m(phi) = m0 e^(beta phi) with a constant potential (doc 86, the
# background of de_coupled_fit_check) is fitted to DESI DR2 BAO + compressed Planck + Pantheon+ TOGETHER with a Gaussian S8 term from weak
# lensing, profiling (Omega_m, h, omega_b) on a grid of beta. S8 is computed for each model: sigma8 from the linear spectrum with A_s and n_s
# fixed at Planck (Eisenstein-Hu no-wiggle transfer of the early universe, which the coupling does not change) times the growth D(a = 1) of
# the coupled model itself (two fluids: dark dust with friction beta phi' and G(1 + 2 beta^2) between dust elements, baryons uncoupled;
# rewritten after paper7 review r14 -- doc 86 coupled all matter), one calibration constant fixing
# sigma8 = 0.811 for Planck LCDM. Two lensing results: KiDS-DES joint 0.818 +- 0.013 and DES Y6 0.789 +- 0.012 (both flat-LCDM summaries:
# an approximation for the coupled model, recorded). The beta bounds are conditional on A_s and n_s fixed: the Planck sigma8 uncertainty
# (~0.006) is not propagated. HS1-HS6.
# Run: ~/.venvs/jev/bin/python -I de_h3_s8_fit_check.py DATA_ROOT [NPROC]
import os, sys
import numpy as np
from multiprocessing import Pool
from scipy.integrate import solve_ivp, quad
from scipy.optimize import minimize, brentq
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'de_desi_fit_check.py')).read().split('# V1 validation')[0]
ns = {'__name__': 'pipeline'}; exec(src, ns)
ok = ns['ok']; Ez0 = ns['Ez']
NP = int(sys.argv[2]) if len(sys.argv) > 2 else 6
As, nsp, kp = 2.1e-9, 0.965, 0.05                                   # Planck amplitude and tilt, pivot 0.05/Mpc

def model(beta, Om, fb=0.157):
    """two fluids (review r14 of paper7): dark dust coupled through m(phi), baryons uncoupled (Cassini). Background ratio table
    R(z) = H_c/H_LCDM(Om) and the growth D(a=1) of the total matter contrast (D = a at a_i); H0 = 1, 8 pi G = 1.
    Growth: dc'' + (2H + beta phi') dc' = (rho_b db + (1 + 2 beta^2) rho_c dc)/2, db'' + 2H db' = (rho_b db + rho_c dc)/2."""
    ai = 1e-3; rci = 3 * Om * (1 - fb) * ai**-3; rbi = 3 * Om * fb * ai**-3
    def rhs(N, y, V0):
        ph, pd, dc, dcp, db, dbp = y
        x = np.exp(-3 * (N - np.log(ai))); rc = rci * x * np.exp(beta * ph); rb = rbi * x
        h = np.sqrt((rc + rb + 0.5 * pd**2 + V0) / 3)
        return [pd / h, (-3 * h * pd - beta * rc) / h,
                dcp / h, (-(2 * h + beta * pd) * dcp + 0.5 * (rb * db + (1 + 2 * beta**2) * rc * dc)) / h,
                dbp / h, (-2 * h * dbp + 0.5 * (rb * db + rc * dc)) / h]
    v0 = ai * np.sqrt((rci + rbi) / 3)
    def run(V0):
        return solve_ivp(rhs, [np.log(ai), 0.0], [0.0, 0.0, ai, v0, ai, v0], args=(V0,), rtol=1e-9, atol=1e-13, dense_output=True)
    def H0of(V0):
        ph, pd = run(V0).sol(0.0)[:2]; rc = rci * ai**3 * np.exp(beta * ph); rb = rbi * ai**3
        return np.sqrt((rc + rb + 0.5 * pd**2 + V0) / 3)
    V0 = brentq(lambda V: H0of(V) - 1.0, 0.0, 6.0, xtol=1e-12)
    s = run(V0)
    zg = np.expm1(np.linspace(0.0, np.log(999.0), 600)); N = -np.log1p(zg)
    sol = s.sol(N); ph, pd = sol[0], sol[1]
    x = np.exp(-3 * (N - np.log(ai))); rc = rci * x * np.exp(beta * ph); rb = rbi * x
    Hc = np.sqrt((rc + rb + 0.5 * pd**2 + V0) / 3); Hl = np.sqrt(Om * (1 + zg)**3 + (1 - Om))
    y0 = s.sol(0.0); rc0 = rci * ai**3 * np.exp(beta * y0[0]); rb0 = rbi * ai**3
    D1 = (rc0 * y0[2] + rb0 * y0[4]) / (rc0 + rb0)
    Om_today = (rc0 + rb0) / 3                                      # H0 = 1: today's matter fraction (the coupling lowers the dust)
    return zg, Hc / Hl, D1, Om_today

def sigma8(Om, h, ob, D1):
    om = Om * h**2; fb = ob / om; th = 2.7255 / 2.7
    s_ = 44.5 * np.log(9.83 / om) / np.sqrt(1 + 10 * ob**0.75)
    al = 1 - 0.328 * np.log(431 * om) * fb + 0.38 * np.log(22.3 * om) * fb**2
    def T(k):                                                         # k in h/Mpc
        G = Om * h * (al + (1 - al) / (1 + (0.43 * k * h * s_)**4)); q = k * th**2 / G
        L = np.log(2 * np.e + 1.8 * q); C = 14.2 + 731 / (1 + 62.5 * q); return L / (L + C * q**2)
    H0 = 1 / 2997.92458                                                # h/Mpc
    D2 = lambda k: 4 / 25 * As * (k * h / kp)**(nsp - 1) * (k / H0)**4 * T(k)**2 * (D1 / Om)**2
    W = lambda x: 3 * (np.sin(x) - x * np.cos(x)) / x**3
    return np.sqrt(quad(lambda lk: D2(np.exp(lk)) * W(8 * np.exp(lk))**2, np.log(1e-4), np.log(1e2), limit=400)[0])

_, _, D1p, _ = model(0.0, 0.315, 0.0224 / (0.315 * 0.674**2))
CAL = 0.811 / sigma8(0.315, 0.674, 0.0224, D1p)
LENS = {'KiDS-DES': (0.818, 0.013), 'DES-Y6': (0.789, 0.012)}
current = {}
def Ez_c(z, Om, h, w0, wa):
    base = Ez0(z, Om, h, -1.0, 0.0); zg, R = current['tab']
    return base * (np.interp(z, zg, R) if np.ndim(z) else (np.interp(z, zg, R) if z <= 998 else R[-1]))
ns['Ez'] = Ez_c

def evaluate(x, beta, lens):
    Om, h, ob = x
    if not (0.2 < Om < 0.45 and 0.6 < h < 0.78 and 0.020 < ob < 0.025): return 1e9, None
    zg, R, D1, Omt = model(beta, Om, ob / (Om * h**2)); current['tab'] = (zg, R)
    p = (Om, h, ob, -1.0, 0.0)
    c = ns['bao_chi2'](p) + ns['cmb_chi2'](p) + ns['sn_chi2'](p, 'P+')
    S8 = CAL * sigma8(Om, h, ob, D1) * np.sqrt(Omt / 0.3)              # review r16: today's Omega_m in the S8 definition
    if lens is not None:
        c += ((S8 - LENS[lens][0]) / LENS[lens][1])**2
    return c, S8

def fit(args):
    beta, lens = args
    best = None
    for x0 in ([0.31, 0.68, 0.0224], [0.30, 0.69, 0.0223], [0.32, 0.67, 0.0225]):
        r = minimize(lambda x: evaluate(x, beta, lens)[0], x0, method='Nelder-Mead', options={'xatol': 1e-5, 'fatol': 1e-3, 'maxiter': 600})
        if best is None or r.fun < best.fun: best = r
    c, S8 = evaluate(best.x, beta, lens)
    zg, R, _, _ = model(beta, best.x[0], best.x[2] / (best.x[0] * best.x[1]**2))
    E = lambda z: np.interp(z, zg, R) * np.sqrt(best.x[0] * (1 + z)**3 + 1 - best.x[0])
    return beta, lens, c, S8, tuple(best.x)

if __name__ == '__main__':
    print(f"   calibration: sigma8 (EH no-wiggle, growth from a = 1e-3) x {CAL:.4f} = 0.811 at Planck LCDM", flush=True)
    ok(f"HS1 sigma8 calibration constant {CAL:.3f} within 5% of 1 (the linear-spectrum pipeline is sane)", abs(CAL - 1) < 0.05)
    betas = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08]
    jobs = [(b, l) for b in betas for l in (None, 'KiDS-DES', 'DES-Y6')]
    with Pool(NP) as pool:
        res = pool.map(fit, jobs)
    tab = {(b, l): (c, S8, x) for b, l, c, S8, x in res}
    for l in (None, 'KiDS-DES', 'DES-Y6'):
        print(f"   lensing {l}: " + "; ".join(f"beta {b:.2f}: chi2 {tab[(b, l)][0]:.2f}, S8 {tab[(b, l)][1]:.3f}" for b in betas), flush=True)
    c0 = tab[(0.0, None)][0]
    ok(f"HS2 beta = 0 without lensing reproduces the LCDM chi2 of doc 70 (1421.22): {c0:.2f}", abs(c0 - 1421.22) < 1.0)
    def summary(l):
        cs = np.array([tab[(b, l)][0] for b in betas]); i = int(np.argmin(cs))
        dchi = cs[i] - cs[0]
        bf = np.linspace(0, betas[-1], 801); cf = np.interp(bf, betas, cs)
        b1 = bf[cf <= cf.min() + 1.0].max(); b2 = bf[cf <= cf.min() + 4.0].max()
        return betas[i], dchi, b1, b2, tab[(betas[i], l)][1]
    sN, sK, sY = summary(None), summary('KiDS-DES'), summary('DES-Y6')
    ok(f"HS3 without lensing: best beta {sN[0]:.2f}, Delta chi2 vs LCDM {sN[1]:+.2f}, S8 there {sN[4]:.3f}; beta < {sN[2]:.3f} (1 sigma), < {sN[3]:.3f} (2 sigma)", True)
    ok(f"HS4 with KiDS-DES S8: best beta {sK[0]:.2f}, Delta chi2 {sK[1]:+.2f}, S8 {sK[4]:.3f}; beta < {sK[2]:.3f} (1 sigma), < {sK[3]:.3f} (2 sigma)", True)
    ok(f"HS5 with DES Y6 S8: best beta {sY[0]:.2f}, Delta chi2 {sY[1]:+.2f}, S8 {sY[4]:.3f}; beta < {sY[2]:.3f} (1 sigma), < {sY[3]:.3f} (2 sigma)", True)
    surv = {k: v[1] < -2.3 for k, v in (('none', sN), ('KiDS-DES', sK), ('DES-Y6', sY))}
    ok("HS6 does m(phi) improve on LCDM by more than one parameter's worth (Delta chi2 < -2.3)? " + ", ".join(f"{k}: {v}" for k, v in surv.items()), True)
    np.save(os.path.join(HERE, 'results', 'h3_s8_fit3.npy'), np.array([(b, {None: 0, 'KiDS-DES': 1, 'DES-Y6': 2}[l], c, S8, *x) for b, l, c, S8, x in res]))
