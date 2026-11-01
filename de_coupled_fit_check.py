# 86c: full background fit of the coupled model (dust with m(phi) = m0 e^(beta phi), scalar with constant potential) to DESI DR2 BAO +
# compressed Planck + SN with our doc-70 pipeline, profiling (Omega_m, h, omega_b) at fixed beta. CF1-CF2.
# The model is parametrised by Om = the matter density today that matter would have WITHOUT the coupling (rho_m,early a^3/(3H0^2)); the
# field starts at rest at z = 999 (where the model equals LCDM with that Om; radiation from the pipeline). For z < 999: E_c(z) = E_LCDM(z) * R(z), R = H_coupled/H_(beta=0) from the background ODE with the same early matter and the
# potential fixed by E_c(0) = 1. Units 8 pi G = 1, H0 = 1.
# Run: ~/.venvs/jev/bin/python -I de_coupled_fit_check.py DATA_ROOT
import os, sys
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, brentq
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'de_desi_fit_check.py')).read().split('# V1 validation')[0]
ns = {'__name__': 'pipeline'}; exec(src, ns)
ok = ns['ok']; Ez0 = ns['Ez']

def ratio_table(beta, Om):
    ai = 1e-3
    rmi = 3 * Om * ai**-3                                       # rho_m at a_i in units 3H0^2 Om a^-3
    def run(V0):
        def f(N, y):
            ph, pd = y; rm = rmi * np.exp(-3 * (N - np.log(ai)) + beta * ph); h = np.sqrt((rm + 0.5 * pd**2 + V0) / 3)
            return [pd / h, (-3 * h * pd - beta * rm) / h]
        s = solve_ivp(f, [np.log(ai), 0.0], [0.0, 0.0], rtol=1e-9, atol=1e-12, dense_output=True)
        return s
    def H0of(V0):
        s = run(V0); ph, pd = s.sol(0.0); rm = rmi * ai**3 * np.exp(beta * ph); return np.sqrt((rm + 0.5 * pd**2 + V0) / 3)
    V0 = brentq(lambda V: H0of(V) - 1.0, 0.0, 6.0, xtol=1e-12)
    s = run(V0)
    zg = np.expm1(np.linspace(0.0, np.log(999.0), 600)); N = -np.log1p(zg)
    ph, pd = s.sol(N); rm = rmi * np.exp(-3 * (N - np.log(ai)) + beta * ph)
    Hc = np.sqrt((rm + 0.5 * pd**2 + V0) / 3)
    Hl = np.sqrt(Om * (1 + zg)**3 + (1 - Om))                     # beta = 0, matter + Lambda
    return zg, Hc / Hl

current = {}
def Ez_c(z, Om, h, w0, wa):
    base = Ez0(z, Om, h, -1.0, 0.0)
    zg, R = current['tab']
    return base * (np.interp(z, zg, R) if np.ndim(z) else (np.interp(z, zg, R) if z <= 998 else R[-1]))
ns['Ez'] = Ez_c

def fit_beta(beta, fun):
    best = None
    for x0 in ([0.31, 0.68, 0.0224], [0.30, 0.67, 0.0223], [0.32, 0.67, 0.0224]):
        def f(x):
            Om, h, ob = x
            if not (0.2 < Om < 0.45 and 0.6 < h < 0.78 and 0.020 < ob < 0.025): return 1e9
            current['tab'] = ratio_table(beta, Om)
            return fun((Om, h, ob, -1.0, 0.0))
        r = minimize(f, x0, method='Nelder-Mead', options={'xatol': 1e-5, 'fatol': 1e-3, 'maxiter': 800})
        if best is None or r.fun < best.fun: best = r
    return best

combos = {'DESI+CMB+Pantheon+': lambda p: ns['bao_chi2'](p) + ns['cmb_chi2'](p) + ns['sn_chi2'](p, 'P+'),
          'DESI+CMB+DESY5(Dovekie)': lambda p: ns['bao_chi2'](p) + ns['cmb_chi2'](p) + ns['sn_chi2'](p, 'D5')}
betas = [0.0, 0.03, 0.05, 0.07]
res = {}
for name, fun in combos.items():
    row = []
    for b in betas:
        r = fit_beta(b, fun); row.append((b, r.fun, r.x))
        print(f"   {name:24s} beta = {b:.2f}: chi2 = {r.fun:9.2f} at Om = {r.x[0]:.3f}, h = {r.x[1]:.3f}", flush=True)
    res[name] = row
ok("CF1 beta = 0 reproduces the LCDM chi2 of doc 70 to 0.5", abs(res['DESI+CMB+Pantheon+'][0][1] - 1421.22) < 0.5 and abs(res['DESI+CMB+DESY5(Dovekie)'][0][1] - 1650.57) < 0.5)
gain = {k: v[0][1] - min(x[1] for x in v) for k, v in res.items()}
bb = {k: min(v, key=lambda x: x[1])[0] for k, v in res.items()}
ok("CF2 best coupling and its chi2 gain over LCDM: " + ", ".join(f"{k}: beta = {bb[k]:.2f}, gain {gain[k]:.2f}" for k in res), True)
