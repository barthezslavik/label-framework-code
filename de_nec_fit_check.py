# 86b: the framework's prediction from 86 -- the CPL phantom of DESI is a parametrisation artefact -- tested with our pipeline (doc 70):
# fit DESI DR2 BAO + compressed Planck + SN with CPL constrained to w(a) >= -1 for all 0 < a <= 1 (null energy condition: w0 >= -1 and
# w0 + wa >= -1) and compare with free CPL and LCDM. N1-N2. Data are the same untrusted files as in doc 70 (path argument, python -I).
# Run: ~/.venvs/jev/bin/python -I de_nec_fit_check.py DATA_ROOT
import os, sys
import numpy as np
from scipy.optimize import minimize
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'de_desi_fit_check.py')).read().split('# V1 validation')[0]
ns = {'__name__': 'pipeline'}; exec(src, ns)
bao_chi2, cmb_chi2, sn_chi2, starts, ok = ns['bao_chi2'], ns['cmb_chi2'], ns['sn_chi2'], ns['starts'], ns['ok']

def fit(fun, mode):
    best = None
    for s in starts:
        x0 = s[:3] if mode == 'lcdm' else s.copy()
        if mode == 'nec':
            x0[3] = max(x0[3], -0.95); x0[4] = max(x0[4], -1 - x0[3] + 0.05)
        def f(x):
            p = (x[0], x[1], x[2], -1.0, 0.0) if mode == 'lcdm' else tuple(x)
            if not (0.1 < p[0] < 0.6 and 0.5 < p[1] < 0.9 and 0.015 < p[2] < 0.03 and -3 < p[3] < 1 and -5 < p[4] < 3 and p[3] + p[4] < 0):
                return 1e9
            if mode == 'nec' and (p[3] < -1 or p[3] + p[4] < -1):
                return 1e9
            return fun(p)
        r = minimize(f, x0, method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-4, 'maxiter': 6000, 'maxfev': 9000})
        if best is None or r.fun < best.fun:
            best = r
    return best

combos = {'DESI+CMB': lambda p: bao_chi2(p) + cmb_chi2(p),
          'DESI+CMB+Pantheon+': lambda p: bao_chi2(p) + cmb_chi2(p) + sn_chi2(p, 'P+'),
          'DESI+CMB+DESY5(Dovekie)': lambda p: bao_chi2(p) + cmb_chi2(p) + sn_chi2(p, 'D5')}
rows = {}
for name, fun in combos.items():
    rl, rc, rn = fit(fun, 'lcdm'), fit(fun, 'cpl'), fit(fun, 'nec')
    rows[name] = (rl.fun, rc.fun, rn.fun, rn.x)
    print(f"   {name:24s} chi2 LCDM {rl.fun:8.2f} | CPL free {rc.fun:8.2f} (w0 {rc.x[3]:+.2f}, wa {rc.x[4]:+.2f}) | CPL with w >= -1 {rn.fun:8.2f} "
          f"(w0 {rn.x[3]:+.2f}, wa {rn.x[4]:+.2f}); LCDM - NEC = {rl.fun - rn.fun:+.2f}, NEC - free = {rn.fun - rc.fun:+.2f}", flush=True)

# N1 the non-phantom (NEC) class cannot do better than LCDM by much: its best fit sits on the boundary w0 + wa = -1 or w0 = -1.
ok("N1 recorded: chi2 of the NEC-respecting CPL class for each combination (printed)", all(np.isfinite(v[2]) for v in rows.values()))
# N2 the price of forbidding the phantom: chi2(NEC) - chi2(free CPL) > 0 measures how much the data rely on the phantom part of CPL.
ok("N2 forbidding w < -1 costs Delta chi2 = " + ", ".join(f"{k}: {v[2] - v[1]:+.1f}" for k, v in rows.items()),
   all(v[2] - v[1] >= -0.05 for v in rows.values()))
