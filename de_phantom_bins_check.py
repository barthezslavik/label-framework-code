# 97b: where in the DESI DR2 BAO bins do the two native phantom mechanisms gain? Pulls (theory - data)/sigma per data point for constant
# Lambda, CPL (DESI's phantom signal), m(phi) at beta = 0.03 and the 5 best discrete-fluctuation realizations of de_phantom_compare_check
# (seeds read from phantom_compare_300.txt). Statistic: correlation across the 13 bins of the chi2 improvement (diag) with CPL's.
# PB1-PB3. Run: ~/.venvs/jev/bin/python -I de_phantom_bins_check.py DATA_ROOT
import os, sys
import numpy as np
from scipy.optimize import minimize
HERE = os.path.dirname(os.path.abspath(__file__))
F = os.path.join(HERE, 'de_phantom_compare_check.py')
pc = {'__name__': 'pc', '__file__': F}; exec(open(F).read().split("if __name__ == '__main__':")[0], pc)
ns, ev, ok = pc['ns'], pc['ev'], pc['ok']
bz, bv, bq, bcov = ns['bz'], ns['bv'], ns['bq'], ns['bcov']; sig = np.sqrt(np.diag(bcov)); c = ns['c']

def bao_theory(p):
    Om, h, ob, w0, wa = p
    rd = ns['rs'](ns['zdrag'](ob, Om * h**2), p) / ns['RD_CAL']; th = []
    for z, q in zip(bz, bq):
        dm = ns['DM'](z, p); dh = c / (100 * h * ns['Ez'](z, Om, h, w0, wa))
        th.append(dm / rd if q == 'DM_over_rs' else dh / rd if q == 'DH_over_rs' else (z * dm**2 * dh)**(1 / 3) / rd)
    return np.array(th)
tot = lambda p: ns['bao_chi2'](p) + ns['cmb_chi2'](p) + ns['sn_chi2'](p, 'P+')
Ez_std = ev['ns']['Ez'] if False else None
# constant Lambda and CPL with the standard Ez (the everpresent module replaced ns['Ez']; restore the CPL form for these two)
src = open(os.path.join(HERE, 'de_desi_fit_check.py')).read().split('# V1 validation')[0]
std = {'__name__': 'std'}; exec(src, std)
def fit_std(x0, lcdm):
    f = lambda x: 1e9 if not (0.2 < x[0] < 0.45 and 0.6 < x[1] < 0.78) else (std['bao_chi2'](tuple(x[:3]) + (-1.0, 0.0)) + std['cmb_chi2'](tuple(x[:3]) + (-1.0, 0.0)) + std['sn_chi2'](tuple(x[:3]) + (-1.0, 0.0), 'P+') if lcdm else std['bao_chi2'](tuple(x)) + std['cmb_chi2'](tuple(x)) + std['sn_chi2'](tuple(x), 'P+'))
    r = minimize(f, x0, method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-4, 'maxiter': 4000}); return r
rL = fit_std([0.30, 0.684, 0.0224], True); rC = fit_std([0.31, 0.68, 0.0224, -0.85, -0.52], False)
def th_std(p):
    ns_b = ns['Ez']; ns['Ez'] = std['Ez']; t = bao_theory(p); ns['Ez'] = ns_b; return t
pL = (rL.x[0], rL.x[1], rL.x[2], -1.0, 0.0); pC = tuple(rC.x)
tL, tC = th_std(pL), th_std(pC)
print(f"   LCDM chi2 {rL.fun:.2f}; CPL chi2 {rC.fun:.2f} (w0 {pC[3]:.2f}, wa {pC[4]:.2f})")
# m(phi) at beta = 0.03 (doc 86c construction)
G = os.path.join(HERE, 'de_coupled_fit_check.py')
cf = {'__name__': 'cf'}; src_c = open(G).read().split('combos = {')[0].replace("src = open(os.path.join(HERE, 'de_desi_fit_check.py')).read().split('# V1 validation')[0]", "src = open(os.path.join(HERE, 'de_desi_fit_check.py')).read().split('# V1 validation')[0]; HERE = HERE")
cf['__file__'] = G; cf['HERE'] = HERE; exec(src_c.replace("HERE = os.path.dirname(os.path.abspath(__file__))", ""), cf)
fun_c = lambda p: cf['ns']['bao_chi2'](p) + cf['ns']['cmb_chi2'](p) + cf['ns']['sn_chi2'](p, 'P+')
rM = cf['fit_beta'](0.03, fun_c)
cf['current']['tab'] = cf['ratio_table'](0.03, rM.x[0]); pM = (rM.x[0], rM.x[1], rM.x[2], -1.0, 0.0)
def th_m(p):
    nb = ns['Ez']; ns['Ez'] = cf['ns']['Ez']; t = bao_theory(p); ns['Ez'] = nb; return t
tM = th_m(pM)
print(f"   m(phi) beta = 0.03: chi2 {rM.fun:.2f}")
# discrete fluctuation: 5 best seeds
res = np.loadtxt(os.path.join(HERE, 'phantom_compare_300.txt'))
best = res[np.argsort(res[:, 1])[:5]]
tS = []
for row in best:
    s = int(row[0]); g = np.random.default_rng(s).standard_normal(len(pc['LNA']))
    r = pc['fit'](g, pc['A']); lamc, om, ob = r.x
    ev['current']['lnE'], h, _ = ev['history'](pc['A'], om, g, lamc)
    tS.append(bao_theory((om / h**2, h, ob, -1.0, 0.0)))
    print(f"   seed {s}: chi2 {r.fun:.2f}")
pulls = {'LCDM': (tL - bv) / sig, 'CPL': (tC - bv) / sig, 'm(phi)': (tM - bv) / sig}
for i, t in enumerate(tS): pulls[f'S{i}'] = (t - bv) / sig
print("   bin  z     quantity      LCDM    CPL   m(phi)  " + " ".join(f"S{i}" for i in range(5)))
for k in range(len(bz)):
    print(f"   {k:2d} {bz[k]:5.3f} {bq[k]:12s} " + " ".join(f"{pulls[n][k]:+6.2f}" for n in pulls))
gain = {n: pulls['LCDM']**2 - pulls[n]**2 for n in pulls if n != 'LCDM'}
corrS = [np.corrcoef(gain['CPL'], gain[f'S{i}'])[0, 1] for i in range(5)]
corrM = np.corrcoef(gain['CPL'], gain['m(phi)'])[0, 1]
ok(f"PB1 per-bin chi2 gain vs LCDM correlated with CPL's: m(phi) {corrM:+.2f}; best discrete realizations {', '.join(f'{x:+.2f}' for x in corrS)}", True)
ok(f"PB2 BAO diag gain (sum over bins): CPL {gain['CPL'].sum():+.2f}, m(phi) {gain['m(phi)'].sum():+.2f}, discrete best-5 mean {np.mean([gain[f'S{i}'].sum() for i in range(5)]):+.2f}", True)
ok("PB3 recorded: the realizations that beat LCDM are compared bin by bin with DESI's phantom signal (CPL); see the table", True)
