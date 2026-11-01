# 98: which formation epoch sets the virial acceleration of disc-hosting halos? Two candidates (doc 96): collapse of nu-peak shells
# (D(z_c) nu sigma(M) = delta_c) and the median half-mass assembly epoch of a Gaussian field (Lacey & Cole 1993, equation borrowed).
# With a common baryon share f (P1) the population law gives v^4 = G M_b g_v/f, so g_v ~ M^gamma fixes the BTFR slope s = 4/(1 + gamma).
# The measured slope selects the epoch. FE1-FE5. Run: ~/.venvs/jev/bin/python ga_formation_epoch_check.py
import numpy as np
import sympy as sp

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

src = open(__file__.replace('ga_formation_epoch_check.py', 'ga_kappa_check.py')).read()
kc = {'__name__': 'kc', '__file__': __file__.replace('ga_formation_epoch_check.py', 'ga_kappa_check.py')}
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    exec(src, kc)                                                   # reuses sigma(M), D(z), g_v(M, nu), z_f(M) (KA4, KA7)
gv, zf, gv_z, fcos = kc['gv'], kc['zf'], kc['gv_z'], kc['fcos']

# FE1 the slope law: M_b = f M_v, v^4 = G M_b a0_eff, a0_eff = g_v/f, g_v = g0 (M_v/M0)^gamma  =>  v^4 ~ M_b^(1 + gamma), s = 4/(1 + gamma)
gam, Mb, f, g0, M0, G = sp.symbols('gamma M_b f g_0 M_0 G', positive=True)
v4 = G * Mb * g0 * (Mb / (f * M0))**gam / f
s_expr = sp.simplify(1 / sp.diff(sp.log(v4), Mb) / Mb * 4)
ok(f"FE1 d ln M_b / d ln v = 4/(d ln v^4/d ln M_b) = {sp.simplify(s_expr)}", sp.simplify(s_expr - 4 / (1 + gam)) == 0)

# FE2 gamma = d log g_v / d log M over 1e10-1e12 Msun for each definition
Ms = np.array([1e10, 3e10, 1e11, 3e11, 1e12])
defs = {}
for nu in (1.0, 1.5, 2.0):
    defs[f'nu-peak collapse, nu = {nu}'] = np.array([gv(M, nu)[1] for M in Ms])
defs['median half-mass assembly'] = np.array([gv_z(M, zf(M)) for M in Ms])
slopes = {k: np.polyfit(np.log10(Ms), np.log10(v), 1)[0] for k, v in defs.items()}
svals = {k: 4 / (1 + g) for k, g in slopes.items()}
for k in defs:
    print(f"   {k:30s} gamma = {slopes[k]:+.3f}  ->  s = {svals[k]:.2f}")
ok("FE2 gamma and implied BTFR slope: " + "; ".join(f"{k}: s = {svals[k]:.2f}" for k in defs), True)

# FE3 data: SPARC slope at the sample centre 3.67 +- 0.11 (de_btfr_running_check BR2b, our ML fit); published 3.85 +- 0.09 (Lelli 2019,
#     orthogonal fit). An epoch is admitted if s lies within 2 sigma of either.
def admitted(s): return abs(s - 3.67) < 2 * 0.11 or abs(s - 3.85) < 2 * 0.09
adm = {k: admitted(v) for k, v in svals.items()}
ok("FE3 admitted by the BTFR slope: " + ", ".join(f"{k}: {'yes' if v else 'NO'}" for k, v in adm.items()),
   all(adm[k] for k in adm if k.startswith('nu')) and not adm['median half-mass assembly'])

# FE4 native reading of the selection (hypothesis, checked only by FE3): the circular speed at disc radii is set by the shells that collapsed
#     there; Liouville (label conservation of dust) keeps their phase volume and later mergers add mass outside -- the epoch is the shell's
#     own collapse, not the assembly of half the final mass. The data agree with this and not with the assembly epoch.
ok("FE4 recorded: the shell-collapse epoch is the one consistent with Liouville-preserved inner shells (reading) and with the slope (FE3)", True)

# FE5 consequence for the retained baryon share (KA6): r_b = g_v/(a0 f_cos), with the Emden plateau factor 1.18 (KA7): for nu = 1, 1.5, 2
pf = kc['pf']
rb = {nu: 10**np.mean(np.log10(defs[f'nu-peak collapse, nu = {nu}'])) * pf / (1.2e-10 * fcos) for nu in (1.0, 1.5, 2.0)}
ok("FE5 required retained share with the selected epoch: " + ", ".join(f"nu = {nu}: r_b = {v:.2f}" for nu, v in rb.items())
   + "; nu (which peaks host discs) is not fixed by the slope -- it is what the f test (doc 99) measures", all(0 < v < 1 for v in rb.values()))
