# The "generalized Doppler" idea (author, 2026-10-19): every epoch looks as if accelerated expansion is just starting, while the real expansion
# accelerates (papers/restart/40-synthesis.md, question 4). Three exact readings, tested on published data. G1-G5.
# Apparent Hubble rate of an observer = what redshift-time data give: H_app = -(1+z)^-1 dz/dt_e at z -> 0; apparent q_app = -1 + dln H_app/dln(1+z).
# Data: DES SN time dilation b = 1.003 +- 0.005 (stat) +- 0.010 (sys), White+ 2024 (MNRAS 533, 3365; arXiv 2406.05050);
#       DESI DR2 BAO Table 4 (arXiv 2503.14738): D_M/r_d, D_H/r_d, D_M/D_H at z_eff.
# Run: uv run --with sympy --with numpy --with scipy python inv_doppler_check.py
import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from ga_core import ok

# G1 reading A: the spectral redshift is a generalized map 1+z = (a_o/a_e)^k (geometry unchanged). Then H_app = k H and
#    q_app = -1 + (1+q)/k. Apparent q = 0 at every epoch needs k = 1+q(t) for all t; for a power law a ~ t^n (q = 1/n - 1) this is k = 1/n,
#    and then H_app t = 1 for every observer: apparent coasting with real acceleration (n > 1).
t, n, k = sp.symbols('t n k', positive=True)
a = t**n
H = sp.diff(a, t) / a
q = sp.simplify(-a * sp.diff(a, t, 2) / sp.diff(a, t)**2)
q_app = -1 + (1 + q) / k
ok(f"G1 power law a ~ t^n: q = {q}; q_app = 0 requires k = {sp.solve(sp.Eq(q_app, 0), k)}; then H_app t = k H t = {sp.simplify((1 / n) * H * t)}",
   sp.solve(sp.Eq(q_app, 0), k) == [1 / n] and sp.simplify((1 / n) * H * t - 1) == 0)

# G2 but the arrival-time stretch of a signal is pure geometry: delta t_o / delta t_e = a_o/a_e = (1+z)^(1/k) = (1+z)^n. So the SN light-curve
#    exponent is b = n = 1/(1+q). DES measures b = 1.003 +- 0.011 -> q = 1/b - 1 = -0.003 +- 0.011: no real acceleration. Real q = -0.53 would
#    need b = 2.1.
b, sb = 1.003, np.hypot(0.005, 0.010)
q_from_b = 1 / b - 1; sq = sb / b**2
ok(f"G2 reading A predicts SN time dilation (1+z)^b with b = 1/(1+q): DES b = {b} +- {sb:.3f} -> q = {q_from_b:+.3f} +- {sq:.3f}; "
   f"q = -0.53 would need b = {1 / (1 - 0.53):.2f} ({(1 / (1 - 0.53) - b) / sb:.0f} sigma): reading A is excluded", (1 / 0.47 - b) / sb > 20)

# G3 reading B: a genuine (generalized) Doppler shift scales frequencies and clock rates alike, so b = 1 and, for comoving emitters and
#    observers, 1+z = a_o/a_e: then q_app = q. 'q_app = 0 at every epoch' is then exactly coasting, a ~ t (H t = 1) -- no acceleration at all.
#    Test coasting itself on DESI DR2 (r_d-free D_M/D_H, and D_H/r_d with one free scale).
z = np.array([0.510, 0.706, 0.934, 1.321, 1.484, 2.330])
FMH = np.array([0.622, 0.892, 1.223, 1.948, 2.386, 4.518]); sF = np.array([0.017, 0.021, 0.019, 0.045, 0.136, 0.097])
DH = np.array([21.863, 19.455, 17.641, 14.176, 12.817, 8.632]); sDH = np.array([0.425, 0.330, 0.193, 0.221, 0.516, 0.101])
F_coast = (1 + z) * np.log(1 + z)
def F_lcdm(Om):
    E = lambda x: np.sqrt(Om * (1 + x)**3 + 1 - Om)
    return np.array([E(zz) * quad(lambda x: 1 / E(x), 0, zz)[0] for zz in z])
chi = lambda m, d, s: np.sum(((m - d) / s)**2)
best = minimize_scalar(lambda Om: chi(F_lcdm(Om), FMH, sF), bounds=(0.1, 0.6), method='bounded')
c_coast = chi(F_coast, FMH, sF); c_l = best.fun
# D_H/r_d shape: coasting D_H ~ 1/(1+z) with one free scale A
def chi_scale(shape):
    A = np.sum(DH * shape / sDH**2) / np.sum(shape**2 / sDH**2)
    return chi(A * shape, DH, sDH)
E_b = lambda x: np.sqrt(best.x * (1 + x)**3 + 1 - best.x)
c_dh_coast = chi_scale(1 / (1 + z)); c_dh_l = chi_scale(1 / E_b(z))
print(f"   D_M/D_H: coasting {np.round(F_coast, 3)} vs data {FMH}")
print(f"   chi2 (6 points): D_M/D_H coasting {c_coast:.1f}, LCDM (Om = {best.x:.3f}) {c_l:.1f}; D_H/r_d with free scale: coasting {c_dh_coast:.0f}, LCDM {c_dh_l:.1f}")
ok(f"G3 reading B = coasting: DESI DR2 chi2 = {c_coast:.0f} (D_M/D_H) and {c_dh_coast:.0f} (D_H) for 6 points vs LCDM {c_l:.1f}, {c_dh_l:.1f}: "
   "excluded (the Ly-alpha D_H shows deceleration at z = 2.3 directly)", c_coast > 25 and c_dh_coast > 100 and c_l < 15)

# G4 reading C: 'the onset is always seen at the same redshift'. In one FLRW history the onset is a fixed event a_acc, and an observer at a_obs
#    sees it at z_t = a_obs/a_acc - 1, which grows monotonically with a_obs (LCDM: -0.35, 0, 0.63, 1.45, 3.9 for a_obs = 0.4..3): impossible
#    unless the history itself depends on the observer's epoch.
a_acc = 0.613
zt = lambda ao: ao / a_acc - 1
ok(f"G4 z_t(a_obs) = a_obs/a_acc - 1: {[round(zt(x), 2) for x in (0.4, 0.613, 1.0, 1.5, 3.0)]}, strictly increasing: a fixed apparent onset is impossible in one history",
   all(np.diff([zt(x) for x in (0.4, 0.613, 1.0, 1.5, 3.0)]) > 0))

# G5 what survives: (i) the coincidence itself, H0 t0 = 0.951 in LCDM (Avelino-Kirshner 2016: 0.96 +- 0.01) -- 'now' is the epoch where
#    the history-averaged q is ~ 0; (ii) clock/ruler renormalisation of non-comoving or bound observers (timescape, Wiltshire), a known and
#    contested model, not tested here.
Om0 = 0.315
Ht = np.sqrt(Om0 + 1 - Om0) * quad(lambda x: 1 / (x * np.sqrt(Om0 / x**3 + 1 - Om0)), 1e-8, 1)[0]
ok(f"G5 LCDM H0 t0 = {Ht:.3f} ~ 1 (the observed coincidence): the history-averaged deceleration is ~0 only now", abs(Ht - 0.951) < 0.005)
