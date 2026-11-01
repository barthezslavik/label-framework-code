# 118: a0(z) with the host epoch taken from the spectrum (G7 of PREDICTIONS.md). Doc 115 had two native epoch rules: (A) fixed nu-peaks
# (too flat) and (B) a fixed growth ratio D(z_c)/D(z) (too steep). A third rule follows from dust collapse in the Gaussian field itself:
# the median epoch at which a halo of mass M observed at z had assembled half its mass (extended Press-Schechter, Lacey & Cole 1993,
# equation borrowed): delta_c/D(z_f) - delta_c/D(z) = w_med sqrt(sigma^2(M/2) - sigma^2(M)), w_med = 0.974. It fixes the difference of
# 1/D, between the fixed z_c of (A) and the fixed ratio of (B), with no free number. a0(z)/a0(0) = g_v(z_f(z))/g_v(z_f(0)) at fixed
# R and eps (paper3 Prop level), g_v ~ rho_vir(z_f)^(2/3) at fixed mass. HE1-HE5.
# Run: ~/.venvs/jev/bin/python ga_a0_host_epoch_check.py
import os, contextlib, io
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
HERE = os.path.dirname(os.path.abspath(__file__))
kc = {'__name__': 'kc', '__file__': os.path.join(HERE, 'ga_kappa_check.py')}
with contextlib.redirect_stdout(io.StringIO()):
    exec(open(kc['__file__']).read(), kc)
D, sigM, ok = kc['D'], kc['sigM'], kc['ok']
Om, dc, wmed = 0.315, 1.686, 0.974
def rho_vir(zc):
    E2 = Om * (1 + zc)**3 + 1 - Om; x = Om * (1 + zc)**3 / E2 - 1
    return (18 * np.pi**2 + 82 * x - 39 * x**2) * E2
gv = lambda zc: rho_vir(zc)**(2 / 3)
def zf(M, z):
    inv = 1 / D(z) + wmed * np.sqrt(sigM(M / 2)**2 - sigM(M)**2) / dc
    return brentq(lambda zz: D(zz) - 1 / inv, z, 80)
tcos = lambda z: quad(lambda zz: 1 / ((1 + zz) * np.sqrt(Om * (1 + zz)**3 + 1 - Om)), z, np.inf)[0]

zs = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
pred = {}
for lM in (11.5, 12.0):
    M = 10**lM; z0 = zf(M, 0.0)
    pred[lM] = np.array([gv(zf(M, z)) / gv(z0) for z in zs])
    print(f"   M = 10^{lM}: z_f(0) = {z0:.2f}, z_f(1) = {zf(M, 1.0):.2f}, z_f(2) = {zf(M, 2.0):.2f}; a0(z)/a0(0) at z = "
          + ", ".join(f"{z:.1f}: {v:.2f}" for z, v in zip(zs, pred[lM])))
it = np.array([tcos(0) / tcos(z) for z in zs])
print("   1/t: " + ", ".join(f"{z:.1f}: {v:.2f}" for z, v in zip(zs, it)))
# HE1 the rule lies between (A) and (B): the relative growth D(z_f)/D(z) falls with z (hosts younger relative to their epoch), z_f rises
M = 10**12; r = [D(zf(M, z)) / D(z) for z in (0.0, 1.0, 2.0)]
ok(f"HE1 rule from the spectrum: D(z_f)/D(z) = {r[0]:.2f}, {r[1]:.2f}, {r[2]:.2f} at z = 0, 1, 2 -- between (A) (falls to 1) and (B) (fixed)",
   r[0] < r[1] < r[2] < 1)

# HE2 shape test exactly as AE3: MUSE-DARK line a0 = (1.00 +- 0.04) + (1.59 +- 0.10) z on 0.33 < z < 1.44, free normalisation. The quoted
#     intervals are 95% CI and the covariance of the two coefficients is not used: the deviation is a diagnostic scale, not a number of sigma
zg = np.linspace(0.33, 1.44, 12)
lin = 1.00 + 1.59 * zg; sl = np.sqrt(0.04**2 + (0.10 * zg)**2)
def maxdev(f):
    w = 1 / sl**2; A = np.sum(w * f * lin) / np.sum(w * f * f)
    return np.max(np.abs(A * f - lin) / sl), A
dev = {}
for lM in (11.5, 12.0):
    M = 10**lM; z0 = zf(M, 0.0)
    dev[lM] = maxdev(np.array([gv(zf(M, z)) / gv(z0) for z in zg]))
dt = maxdev(np.array([tcos(0) / tcos(z) for z in zg]))
ok("HE2 shape vs the MUSE-DARK line (diagnostic deviation in units of the quoted 95% intervals, no covariance; not sigma): " + ", ".join(f"M = 10^{k}: {v[0]:.1f} (norm {v[1]:.2f})" for k, v in dev.items())
   + f"; 1/t {dt[0]:.1f}; doc 115: (A) 4.6-10.7, (B) 3.4-3.5", True)
# HE2b the same with the normalisation fixed at a0(0) = 1 (the line's intercept 1.00 +- 0.04 is in units of the local a0)
fix = lambda f: np.max(np.abs(f - lin) / sl)
d_fix = {lM: fix(np.array([gv(zf(10**lM, z)) / gv(zf(10**lM, 0.0)) for z in zg])) for lM in (11.5, 12.0)}
ok("HE2b fixed normalisation: " + ", ".join(f"M = 10^{k}: {v:.1f}" for k, v in d_fix.items())
   + f"; 1/t {fix(np.array([tcos(0) / tcos(z) for z in zg])):.1f} -- the spectrum rule rises too slowly in amplitude (about 1.3x below the line)", True)
# HE3 the prediction as numbers to z = 3 (where the line has no data): ratio a0(z)/a0(0)
ok("HE3 prediction a0(z)/a0(0) at fixed host mass, R, eps: " + "; ".join(
   f"10^{k}: z = 1 {v[2]:.2f}, z = 2 {v[4]:.2f}, z = 3 {v[5]:.2f}" for k, v in pred.items()) + f"; 1/t: z = 2 {it[4]:.2f}, z = 3 {it[5]:.2f}; MOND: 1",
   all(v[5] > v[2] > 1 for v in pred.values()))
# HE4 sensitivity: the median vs the quartiles of the formation-time distribution (w = 0.6, 1.4 brackets the Lacey-Cole spread)
spread = []
for w_ in (0.6, 1.4):
    def zf_w(M, z, w_=w_):
        inv = 1 / D(z) + w_ * np.sqrt(sigM(M / 2)**2 - sigM(M)**2) / dc
        return brentq(lambda zz: D(zz) - 1 / inv, z, 80)
    M = 10**12; spread.append(gv(zf_w(M, 2.0)) / gv(zf_w(M, 0.0)))
ok(f"HE4 spread of the formation epoch (w = 0.6 / 1.4 instead of 0.974): a0(2)/a0(0) = {spread[0]:.2f} / {spread[1]:.2f} (median {pred[12.0][4]:.2f})", True)
ok("HE5 recorded: the rule is borrowed from the literature as an equation (EPS, dust collapse in the Gaussian field); R, eps held fixed "
   "-- their evolution (gas fractions) is not derived and would shift the curve", True)
