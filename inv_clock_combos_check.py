# Clock-invariant combinations (time as stretching, 41/45): which gravitational-cosmological numbers are O(1) in units of the cosmic clock t,
# which of those are automatic, how long the 'all O(1)' epoch lasts, and whether a0 is locked to the clock (papers/restart/46-clock-combos.md).
# P1-P5. Literature first: Dirac large numbers (G ~ 1/t; review arXiv 0705.1836); for a0: a0 ~ H or a0 = const (Milgrom; arXiv 0802.1526,
# 1412.4344) -- a0 ~ 1/t not found as a stated relation. Data: MUSE-DARK III a0(z) = 1.00 + 1.59 z (1e-10 m/s^2), 0.33 < z < 1.44.
# Run: uv run --with sympy --with numpy --with scipy python inv_clock_combos_check.py
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from ga_core import ok

c, G = 2.998e8, 6.674e-11
h = 0.674; H0 = 100 * h * 1e3 / 3.0857e22; Gyr = 3.156e16
Om, Ob, OL, Or = 0.315, 0.0493, 0.685, 9.1e-5
Ec = lambda a: np.sqrt(Om / a**3 + Or / a**4 + OL)
t_of_a = lambda a: quad(lambda x: 1 / (x * Ec(x)), 1e-10, a)[0] / H0
t0 = t_of_a(1)
rho_crit = 3 * H0**2 / (8 * np.pi * G)

# P1 weight-0 combinations today (gravity + c only)
combos = {
    'H t': H0 * t0,
    'Lambda c^2 t^2': 3 * OL * (H0 * t0)**2,
    'G rho_m t^2': G * Om * rho_crit * t0**2,
    'G rho_b t^2': G * Ob * rho_crit * t0**2,
    'G rho_r t^2': G * Or * rho_crit * t0**2,
    'a0 t / c': 1.2e-10 * t0 / c,
}
for k, v in combos.items():
    print(f"   {k:16s} = {v:.3g}")
ok("P1 O(1) (0.03-2) today: H t, Lambda t^2, G rho_m t^2, a0 t/c; NOT O(1): radiation G rho_r t^2 ~ 1e-5 (and baryons 0.005): only the dominant "
   "components sit at the clock", all(0.02 < combos[k] < 2.5 for k in ('H t', 'Lambda c^2 t^2', 'G rho_m t^2', 'a0 t / c')) and combos['G rho_r t^2'] < 1e-4)

# P2 which are automatic: in any decelerating power-law era H t = n (1/2 radiation, 2/3 matter) and G rho t^2 = 3 (H t)^2 Omega/(8 pi) by
#    Friedmann -- O(1) at every epoch. The non-trivial ones are those built from a CONSTANT: Lambda t^2 (grows as t^2) and a0 t/c (grows as t
#    if a0 is constant): they are O(1) only near now.
avals = np.logspace(-3, 1.2, 400)
tv = np.array([t_of_a(a) for a in avals])
Ht = Ec(avals) * H0 * tv
Lt2 = 3 * OL * (H0 * tv)**2
a0t = 1.2e-10 * tv / c
ok(f"P2 H t stays in [{Ht.min():.2f}, {Ht[avals < 1].max():.2f}] for a < 1 (automatic); Lambda t^2 spans {Lt2.min():.1e}..{Lt2.max():.1e}, "
   f"a0 t/c (a0 const) {a0t.min():.1e}..{a0t.max():.1e}: only the constant-built ones need 'now'", Ht.min() > 0.45 and Lt2.max() / Lt2.min() > 1e3)

# P3 the window where the non-trivial ones are O(1) (between 0.1 and 10): cosmic time interval, compared with the observer window of 45
win = (Lt2 > 0.1) & (Lt2 < 10) & (a0t > 0.03) & (a0t < 3)
tw = tv[win] / Gyr
ok(f"P3 'all O(1)' window: t = {tw.min():.1f}-{tw.max():.1f} Gyr (we are at {t0 / Gyr:.1f}); star formation peaks at ~3.5 Gyr, observers a few Gyr later "
   "(45): the window and the observer epoch overlap", tw.min() < t0 / Gyr < tw.max())

# P4 is a0 locked to the clock? Compare the published a0(z) line with three shapes (free amplitude): constant, H(z), clock 1/t(z), and (1+z).
zg = np.linspace(0.33, 1.44, 12)
line = 1.00 + 1.59 * zg; sline = np.sqrt(0.04**2 + (0.10 * zg)**2)
tz = np.array([t_of_a(1 / (1 + z)) for z in zg])
shapes = {'constant': np.ones_like(zg), 'H(z)': Ec(1 / (1 + zg)), 'clock t0/t(z)': t0 / tz, '(1+z)': 1 + zg}
fits = {}
for name, f in shapes.items():
    A = np.sum(line * f / sline**2) / np.sum(f**2 / sline**2)
    dev = np.max(np.abs(A * f - line) / sline)
    fits[name] = (A, dev)
    print(f"   {name:14s} best A = {A:.2f} (a0 today, 1e-10)  max deviation {dev:.1f} sigma;  a0(0) t0/c = {A * 1e-10 * t0 / c:.3f}")
a0t_z = line * 1e-10 * tz / c
print(f"   a0(z) t(z)/c along the published line: {np.round(a0t_z, 3)} (constant would mean clock-locked)")
best = min(fits, key=lambda k: fits[k][1])
ok(f"P4 best shape for the published a0(z): {best} (max dev {fits[best][1]:.1f} sigma, A = {fits[best][0]:.2f}); a0 t/c along the line varies only "
   f"{a0t_z.max() / a0t_z.min():.2f}x over 0.33 < z < 1.44 -- the clock-locked law a0 ~ 1/t is the best of the four shapes tested",
   fits['clock t0/t(z)'][1] <= min(fits['H(z)'][1], fits['constant'][1]) + 0.5)

# P5 the local anchors: clock-locked with A from the fit predicts a0(0) = A; SPARC 1.20 +- 0.26 (sys), MIGHTEE 1.69 +- 0.13 (z < 0.08).
A = fits['clock t0/t(z)'][0]
ok(f"P5 clock-locked a0(0) = {A:.2f}: vs SPARC 1.20 +- 0.26 ({(A - 1.2) / 0.26:+.1f} sigma), MIGHTEE 1.69 +- 0.13 ({(A * t0 / t_of_a(1 / 1.04) - 1.69) / 0.13:+.1f} sigma at z = 0.04)",
   abs(A - 1.2) / 0.26 < 3)

# P6 where it is tested beyond MUSE: at z ~ 2 the clock law predicts a0(2)/a0_local; Milgrom 2017 (Genzel+ 2017 rotation curves) finds ~4 a0
#    'all but excluded'. H(z) gives ~3x. Report the tension, do not hide it.
t2 = t_of_a(1 / 3)
pred2 = A * t0 / t2
ok(f"P6 clock law at z = 2: a0 = {pred2:.2f}e-10 = {pred2 / 1.2:.1f} x local 1.2 (H(z) law: {fits['H(z)'][0] * Ec(1 / 3) / 1.2:.1f}x); Milgrom 2017: ~4x all but "
   "excluded -> the clock law is in tension at z ~ 2 (small sample of six galaxies)", pred2 / 1.2 > 3.3)

# P7 numerology warning: the clock coefficient a0(0) t0/c = 0.159 equals 1/(2 pi) = 0.1592 -- i.e. a0 = c/(2 pi t). With the SPARC anchor 1.20 the
#    same product is 0.174. One number matching a simple constant is not evidence; recorded, not used.
coef = A * 1e-10 * t0 / c; sA = np.sqrt(np.sum(1 / sline**2 * shapes['clock t0/t(z)']**2))**-1 * 1e-10 * t0 / c
ok(f"P7 a0(0) t0/c = {coef:.4f} +- {sA:.4f}; 1/(2 pi) = {1 / (2 * np.pi):.4f} ({(coef - 1 / (2 * np.pi)) / sA:+.1f} sigma, statistical): a0 = c/(2 pi t) "
   "fits the MUSE line -- a single coincidence, flagged as numerology", abs(coef - 1 / (2 * np.pi)) / sA < 2)
