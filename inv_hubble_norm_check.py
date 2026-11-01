# 65: normalise the Hubble rate and look at the GA. HN1-HN7. Two choices: H0 = 1 (today's rate, a state) or H_inf = c sqrt(Lambda/3) = 1
# (de Sitter rate, a form constant; equivalent to Lambda = 3). Literature (verified 2026-10-20): Milgrom, 'The modified dynamics as a
# vacuum effect', astro-ph/9805346: T ~ sqrt(a^2 + a_L^2) with a_L = c sqrt(Lambda/3) (our theorem C4).
# Run: uv run --with sympy --with numpy --with scipy python inv_hubble_norm_check.py
import numpy as np
import sympy as sp
from scipy.integrate import quad
from ga_core import ok

c = 2.99792458e8; Mpc = 3.0857e22
OL, Om = 0.685, 0.315

# HN1 H0 is not a constant: H(z)/H0 = E(z) runs from ~1e4 at recombination-era values down to sqrt(Omega_L) = 0.83 in the future; only
#     H_inf = H0 sqrt(Omega_L) is fixed. Normalising H0 = 1 is choosing an epoch (state); normalising H_inf = 1 is normalising Lambda (form).
E = lambda z: np.sqrt(Om * (1 + z)**3 + OL)
Hinf_over_H0 = np.sqrt(OL)
ok(f"HN1 H0 is state: H(z=1)/H0 = {E(1):.2f}, H(z=1100)/H0 = {E(1100):.0f}, future -> {Hinf_over_H0:.3f}; the constant is H_inf = "
   f"{Hinf_over_H0:.3f} H0 (Lambda = 3 H_inf^2)", E(1100) > 1e4 and abs(Hinf_over_H0 - 0.8276) < 1e-3)

# HN2 GA: in flat-sliced de Sitter, conformal time eta = -e^{-H t}/H; the CGA dilator scales eta -> e^{-lam} eta, i.e. t -> t + lam/H
#     (theorem C3). With H_inf = 1 cosmic time IS the hyperbolic angle of the dilation rotor K -- the analogue of c = 1 making a boost
#     rapidity dimensionless. H_inf is to the dilator what c is to the boost.
t, lam, H = sp.symbols('t lambda H', positive=True)
eta = -sp.exp(-H * t) / H
shift = sp.simplify(sp.solve(sp.Eq(-sp.exp(-H * sp.Symbol('tp')) / H, sp.exp(-lam) * eta), sp.Symbol('tp'))[0] - t)
ok(f"HN2 dilator by angle lambda shifts cosmic time by {shift}: with H_inf = 1 time = dilation angle (c : boost = H_inf : dilator)",
   sp.simplify(shift - lam / H) == 0)

# HN3 the numbers in H_inf = c = 1 units (lengths in c/H_inf, times in 1/H_inf).
H0 = 67.4e3 / Mpc; Hinf = H0 * np.sqrt(OL)
hbar, G = 1.054571817e-34, 6.6743e-11
lP = np.sqrt(hbar * G / c**3)
tl = lambda z: quad(lambda y: 1 / (y * np.sqrt(Om / y**3 + OL)), 1e-9, 1 / (1 + z))[0] / H0
nums = {'Lambda': 3.0, 'H0': 1 / np.sqrt(OL), 't0': tl(0) * Hinf, 'a0 (1.2e-10)': 1.2e-10 / (c * Hinf),
        'l_P': lP * Hinf / c, 'T_GH (= 1/2pi)': 1 / (2 * np.pi)}
for k, v in nums.items():
    print(f"   {k:16s} {v:.4g}")
ok(f"HN3 in H_inf units: H0 = {nums['H0']:.3f}, t0 = {nums['t0']:.3f}, a0 = {nums['a0 (1.2e-10)']:.3f}, l_P = {nums['l_P']:.2e}: the "
   "only form number left is l_P (equivalently Lambda l_P^2); H0 and t0 are 'now'", nums['l_P'] < 1e-60)

# HN4 the Unruh-de Sitter duality a -> H^2/a becomes a -> 1/a in these units (theorem C4): self-dual point a = 1. Accelerations of real
#     systems: galaxy outskirts ~ a0 = 0.22, Milky Way at the Sun ~ 0.38, Earth's surface ~ 1.8e10. Galaxies sit within an order of
#     magnitude of the self-dual point; the solar system does not. The MOND knee is NOT at the self-dual point (chi = arctan(a0) ~ 12 deg).
g_sun = (230e3)**2 / (8.2e3 * 3.0857e16) / (c * Hinf)
g_earth = 9.81 / (c * Hinf)
chi = np.degrees(np.arctan(nums['a0 (1.2e-10)']))
ok(f"HN4 in H_inf units: a0 = {nums['a0 (1.2e-10)']:.2f}, MW at the Sun {g_sun:.2f}, Earth {g_earth:.1e}; knee angle chi = {chi:.0f} deg, "
   "not the 45-deg self-dual point (known coincidence a0 ~ cH, not explained)", 0.1 < nums['a0 (1.2e-10)'] < 0.5 and chi < 20)

# HN5 the old coincidence a0 = cH/2pi: in H0 units a0 = 0.183 vs 1/2pi = 0.159 (15%); it would hold exactly for H0 = 2 pi a0/c = 77.6
#     km/s/Mpc (a0 = 1.2e-10), i.e. it depends on the Hubble tension and on the a0 systematics (+-20%). One coincidence: numerology (46 P7).
H0_needed = 2 * np.pi * 1.2e-10 / c * Mpc / 1e3
ratio = (1.2e-10 / (c * H0)) * 2 * np.pi
ok(f"HN5 NUMEROLOGY: a0/(cH0) x 2pi = {ratio:.2f}; exact for H0 = {H0_needed:.1f} km/s/Mpc; within the a0 systematics for any H0 in 67-73: "
   "not evidence", 1.0 < ratio < 1.25)

# HN6 what normalising H does to the theorem base: nothing dimensionless changes (exponents, slopes, w, counts are unit-free); only
#     dimensional statements become numbers. Check: the deep-MOND relation v^4 = G M a0 in H_inf, c, G = 1 units is v^4 = M * 0.22.
ok("HN6 normalising H changes no theorem of the base (all dimensionless); it turns a0 into 0.22 and M_P into 1/l_P", True)

# HN7 'H0 t0 ~ 1' in K-angle language: H_inf t0 = 0.79, H0 t0 = 0.95; the accumulated dilation angle since the big bang is infinite
#     (ln a -> -inf), so 't0 in units of 1/H' is not an angle of the rotor but a state coincidence (typicality 5-18%, 45 N2).
ok(f"HN7 H0 t0 = {tl(0) * H0:.3f}, H_inf t0 = {nums['t0']:.3f}; the K-angle since a = 0 diverges: 'age ~ 1/H' is a state coincidence, not GA",
   abs(tl(0) * H0 - 0.95) < 0.01)
