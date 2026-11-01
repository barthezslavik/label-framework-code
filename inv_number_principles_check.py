# Principles that set numbers without deriving them (papers/restart/45-number-principles.md): attractor, balance, counting, selection. N1-N5.
# Literature (searched first): Sorkin's everpresent Lambda (astro-ph/0209274; Das-Nasiri-Yazdi 2023, arXiv 2304.03819, 2307.13743);
# Lineweaver-Egan 2007 (astro-ph/0703429): temporal selection of observers explains Omega_m ~ Omega_L (~68%); asymmetric dark matter
# (Kaplan-Luty-Zurek 2009, arXiv 0901.4117; review 1308.0338): Omega_c/Omega_b = m_DM/m_p for equal asymmetries.
# Here the numbers are reproduced with simple inputs; nothing is fitted.
# Run: uv run --with sympy --with numpy --with scipy python inv_number_principles_check.py
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from ga_core import ok

Om0, OL0, h = 0.315, 0.685, 0.674
H0 = 100 * h * 1e3 / 3.0857e22                       # 1/s
tP = 5.391e-44; Gyr = 3.156e16
E = lambda a: np.sqrt(Om0 / a**3 + OL0)
t_of_a = lambda a: quad(lambda x: 1 / (x * E(x)), 1e-9, a)[0] / H0
t0 = t_of_a(1.0)

# N1 counting: Lambda l_P^2 = 3 Omega_L (H0 t_P)^2; Sorkin's estimate Lambda ~ 1/sqrt(V4) with V4 ~ (c t0)^4 gives Lambda l_P^2 ~ (t_P/t0)^2.
lam_obs = 3 * OL0 * (H0 * tP)**2
lam_sorkin = (tP / t0)**2
ok(f"N1 Lambda l_P^2 = {lam_obs:.2e}; Sorkin (t_P/t0)^2 = {lam_sorkin:.2e}; ratio {lam_obs / lam_sorkin:.1f}: the 10^-122 is the age in Planck units "
   "(order of magnitude right; the O(1) factor and the sign are not fixed)", 0.1 < lam_obs / lam_sorkin < 10)

# N2 selection in time (Lineweaver-Egan type): observers appear a delay tau after star formation (Madau-Dickinson SFR for z > 0, exponential
#    decline after today). Fraction of observers who see Omega_L/Omega_m at least as close to 1 as we do (|ln r| <= |ln r0|), and who see
#    H t within 5% of 1, for several delays and future decline times.
def a_of_t(t):
    return brentq(lambda a: t_of_a(a) - t, 1e-4, 50)
def sfr_z(z):
    return 0.015 * (1 + z)**2.7 / (1 + ((1 + z) / 2.9)**5.6)
ts = np.linspace(0.2, 60, 900) * Gyr
avals = np.array([a_of_t(t) for t in ts])
psi0 = sfr_z(0)
def psi(t, a, tau_f):
    return sfr_z(1 / a - 1) if a <= 1 else psi0 * np.exp(-(t - t0) / (tau_f * Gyr))
r0 = OL0 / Om0
res = {}
for tau in (2.0, 4.5, 8.0):
    for tau_f in (3.0, 10.0):
        sf = np.array([psi(t, a, tau_f) for t, a in zip(ts, avals)])
        tobs = ts + tau * Gyr                            # observer appears tau after its star formed
        aobs = np.array([a_of_t(t) for t in tobs[tobs < ts[-1]]])
        w = sf[:len(aobs)]
        r = (OL0 / Om0) * aobs**3
        Ht = np.array([E(a) * H0 * t for a, t in zip(aobs, tobs[:len(aobs)])])
        p_close = np.sum(w * (np.abs(np.log(r)) <= abs(np.log(r0)))) / np.sum(w)
        p_sync = np.sum(w * (np.abs(Ht - 1) <= abs(E(1) * H0 * t0 - 1))) / np.sum(w)
        res[(tau, tau_f)] = (p_close, p_sync)
        print(f"   delay {tau:3.1f} Gyr, future SFR e-fold {tau_f:4.1f} Gyr: P(Omega ratio as close as ours) = {p_close:.2f}, P(|Ht-1| <= {abs(E(1)*H0*t0-1):.3f}) = {p_sync:.2f}")
pc = [v[0] for v in res.values()]; ps = [v[1] for v in res.values()]
ok(f"N2 selection: observers who see Omega_L/Omega_m as close to 1 as we do: {min(pc):.2f}-{max(pc):.2f} (Lineweaver-Egan ~0.68): 'now' is typical for observers",
   min(pc) > 0.3)
print(f"   synchronicity H t ~ 1 to 5%: {min(ps):.2f}-{max(ps):.2f} of observers")

# N3 balance at birth (asymmetric dark matter): equal particle asymmetries give Omega_c/Omega_b = m_DM/m_p -> m_DM = 5.36 m_p ~ 5 GeV.
ratio = 0.120 / 0.0224
ok(f"N3 Omega_c/Omega_b = {ratio:.2f}; equal number asymmetries -> m_DM = {ratio * 0.938:.2f} GeV: the ratio becomes a mass ratio (still an input)",
   4 < ratio * 0.938 < 6)

# N4 attractors (time as stretching): in log time the flow of r = Omega_L/Omega_m is d ln r / d ln a = 3 (a pure stretch): matter is the
#    repelling fixed half, Lambda the attracting one; H t -> 2/3 in the matter era is a fixed point (EdS homothety); the virial ratio 2T/|V| = 1.
ln_a = np.linspace(-6, 3, 200)
r_flow = (OL0 / Om0) * np.exp(3 * ln_a)
OL_flow = r_flow / (1 + r_flow)
early = E(1e-3) * H0 * t_of_a(1e-3)
import sympy as sp
GM, rr = sp.symbols('GM r', positive=True); v2 = GM / rr
vir = sp.simplify(2 * (v2 / 2) / (GM / rr))
ok(f"N4 Omega_L(ln a) runs from {OL_flow[0]:.1e} to {OL_flow[-1]:.4f} (attractor 1); H t at a = 1e-3: {early:.4f} (fixed point 2/3); "
   f"virial 2T/|V| = {vir} on a circular Kepler orbit", vir == 1 and abs(early - 2 / 3) < 1e-3 and OL_flow[-1] > 0.999 and OL_flow[0] < 1e-6)

# N5 the 'middle' of the stretch flow: Omega_L = Omega_m at a_eq = (Om0/OL0)^(1/3) is the null direction of the split between the two
#    fixed halves; observers (N2) are concentrated around it because star formation peaks during the transit.
a_eq = (Om0 / OL0)**(1 / 3)
peak_a = avals[np.argmax([psi(t, a, 10.0) for t, a in zip(ts, avals)])]
ok(f"N5 a_eq = {a_eq:.3f} (z = {1 / a_eq - 1:.2f}); star formation peaks at a = {peak_a:.3f} (z = {1 / peak_a - 1:.2f}); observers arrive a few Gyr later, "
   "near the middle", 0.2 < peak_a < 0.5 and 0.6 < a_eq < 0.9)
