# 108: the self-interaction axiom of doc 107 at large scales, and whether the framework's own scalar coupling m(phi) can supply it. SS1-SS3.
# Units SI. Run: ~/.venvs/jev/bin/python ga_si_scales_check.py
import numpy as np

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

G = 6.674e-11; Msun = 1.989e30
h, Om = 0.674, 0.315
H0 = 100 * h * 1e3 / 3.0857e22
rho_m0 = Om * 3 * H0**2 / (8 * np.pi * G)
sm = 0.13 * 1e-4 / 1e-3                                          # 0.13 cm^2/g -> m^2/kg (doc 107, upper of the two medians)

# SS1 cosmic background: Gamma/H = rho_m (s/m) v / H with v the largest relevant velocity (bulk flows ~ 600 km/s today, growing as D a ~ a^(1/2)
#     in the matter era; the thermal dispersion of cold dust is far smaller). Upper bound over 0 < z < 1000.
def ratio(z):
    a = 1 / (1 + z); H = H0 * np.sqrt(Om * a**-3 + 1 - Om)
    v = 6e5 * min(1.0, a**0.5)
    return rho_m0 * a**-3 * sm * v / H
zs = np.array([0, 1, 10, 100, 1000])
r = [ratio(z) for z in zs]
ok("SS1 collisions in the cosmic background: Gamma/H = " + ", ".join(f"z={z}: {x:.1e}" for z, x in zip(zs, r))
   + " (an overestimate: coherent bulk flows do not collide; the relative thermal speed of cold dust is far smaller) -- the medium stays dust "
   "on linear scales; growth and S8 (doc 86) are unchanged", max(r) < 0.1)

# SS2 inside haloes Gamma t_age ~ 1 by construction (doc 107): the axiom acts only where the density is ~1e8 times the mean.
rho_core = 10**-1.3 * Msun / (3.0857e16)**3
ok(f"SS2 core density / mean matter density today = {rho_core / rho_m0:.1e}: self-interaction matters only in halo cores", rho_core / rho_m0 > 1e5)

# SS3 scattering mediated by the framework's scalar (dust coupling m(phi) = m0 e^(beta phi), paper1): between two dust elements of mass m
#     the exchange adds 2 beta^2 G m^2 / r to gravity (82 CC3). A long-range 1/r force gives Rutherford scattering; momentum-transfer
#     cross-section sigma_T ~ 8 pi (G_eff m / v^2)^2 ln(Lambda) with G_eff = G(1 + 2 beta^2). Per unit mass sigma_T/m ~ 8 pi G_eff^2 m ln(Lambda)/v^4.
#     The element mass needed for s/m = 0.13 cm^2/g at v = 115 km/s (ln Lambda ~ 10):
v = 115e3; lnL = 10.0
for beta in (0.03, 1.0):
    Geff = G * (1 + 2 * beta**2)
    m_need = sm * v**4 / (8 * np.pi * Geff**2 * lnL)
    print(f"   beta = {beta}: element mass needed {m_need / Msun:.1e} Msun")
ok("SS3 the framework's scalar exchange has gravitational strength: s/m ~ 0.1 cm^2/g at 115 km/s would need dust elements of ~1e5-1e6 Msun "
   "(two-body relaxation of the halo itself, i.e. not cold dust); the velocity dependence of the self-interaction cannot come from m(phi)",
   sm * v**4 / (8 * np.pi * (G * (1 + 2))**2 * lnL) / Msun > 1e3)
