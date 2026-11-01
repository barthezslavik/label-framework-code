# 121: G9 of PREDICTIONS.md -- Crater II under the framework's dynamics: Newton with the strong equivalence principle (no external field
# effect, doc 79) and dust halos. Data (literature, not computed here): sigma_los = 2.7 +0.5/-0.4 km/s (Fu et al. 2019), r_h = 1.07 kpc,
# L_V ~ 1.6e5 Lsun (Torrealba et al. 2016), D = 117 kpc, small pericentre from Gaia orbits (Fritz et al. 2018; taken as 20-40 kpc).
# Computed: (i) the dispersion from the stars alone (Wolf estimator M(<r12) = 3 sigma^2 r12/G, r12 = 4/3 r_h); (ii) the mean density the
# measured dispersion implies inside r12; (iii) the Jacobi survival density 3 rho_host(<r_p) for a flat 220 km/s host; (iv) the pericentre
# above which a halo of today's density would be untouched (rewritten after review r1: a stripped remnant may re-equilibrate). The tidal-track shapes of stripped halos (N-body literature) are not used. CR1-CR5.
# Run: ~/.venvs/jev/bin/python ga_crater2_tides_check.py
import numpy as np

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

G, Msun, pc, kpc = 6.674e-11, 1.989e30, 3.086e16, 3.086e19
rh, Lv, ML, sig, D = 1.07, 1.6e5, 2.0, 2.7, 117.0
r12 = 4 / 3 * rh * kpc
Mst = ML * Lv
s_st = np.sqrt(G * Mst / 2 * Msun / (3 * r12)) / 1e3
ok(f"CR1 stars alone (M/L = {ML}): sigma = {s_st:.2f} km/s against the measured {sig}: Newton with the SEP needs a dark halo, "
   f"M(<r12) = {3 * (sig * 1e3)**2 * r12 / G / Msun:.2e} Msun, M/L(<r12) = {3 * (sig * 1e3)**2 * r12 / G / Msun / (Lv / 2):.0f}", s_st < 1)
rho_in = 3 * 3 * (sig * 1e3)**2 * r12 / G / (4 * np.pi * r12**3)
rho_in_pc = rho_in / (Msun / pc**3)
ok(f"CR2 mean density inside r12 = {r12 / kpc:.2f} kpc implied by sigma = {sig}: {rho_in_pc:.1e} Msun/pc^3 ({rho_in_pc * (2.0 / 2.7)**2:.1e}-{rho_in_pc * (3.2 / 2.7)**2:.1e} over the error)",
   rho_in_pc < 2e-3)
vh = 220e3
res = []
for rp in (20, 30, 40, 60):
    rho_host = 3 * vh**2 / (4 * np.pi * G * (rp * kpc)**2) / (Msun / pc**3)
    res.append((rp, 3 * rho_host, 3 * rho_host / rho_in_pc))
print("   Jacobi survival density 3 rho_host(<r_p): " + "; ".join(f"r_p {a} kpc: {b:.1e} Msun/pc^3 ({c:.0f}x the Crater II density)" for a, b, c in res))
ok(f"CR3 at every pericentre 20-60 kpc the measured density is below the survival density by {min(r[2] for r in res):.0f}-{max(r[2] for r in res):.0f}x: "
   "at past pericentres the stars at r12 lay outside the tidal radius of any halo of today's density", all(r[2] > 1 for r in res))
rp_eq = np.sqrt(9 * vh**2 / (4 * np.pi * G * rho_in)) / kpc
ok(f"CR4 an untouched halo of today's density would need every pericentre >= {rp_eq:.0f} kpc (present distance {D} kpc); with Gaia orbits "
   "at 20-40 kpc the low density is the result of tidal mass loss -- whether the remnant is in equilibrium now is not decided here "
   "(simulations put the inner ~1 kpc near equilibrium ~1 Gyr after pericentre)", rp_eq > 60)
frac = (rho_in_pc / res[1][1])**(1 / 3)
ok(f"CR5 a dark core dense enough to survive pericentre (r_p = 30 kpc) can fill at most {frac:.2f} of r12 ({frac * r12 / kpc:.2f} kpc) without "
   "exceeding the measured mean density; the initial mass is not set, so the amount of mass lost does not follow (review r2). Tidal streams of "
   "Crater II are observed (MNRAS 535, 443, 2024): agreement, not a distinction from MOND, which does not forbid stripping", frac < 1)
