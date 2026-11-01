# 85: program item 1, premise P1 of paper3 (common dark-to-baryon ratio at the plateau onset). PA1-PA4.
# Run: ~/.venvs/jev/bin/python ga_p1_check.py
import numpy as np
import sympy as sp

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

# PA1 homology: a ratio of two masses measured inside a radius fixed in units of the system's own scale (r_t = x_t r_c) is dimensionless and
#     invariant under the scaling group (x -> lam x, masses -> w M with one common weight). Galaxies on one orbit (P2, doc 84) therefore share
#     every such ratio, f_t included, provided baryons and dust carry the same mass weight -- i.e. the baryon share of the label measure is
#     the same function of position in every system (adiabatic label data plus homologous settling).
lam, w, Md, Mb = sp.symbols('lambda w M_d M_b', positive=True)
ok("PA1 f_t = M_d(x_t r_c)/M_b is invariant when both masses carry the same weight w and radii scale with r_c",
   sp.simplify((w * Md) / (w * Mb) - Md / Mb) == 0)

# PA2 the reformulation that removes the construction artefact: beyond r_t the speed is set by the halo, v^2 = 2 sigma^2 (isothermal plateau);
#     with M_bar = eps M_c (M_c the halo's characteristic mass on the orbit, eps a retained baryon fraction) the slope is
#     s_bar = s_halo + d ln eps / d ln v. P1 <=> eps independent of the system.
v, eps0, g, sh = sp.symbols('v epsilon_0 gamma s_h', positive=True)
Mc = v**sh; eps = eps0 * v**g
s_bar = sp.simplify(sp.diff(sp.log(eps * Mc), v) * v)
ok(f"PA2 M_bar = eps M_c with eps ~ v^gamma: s_bar = {s_bar} = s_halo + gamma; P1 <=> gamma = 0", sp.simplify(s_bar - sh - g) == 0)

# PA3 why the BTFR has no surface-brightness residuals in this reading: v is set by the halo plateau, so at fixed M_bar it does not depend
#     on how concentrated the baryons are: d ln v/d ln R_d = 0. If instead the speed were set by the baryons (Newtonian disk, v^2 ~ G M_bar/R_d),
#     d ln v/d ln R_d = -1/2, and high- and low-surface-brightness galaxies would separate.
Rd = sp.symbols('R_d', positive=True); G = sp.symbols('G', positive=True)
v_halo = sp.sqrt(2) * sp.Symbol('sigma'); v_disk = sp.sqrt(G * Mb / Rd)
ok(f"PA3 d ln v/d ln R_d: halo-set {sp.diff(sp.log(v_halo), Rd) * Rd}, baryon-set {sp.simplify(sp.diff(sp.log(v_disk), Rd) * Rd)}",
   sp.diff(sp.log(v_halo), Rd) == 0 and sp.simplify(sp.diff(sp.log(v_disk), Rd) * Rd + sp.Rational(1, 2)) == 0)

# PA4 reduction to data: with s_halo = 12/(1 - n_eff) = 3.57..3.87 (doc 84, galaxy masses 1e10..1e12) and the measured s_bar = 3.85 +- 0.09
#     (systematic range 3.5-4.0), the required retention trend is gamma = s_bar - s_halo in [-0.02, 0.28] (central) -- the retained baryon
#     fraction must be nearly the same across galaxy masses (at most ~ v^0.3, i.e. ~ M^0.08). Whether it is, is a question for the data on
#     baryon budgets, not for the framework.
s_obs, s_h_lo, s_h_hi = 3.85, 3.57, 3.87
gam = (s_obs - s_h_hi, s_obs - s_h_lo)
ok(f"PA4 required gamma = d ln eps/d ln v in [{gam[0]:.2f}, {gam[1]:.2f}] (i.e. eps ~ M^{gam[0]/3.85:.2f}..M^{gam[1]/3.85:.2f}): P1 reduced to the data on retained baryon fractions",
   gam[0] < 0.01 and gam[1] < 0.3)

# PA5 data (Gentile, Famaey, Zhao, Salucci 2009, Nature 461, 627; arXiv:0909.5203): inside one dark-halo scale length the mean surface
#     density of luminous matter is constant across 14 magnitudes and the Hubble sequence, so the luminous-to-dark ratio inside r0 is the same
#     although the total ratio is not -- this is P1 at a radius fixed in units of r_c (PA1). Contested for early-type galaxies (Napolitano,
#     Romanowsky, Tortora 2010), which also leave the k = 2 population at high acceleration (doc 78, DU7).
ok("PA5 recorded: constant luminous-to-dark ratio inside r0 (Gentile et al. 2009) = P1 at fixed x_t; early types differ, as for the slope", True)

# PA6 P1 derived under the framework's native assumptions -- adiabatic label data (the baryon share of the label measure is the cosmic one
#     everywhere, a conserved label ratio, base F3), baryons settling at fixed specific angular momentum with a spin label independent of mass
#     (R_d = q r0, doc 77 AM2), and no mass-dependent loss of baryons (no feedback; feedback is outside the framework). Model: cored halo
#     rho ~ 1/(r^2 + r0^2), exponential disk with M_bar = f M_c (M_c = rho0 r0^3, the halo's characteristic mass on the orbit). Along the
#     orbit (masses scaled by 10^3) the luminous-to-dark ratio inside r0 is the same; a lognormal spin label (sigma_ln = 0.5) gives its scatter.
import numpy as np
def ratio_inside_r0(r0, rho0, f, q):
    Md = 4 * np.pi * rho0 * r0**3 * (1 - np.pi / 4)                       # int_0^r0 4 pi r^2 rho0 r0^2/(r^2+r0^2) dr
    Mb = f * rho0 * r0**3; Rd = q * r0
    Mb_in = Mb * (1 - (1 + r0 / Rd) * np.exp(-r0 / Rd))                    # exponential disk mass inside R = r0
    return Mb_in / Md
base = [ratio_inside_r0(r0, rho0, 0.5, 0.5) for r0, rho0 in ((1.0, 1.0), (10.0, 0.1), (100.0, 0.01))]   # common surface density rho0 r0
rng = np.random.default_rng(1)
qs = 0.5 * np.exp(rng.normal(0, 0.5, 20000))
sc = np.std(np.log10([ratio_inside_r0(1.0, 1.0, 0.5, q) for q in qs]))
ok(f"PA6 along the orbit (r0 x10, x100 at fixed rho0 r0) the ratio inside r0 = {[round(b, 4) for b in base]} (identical); spin-label scatter 0.5 in ln -> {sc:.2f} dex",
   max(base) - min(base) < 1e-12 and 0.05 < sc < 0.6)
