# 76: baryons and dust in one mixing. Inside the framework both are label media with their own phase-space SDiff (each fine-grained
# phase density is conserved separately); they share only the potential and the total energy. Maximal coarse-grained entropy
# (non-degenerate limit) under these constraints. BL1-BL5. Structural; no data.
# Run: ~/.venvs/jev/bin/python ga_two_species_mixing_check.py
import sympy as sp

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

f, eta, eps, al, be = sp.symbols('f eta epsilon alpha beta', positive=True)
# BL1 pointwise variation of  -(f/eta) ln(f/eta)  - alpha f - beta eps f  (entropy of one species in units of its phase level eta,
#     its mass constraint alpha, the shared energy constraint beta, eps = v^2/2 + Phi per unit mass): f ~ exp(-beta eta eps).
L = -(f / eta) * sp.log(f / eta) - al * f - be * eps * f
fsol = sp.solve(sp.diff(L, f), f)[0]
ok(f"BL1 maximal mixing per species: f = {sp.simplify(fsol)} -> f ~ exp(-beta eta eps)",
   sp.simplify(sp.diff(sp.log(fsol), eps) + be * eta) == 0)

# BL2 dispersion: f ~ exp(-v^2/(2 sigma^2)) with sigma^2 = 1/(beta eta): sigma_i^2 eta_i = 1/beta is shared; sigma_b^2/sigma_d^2 = eta_d/eta_b.
eb, ed = sp.symbols('eta_b eta_d', positive=True)
sig2 = lambda e: 1 / (be * e)
ok(f"BL2 sigma^2 eta = 1/beta for every species; sigma_b^2/sigma_d^2 = {sp.simplify(sig2(eb) / sig2(ed))}",
   sp.simplify(sig2(eb) * eb - sig2(ed) * ed) == 0 and sp.simplify(sig2(eb) / sig2(ed) - ed / eb) == 0)

# BL3 densities in the common potential: rho_i ~ exp(-Phi/sigma_i^2), so rho_b ~ rho_d^(eta_b/eta_d): baryons trace dust (rho_b/rho_d
#     constant) iff eta_b = eta_d.
Phi = sp.symbols('Phi', real=True); Ab, Ad = sp.symbols('A_b A_d', positive=True)   # normalisations fixed by the masses (review 2026-10-22)
rb, rd = Ab * sp.exp(-Phi / sig2(eb)), Ad * sp.exp(-Phi / sig2(ed))
ok(f"BL3 rho_b = A_b (rho_d/A_d)^(eta_b/eta_d): {sp.simplify(rb - Ab * (rd / Ad)**(eb / ed)) == 0}; trace each other iff eta_b = eta_d",
   sp.simplify(rb - Ab * (rd / Ad)**(eb / ed)) == 0)

# BL4 in the dust plateau, Phi = 2 sigma_d^2 ln r (rho_d ~ 1/r^2), the baryons have rho_b ~ r^(-2 eta_b/eta_d): the baryon profile is a
#     power set by the ratio of phase-space labels.
r = sp.symbols('r', positive=True)
expo = sp.simplify(sp.diff(sp.log(sp.exp(-2 * sig2(ed) * sp.log(r) / sig2(eb))), r) * r)
ok(f"BL4 baryons in the dust plateau: rho_b ~ r^({expo})", sp.simplify(expo + 2 * eb / ed) == 0)

# BL5 what follows for the baryon link: v_flat^2 = 2 sigma_d^2 = 2 sigma_b^2 eta_b/eta_d -- the halo dispersion is tied to the baryons
#     only through the ratio eta_b/eta_d, a label (initial data). The BTFR scaling M_b ~ v^4 needs, in addition, a rule for the radius
#     out to which baryons are collected (k = 2 level set -> 4, k = 3 -> 3; 71 FC12); mixing does not supply it. Dissipation of baryons
#     breaks their phase-space SDiff (entropy by shear, doc 58), so the shared beta holds only before cooling.
ok("BL5 recorded: shared beta ties sigma_d to sigma_b via the label eta_b/eta_d; no M_b ~ v^4 without a radius rule; link set before baryons dissipate",
   True)
