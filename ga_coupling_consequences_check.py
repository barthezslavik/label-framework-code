# 82: consequences and limits of the dust-scalar coupling of 80 (TL5). CC1-CC5. Structural, with one Solar-System number.
# Run: ~/.venvs/jev/bin/python ga_coupling_consequences_check.py
import sympy as sp

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

tau = sp.symbols('tau'); ph = sp.Function('phi')(tau); phd = sp.diff(ph, tau)
# CC1 scope of TL5: with proper time tau, any L(phi, dphi/dtau) dtau is reparametrisation invariant. Terms LINEAR in dphi/dtau are total
#     derivatives (TL5); terms of higher order, e.g. gamma(phi) (dphi/dtau)^2 (a disformal coupling), are NOT: their Euler-Lagrange term is
#     non-zero. So TL5 holds at lowest order in derivatives of phi; derivative (disformal) couplings are allowed and not covered.
ga = sp.Function('gamma')
Lq = ga(ph) * phd**2
ELq = sp.simplify(sp.diff(sp.diff(Lq, phd), tau) - sp.diff(Lq, ph))
ok(f"CC1 gamma(phi) (dphi/dtau)^2 has EL term {ELq} != 0: derivative (disformal) couplings survive; TL5 = lowest derivative order only",
   ELq != 0)

# CC2 the same argument for every additive rung: an element of world-volume dimension d = 4 - k with action -int m(phi) dV_d; a term linear in
#     the derivative of phi needs a vector on the world volume, which exists only for d = 1 (world lines). For d >= 2 the lowest-order coupling
#     is again a field-dependent tension m(phi) (recorded; the scalar dV_d gives no invariant linear in d phi for d >= 2).
ok("CC2 recorded: for strings and walls the lowest-order coupling is a field-dependent tension m(phi) (no invariant linear in d phi for d >= 2)", True)

# CC3 fifth force between dust elements: canonical massless phi (units 8 pi G = 1, so G = 1/(8 pi)), source beta m per element
#     (m(phi) = m0 e^(beta phi)); static field phi = -beta m1/(4 pi r); scalar force beta^2 m1 m2/(4 pi r^2); gravity m1 m2/(8 pi r^2).
#     Ratio 2 beta^2: G_eff,dust = G (1 + 2 beta^2) between dust elements; between dust and uncoupled baryons the force is G.
b, m1, m2, r = sp.symbols('beta m1 m2 r', positive=True)
Fs = b**2 * m1 * m2 / (4 * sp.pi * r**2); Fg = m1 * m2 / (8 * sp.pi * r**2)
ok(f"CC3 scalar/gravity force ratio between dust elements = {sp.simplify(Fs / Fg)}: G_eff = G (1 + 2 beta^2); dust-baryon force unchanged",
   sp.simplify(Fs / Fg - 2 * b**2) == 0)

# CC4 baryons obey the same axioms before they dissipate, so the theorem also allows m_b(phi). A universal coupling beta_b of the visible
#     sector is a Brans-Dicke scalar with 2 beta_b^2 = 1/(2 omega + 3); Cassini |gamma - 1| < 2.3e-5 (Bertotti, Iess, Tortora 2003) with
#     gamma = (1 + omega)/(2 + omega) gives beta_b < 2.4e-3. A coupling to dust only (beta ~ 0.1 in 80, TL6) is therefore an extra premise.
om = sp.symbols('omega', positive=True)
gam = (1 + om) / (2 + om)
om_c = sp.solve(sp.Eq(1 - gam, sp.Rational(23, 10**6)), om)[0]
beta_c = sp.sqrt(1 / (2 * (2 * om_c + 3)))
ok(f"CC4 Cassini bound: omega > {sp.N(om_c, 5)}, universal beta_b < {sp.N(beta_c, 3)}; coupling to dust only is an assumption",
   abs(float(beta_c) - 2.4e-3) < 1e-4)

# CC5 null labels are not covered: for a null gradient n = 0 and F(n) degenerates; null dust (a beam of radiation) needs another action.
ok("CC5 recorded: null label gradients (null dust) lie outside the F(n) classification", True)
