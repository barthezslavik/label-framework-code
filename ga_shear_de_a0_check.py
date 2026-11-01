# 61: hypotheses 'shear explains dark energy' and 'shear explains a0' (chat 2026-10-20), each attacked. SH1-SH6.
# Literature (from memory, network down, not verified): Buchert 2000 (averaged Raychaudhuri, backreaction), Rasanen; Green-Wald 2011
# (small perturbations cannot mimic dark energy in GR); viscous cosmology (Zimdahl, Brevik; Velten-Schwarz 2011 constraints).
# Run: uv run --with sympy --with numpy --with scipy python ga_shear_de_a0_check.py
import numpy as np
import sympy as sp
from ga_core import ok

t, a, Hs, sig, th, om, rho, p, G, Lam = sp.symbols('t a H sigma theta omega rho p G Lambda', real=True)

# SH1 global shear (Bianchi I, matter-free shear propagation sigma' = -theta sigma = -3 H sigma): sigma ~ a^-3, its energy ~ sigma^2 ~ a^-6,
#     i.e. w = +1 (stiff), the opposite of dark energy (w = -1).
A = sp.Function('A')(t); S = sp.Function('S')(t)
sol = sp.dsolve(sp.Eq(S.diff(t), -3 * A.diff(t) / A * S), S).rhs
w_shear = sp.Rational(-6, -3) - 1                                        # rho ~ a^{-3(1+w)} with rho ~ a^-6
ok(f"SH1 REFUTED (DE from global shear): sigma = {sol} ~ a^-3, shear energy ~ a^-6, w = {w_shear}: stiff, decelerating", w_shear == 1)

# SH2 local shear in the Raychaudhuri equation: d theta/d tau = -theta^2/3 - 2 sigma^2 + 2 omega^2 - 4 pi G (rho + 3p) + Lambda.
#     Shear enters with -2 sigma^2 < 0: it always focuses (decelerates). Averaged (Buchert): Q = (2/3)(<theta^2> - <theta>^2) - 2<sigma^2>;
#     acceleration from inhomogeneity needs the expansion VARIANCE to beat the shear. REFUTED: shear works against acceleration.
ray = -th**2 / 3 - 2 * sig**2 + 2 * om**2 - 4 * sp.pi * G * (rho + 3 * p) + Lam
d_sig = sp.diff(ray, sig)
ok(f"SH2 REFUTED (DE from local shear): d(theta')/d sigma = {d_sig} (< 0 for sigma > 0): shear decelerates; only expansion variance or "
   "vorticity could accelerate, and Green-Wald (from memory) bounds that for small perturbations", sp.simplify(d_sig + 4 * sig) == 0)

# SH3 FLRW is shear-free: the traceless symmetric part of the Hubble flow v = H x vanishes, so shear viscosity has no effect on the
#     background; only bulk viscosity (the K part) does (60 RH9).
x, y, z = sp.symbols('x y z', real=True)
v = [Hs * x, Hs * y, Hs * z]
Gm = sp.Matrix(3, 3, lambda i, j: sp.diff(v[i], [x, y, z][j]))
shear = sp.simplify((Gm + Gm.T) / 2 - Gm.trace() / 3 * sp.eye(3))
ok(f"SH3 the Hubble flow has shear {shear.tolist()}: shear viscosity is blind to the background; viscous DE must be BULK (the K part)",
   shear == sp.zeros(3, 3))

# SH4 bulk viscosity (K, not shear) as DE: p_eff = p - 3 H zeta; for dust w_eff = -1 needs zeta = rho/(3H) = H/(8 pi G): a viscosity tied
#     to H by hand. Survives as a possibility, known, and constrained by structure formation (Velten-Schwarz, from memory). Not shear.
zeta = sp.solve(sp.Eq(0 - 3 * Hs * sp.Symbol('zeta'), -rho), sp.Symbol('zeta'))[0]
zeta_H = sp.simplify(zeta.subs(rho, 3 * Hs**2 / (8 * sp.pi * G)))
ok(f"SH4 NARROWED (known): w_eff = -1 from bulk viscosity needs zeta = {zeta_H}: tuned to H; this is the K part, not shear", zeta_H == Hs / (8 * sp.pi * G))

# SH5 a0 from galactic shear. For a flat curve the Oort shear rate is A = (v/r - dv/dr)/2 = v/(2r). Hypothesis: the knee (v^2/r = a0) sits
#     where the shear rate equals the cosmic rate, A = H0. Then a0 = 2 H0 v: (i) a0 ~ v, beta = 1, against the measured beta = 0.08 +- 0.15
#     (53); (ii) magnitude for v = 150 km/s.
H0 = 67.4e3 / 3.0857e22
a0_pred = 2 * H0 * 1.5e5
beta_sig = (1 - 0.08) / 0.15
ok(f"SH5 REFUTED (a0 from shear = H0): a0 = 2 H0 v = {a0_pred:.1e} m/s^2 (observed 1.2e-10, x{1.2e-10 / a0_pred:.0f} off) and a0 ~ v "
   f"(beta = 1, {beta_sig:.1f} sigma from the measured 0.08)", a0_pred < 1.2e-11 and beta_sig > 5)

# SH6 variant: the knee where the shear rate equals a0/c (the only rate built from a0 and c). With A = v/(2r) and v^2/r = a0 this forces
#     v = c/2 for every galaxy. REFUTED. No combination of the galactic shear rate with H0 or a0/c gives a universal acceleration:
#     A has the profile weight lambda^(p-1), the knee must be a fixed point of the p = 0 subgroup (51), where A ~ 1/r is not fixed.
vv, r, a0s, c = sp.symbols('v r a0 c', positive=True)
sol_v = sp.solve([sp.Eq(vv / (2 * r), a0s / c), sp.Eq(vv**2 / r, a0s)], [vv, r], dict=True)
ok(f"SH6 REFUTED (A = a0/c): solution {sol_v}: v = c/2, impossible", sol_v and sp.simplify(sol_v[0][vv] - c / 2) == 0)

# SH7 the remaining loophole in Raychaudhuri: vorticity (+2 omega^2) accelerates. For dust, angular momentum conservation gives
#     omega ~ a^-2, so omega^2/H^2 ~ a^-4 / a^-3 = a^-1 in the matter era (and a^-4 with Lambda): it decays, it cannot switch on late.
ap = sp.symbols("a_p", positive=True)
w_ratio_matter = sp.simplify(ap * sp.diff(sp.log((ap**-2)**2 / ap**-3), ap))
ok(f"SH7 REFUTED (late DE from vorticity): omega^2/H^2 ~ a^{w_ratio_matter} in the matter era: decays, cannot produce late acceleration",
   w_ratio_matter == -1)
