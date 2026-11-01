# Why a0 ~ 1/t: the matter-era background is self-similar under (t, r) -> lambda (t, r), which is the p = 0 element of the scaling group
# (velocities invariant, accelerations ~ 1/lambda) -- the same subgroup as Milgrom's deep-MOND scaling and the self-dual 'middle' of 35.
# An acceleration set by that background must scale as 1/t; a0 t is its velocity invariant (papers/restart/46-clock-combos.md, "GA reason").
# Q1-Q5. Known: self-similarity of Einstein-de Sitter (Carr-Coley self-similar cosmologies).
# Run: uv run --with sympy --with numpy --with scipy python ga_a0_clock_check.py
import numpy as np
import sympy as sp
from scipy.integrate import quad
from ga_core import ok

t, x, lam, H = sp.symbols('t x lambda H', positive=True)

# Q1 EdS ds^2 = dt^2 - t^(4/3) dx^2 is homothetic: t -> lam t, x -> lam^(1/3) x multiplies it by lam^2; physical r = t^(2/3) x -> lam r.
a_eds = t**sp.Rational(2, 3)
g_tt, g_xx = 1, -a_eds**2
dt_, dx_ = sp.symbols('dt dx')
ds2 = g_tt * dt_**2 + g_xx * dx_**2
ds2_scaled = ds2.subs({t: lam * t, dt_: lam * dt_, dx_: lam**sp.Rational(1, 3) * dx_}, simultaneous=True)
r_phys = a_eds * x
r_scaled = r_phys.subs({t: lam * t, x: lam**sp.Rational(1, 3) * x}, simultaneous=True)
ok(f"Q1 EdS: ds^2 -> {sp.simplify(ds2_scaled / ds2)} ds^2 (homothety); physical r -> {sp.simplify(r_scaled / r_phys)} r, t -> lam t: (t, r) -> lam (t, r)",
   sp.simplify(ds2_scaled / ds2 - lam**2) == 0 and sp.simplify(r_scaled / r_phys - lam) == 0)

# Q2 (t, r) -> lam (t, r) is mu = 1 in the scaling group (35): velocities invariant, accelerations -> a/lam, so a t is invariant (a velocity).
mu = 1
acc_w = sp.Rational(mu**2, 1) / lam
ok(f"Q2 p = 0 element: v -> v, a -> {acc_w} a, a t -> {sp.simplify(acc_w * lam)} (a t is a velocity invariant): a background-set acceleration scales as 1/t",
   sp.simplify(acc_w * lam) == 1)

# Q3 Lambda breaks it: a(t) = sinh^(2/3)(3/2 H t) is not homothetic -- a(lam t)/a(t) depends on t (cosmic_dilator M4 in density language).
a_l = sp.sinh(sp.Rational(3, 2) * H * t)**sp.Rational(2, 3)
dep = sp.simplify(sp.diff(a_l.subs(t, lam * t) / a_l, t))
ok(f"Q3 LCDM: d/dt[a(lam t)/a(t)] != 0 (e.g. {sp.N(dep.subs({lam: 2, H: 1, t: sp.Rational(1, 2)}), 4)}): the homothety and the clock law are matter-era features",
   sp.N(dep.subs({lam: 2, H: 1, t: sp.Rational(1, 2)})) != 0)

# Q4 in the matter era 1/t and H differ by the constant H t = 2/3, so 'a0 ~ 1/t' and 'a0 ~ H' coincide there; they separate only when Lambda
#    acts (z < ~1.5): H t runs from ~0.66 to 0.95. The MUSE range 0.33 < z < 1.44 is exactly where they separate -- and the data pick 1/t
#    (inv_clock_combos_check P4: 1.0 sigma vs 2.7 sigma).
Om, OL = 0.315, 0.685
E = lambda a: np.sqrt(Om / a**3 + OL)
Ht = lambda z: E(1 / (1 + z)) * quad(lambda y: 1 / (y * E(y)), 1e-9, 1 / (1 + z))[0]
vals = {z: Ht(z) for z in (5, 2, 1.44, 0.33, 0)}
ok(f"Q4 H t: " + ", ".join(f"z={z}: {v:.3f}" for z, v in vals.items()) + " -> 1/t and H coincide (x 2/3) at high z, differ by ~30% over the MUSE range",
   abs(vals[5] - 2 / 3) < 0.01 and vals[0] - vals[1.44] > 0.15)

# Q5 the p = 0 element is the self-dual line of the split-complex group (35 S6: swap rotor <-> dilation sends p -> -p, p = 0 fixed) and the
#    deep-MOND scaling (32): cosmic time in the matter era, Milgrom's symmetry and the 'middle' are one subgroup.
# (Q5 is a reading of 35 S6, not a computation: p -> -p fixes p = 0.)
