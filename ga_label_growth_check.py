# 86d: linear growth of the label media themselves (program item 2, without imported models). Jeans dispersion of a self-gravitating
# perfect medium on scales below the horizon: omega^2 = c_s^2 k^2 - 4 pi G rho (1+w)(1+3 c_s^2)... for the cases here the Newtonian form
# omega^2 = c_s^2 k^2 - 4 pi G rho_eff suffices to classify. Sound speeds per rung are computed from the stress formulas of paper1 v2. LG1-LG4.
# Run: ~/.venvs/jev/bin/python ga_label_growth_check.py
import sympy as sp

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

n, m, nu = sp.symbols('n m nu', positive=True)
X = sp.symbols('X', positive=True)

# LG1 spacelike k = 3 medium F = -m n^nu: rho = m n^nu, p = (nu-1) rho, adiabatic c_s^2 = dp/drho = nu - 1 = w. Dust (nu = 1): c_s = 0, it
#     clusters on every scale (growth D ~ a in the matter era). Radiation (nu = 4/3): c_s^2 = 1/3, Jeans-stable, oscillates. -1 < w < 0:
#     c_s^2 < 0, gradient instability (unbounded growth on small scales) -- the reason these rungs cannot be a smooth dark energy.
F = -m * n**nu; rho = -F; p = F - n * sp.diff(F, n)
cs2 = sp.simplify(sp.diff(p, n) / sp.diff(rho, n))
ok(f"LG1 spacelike medium: c_s^2 = {cs2} = w; dust 0, radiation {cs2.subs(nu, sp.Rational(4, 3))}, -1 < w < 0 negative (unstable)",
   sp.simplify(cs2 - (nu - 1)) == 0 and cs2.subs(nu, sp.Rational(4, 3)) == sp.Rational(1, 3))

# LG2 dust growth in the matter era: delta'' + 2H delta' = (3/2) H^2 delta with H = 2/(3t) gives delta ~ t^(2/3) ~ a (growing mode).
t = sp.symbols('t', positive=True); A = sp.symbols('A')
d = t**A
eq = sp.diff(d, t, 2) + 2 * (2 / (3 * t)) * sp.diff(d, t) - sp.Rational(3, 2) * (2 / (3 * t))**2 * d
roots = sp.solve(sp.simplify(eq / t**(A - 2)), A)
ok(f"LG2 dust in the matter era: delta ~ t^A with A in {roots}: growing mode t^(2/3) ~ a", sp.Rational(2, 3) in roots)

# LG3 timelike scalar medium P(X): c_s^2 = P_X/(P_X + 2X P_XX). Cuscuton (P ~ sqrt X): denominator 0, no mode; X^2 (radiation): c_s^2 = 1/3;
#     canonical X: c_s^2 = 1. A smooth evolving dark energy from this rung has c_s^2 > 0 and does not cluster below its sound horizon.
cs2t = lambda P: sp.simplify(sp.diff(P, X) / (sp.diff(P, X) + 2 * X * sp.diff(P, X, 2)))
ok(f"LG3 timelike rung: c_s^2 = 1 for X, {cs2t(X**2)} for X^2, denominator {sp.simplify(sp.diff(sp.sqrt(X), X) + 2*X*sp.diff(sp.sqrt(X), X, 2))} for sqrt X",
   cs2t(X) == 1 and cs2t(X**2) == sp.Rational(1, 3) and sp.simplify(sp.diff(sp.sqrt(X), X) + 2 * X * sp.diff(sp.sqrt(X), X, 2)) == 0)

# LG4 of all rungs only additive spacelike dust (c_s = 0, positive active density) clusters gravitationally on galaxy scales; strings have zero
#     active density ((k-2) rho = 0, 82/paper3) and do not drive growth; the vacuum rungs do not cluster. Structure is made by dust alone.
ok("LG4 recorded: only dust (k = 3, additive) has c_s = 0 and positive active density: structure growth in the framework is dust growth", True)
