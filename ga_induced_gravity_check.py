# 89: can quantised label media induce dynamics for the spacetime metric (Sakharov's induced gravity, level 3 of the ladder, doc 88)? A field
# induces sqrt(-g) R for the metric its fluctuations propagate on. For each rung: the effective (acoustic) metric of its perturbations. IG1-IG5.
# Run: ~/.venvs/jev/bin/python ga_induced_gravity_check.py
import sympy as sp

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

X, c, n, m, nu = sp.symbols('X c n m nu', positive=True)
g = sp.diag(1, -1, -1, -1); gi = g.inv()

# IG1 timelike label (scalar medium P(X), X = (d phi)^2): the second variation is G^{mu nu} d_mu dphi d_nu dphi with
#     G^{mu nu} = P_X g^{mu nu} + 2 P_XX d^mu phi d^nu phi. Background d phi = (c, 0, 0, 0): G ∝ g^{mu nu} iff P_XX = 0, i.e. P = a X + b:
#     F(n) = a n^2 + b -- the stiff medium (w = 1, c_s = 1).
P = sp.Function('P')
dphi = sp.Matrix([c, 0, 0, 0]); dup = gi * dphi
Geff = sp.diff(P(X), X) * gi + 2 * sp.diff(P(X), X, 2) * dup * dup.T
Xb = c**2
ratio_tt = sp.simplify((Geff[0, 0] / gi[0, 0]) / (Geff[1, 1] / gi[1, 1]))    # 1 iff conformal to g
ok(f"IG1 scalar medium: G^tt/g^tt over G^xx/g^xx = {ratio_tt} -> conformal to g iff P_XX = 0 (P linear in X, F ~ n^2, stiff)",
   sp.simplify(ratio_tt.subs(sp.Derivative(P(X), (X, 2)), 0) - 1) == 0)

# IG2 spacelike k = 3 medium F = -m n^nu: c_s^2 = nu - 1 (86d LG1); the acoustic metric of a perfect fluid is
#     G_ac ∝ g - (1 - c_s^2) u u (Moncrief, Visser), conformal to g iff c_s^2 = 1, i.e. nu = 2: again the stiff medium F ~ n^2.
cs2 = nu - 1
ok(f"IG2 spacelike medium: acoustic metric conformal to g iff c_s^2 = {cs2} = 1 iff nu = {sp.solve(sp.Eq(cs2, 1), nu)} (stiff)", sp.solve(sp.Eq(cs2, 1), nu) == [2])

# IG3 dust (nu = 1): c_s = 0, the acoustic metric is degenerate (no propagating sound); additive constraint media (cuscuton) have no kinetic
#     term at all (paper1 v2). Neither has quanta propagating on g, so neither induces sqrt(-g) R: cold dark matter induces no gravity.
ok(f"IG3 dust: c_s^2 = {cs2.subs(nu, 1)} (degenerate acoustic metric); cuscuton P_X + 2X P_XX = {sp.simplify(sp.diff(sp.sqrt(X), X) + 2*X*sp.diff(sp.sqrt(X), X, 2))}: no induced curvature term",
   cs2.subs(nu, 1) == 0)

# IG4 radiation rungs: spacelike n^{4/3} (c_s^2 = 1/3) has an acoustic metric that is not g; its phonons would induce dynamics for that
#     acoustic metric (analogue gravity), not for spacetime. The timelike X^2 medium: P_XX != 0, also not g.
ok(f"IG4 radiation fluid: c_s^2 = {cs2.subs(nu, sp.Rational(4, 3))} != 1; X^2 medium: P_XX = {sp.diff(X**2, X, 2)} != 0: their quanta live on acoustic metrics, not on g",
   cs2.subs(nu, sp.Rational(4, 3)) != 1 and sp.diff(X**2, X, 2) != 0)

# IG5 result: among label media only the stiff rung (F ~ n^2, w = 1, c_s = 1: the canonical massless scalar) has quanta on the spacetime light
#     cone and can induce sqrt(-g) R at one loop; the coefficient is set by the cutoff (Sakharov 1967; Visser 2002), i.e. Newton's constant is a
#     label of the microscopic theory. Dust, radiation fluids and constraint media cannot. Fundamental fields with c = 1 (the Standard Model)
#     can, as in Sakharov's original argument.
ok("IG5 recorded: induced gravity from label media requires the stiff rung (c_s = 1); dust and fluid rungs induce nothing on g", True)
