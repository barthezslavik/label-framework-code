# What the Vlasov-Poisson scaling group is in GA, and how it meets the rotor, SDiff and the bivectors (papers/restart/35-scaling-group.md).
# S1-S7. Phase plane of one degree of freedom (x, v) as Cl(1,1): e^2 = 1, ebar^2 = -1, K = e ebar (K^2 = 1), null n = (e+ebar)/2,
# nbar = (e-ebar)/2 (n.nbar = 1/2), phase point X = x n + v nbar. Scaling: x -> lambda x, v -> mu v, t -> (lambda/mu) t.
# Run: uv run --with sympy python ga_scaling_group_check.py
import sympy as sp
from ga_core import *

C11 = {0: 1, 1: -1}
x, v, x2, v2, th, s, lam, mu, p = sp.symbols('x v x2 v2 theta s lambda mu p', positive=True)
simp = lambda M: {k: w for k, w in ((k, sp.simplify(sp.expand(v_))) for k, v_ in M.items()) if w != 0}
rev = lambda M: {k: (c if len(k) % 4 in (0, 1) else -c) for k, c in M.items()}
e, eb = {(0,): 1}, {(1,): 1}
K = gp(e, eb, C11)
n = scal(sp.Rational(1, 2), add(e, eb)); nb = scal(sp.Rational(1, 2), add(e, scal(-1, eb)))
X = add(scal(x, n), scal(v, nb)); Y = add(scal(x2, n), scal(v2, nb))
coef = lambda V: (sp.simplify(V.get((0,), 0) + V.get((1,), 0)), sp.simplify(V.get((0,), 0) - V.get((1,), 0)))   # (x-part, v-part)

# S1 the squeeze x -> e^th x, v -> e^-th v is the rotor of K: R = exp(K th/2). It keeps X^2 (= x v / 1 ... the hyperbolic norm) and the
#    bivector X ^ Y (the symplectic form x v2 - v x2): it is linear symplectic, i.e. inside phase-space SDiff (Sp(2) = SL(2)).
R = add({(): sp.cosh(th / 2)}, scal(sp.sinh(th / 2), K))
RX = simp(gp(gp(R, X, C11), rev(R), C11)); RY = simp(gp(gp(R, Y, C11), rev(R), C11))
xs, vs = coef(RX)
sq = lambda V: sp.simplify(sp.expand(sp.sympify(gp(V, V, C11).get((), 0)).rewrite(sp.exp)))
w2 = lambda A, B: sp.simplify(wedge(A, B, C11).get((0, 1), 0))
ok(f"S1 K^2 = {gp(K, K, C11)}; R X R~ has x -> {sp.simplify(xs / x)} x, v -> {sp.simplify(vs / v)} v; X^2 = {sq(X)} kept: {sp.simplify(sq(RX) - sq(X)) == 0}; "
   f"X^Y = {w2(X, Y)} kept: {sp.simplify(w2(RX, RY) - w2(X, Y)) == 0}",
   gp(K, K, C11) == {(): 1} and {sp.simplify(xs / x), sp.simplify(vs / v)} == {sp.exp(th), sp.exp(-th)}
   and sp.simplify(sq(RX) - sq(X)) == 0 and sp.simplify(w2(RX, RY) - w2(X, Y)) == 0)

# S2 the whole scaling group is z = s R(th): a split-complex number acting on the phase plane; lambda = s e^th, mu = s e^-th. The scalar part
#    s (Liouville dilation, X -> sX) multiplies the symplectic form by s^2: conformally symplectic, outside SDiff.
zX = simp(scal(s, RX))
ok(f"S2 s R: (lambda, mu) = ({sp.simplify(coef(zX)[0] / x)}, {sp.simplify(coef(zX)[1] / v)}) up to th -> -th; X^Y -> {sp.simplify(w2(scal(s, RX), scal(s, RY)) / w2(X, Y))} X^Y",
   sp.simplify(w2(scal(s, RX), scal(s, RY)) / w2(X, Y) - s**2) == 0)

# S3 the two bivectors of an orbit in a plane B: angular velocity Omega ~ v/r (weight mu/lambda) and specific angular momentum L = x^v ~ r v
#    (weight lambda mu). The subgroup mu = lambda^p keeps I_p = |Omega|^(1+p) |L|^(1-p):
r = sp.symbols('r', positive=True)
Om, L = v / r, r * v
I = lambda pp: Om**(1 + pp) * L**(1 - pp)
named = {1: Om**2, sp.Rational(1, 2): v**2 / r, 0: v**2, sp.Rational(-1, 2): v**2 * r, -1: (r * v)**2}
lab = {1: 'Omega^2 (rigid rotation)', sp.Rational(1, 2): 'a = v^2/r (acceleration, "a0")', 0: 'v^2 (flat)', sp.Rational(-1, 2): 'v^2 r = GM (Kepler)', -1: 'L^2 (symplectic squeeze)'}
gen = sp.simplify(I(p).subs({r: lam * r, v: lam**p * v}) / I(p))
ok(f"S3 I_p = Omega^(1+p) L^(1-p) invariant under (r, v) -> (lambda r, lambda^p v): ratio {gen}; " +
   "; ".join(f"p={k}: {lab[k]}" for k in named),
   gen == 1 and all(sp.simplify(sp.powsimp(I(k) / named[k], force=True)) == 1 for k in named))

# S4 the orbit's rotor exp(-B Omega t/2) is invariant under the whole group: Omega t -> (mu/lambda)(lambda/mu) Omega t.
t, Omega = sp.symbols('t Omega', positive=True)
ok(f"S4 Omega t -> {sp.simplify((mu / lam * Omega) * (lam / mu * t) / (Omega * t))} Omega t: the orbital rotor (phase) is untouched; the group only "
   "changes size and clock", sp.simplify((mu / lam * Omega) * (lam / mu * t) / (Omega * t)) == 1)

# S5 SDiff: phase-space volume dx^3 dv^3 -> (lambda mu)^3: preserved only on mu = 1/lambda (p = -1, the K-rotor); Kelvin circulation
#    oint v.dx and L scale as lambda mu: preserved on the same subgroup. Rigid rotation v = x.Omega is the fixed point of p = 1 (pure dilation).
ok(f"S5 phase volume ~ (lambda mu)^3 = {sp.simplify(((lam * mu)**3).subs(mu, 1 / lam))} on p = -1; circulation ~ lambda mu = "
   f"{sp.simplify((lam * mu).subs(mu, 1 / lam))} on p = -1; Omega ~ mu/lambda = {sp.simplify((mu / lam).subs(mu, lam))} on p = 1",
   sp.simplify(((lam * mu)**3).subs(mu, 1 / lam)) == 1 and sp.simplify((mu / lam).subs(mu, lam)) == 1)

# S6 in log coordinates (ln lambda, ln mu) = (s' + th, s' - th): subgroup p has th/s' = (1-p)/(1+p). Fixed mass (M ~ lambda mu^2 = 1): Newton p = -1/2;
#    deep MOND keeps M with weight-0 potential: p = 0; fixed acceleration: p = 1/2. The swap s' <-> th (multiplication of log z by K) maps p -> -p:
#    Newton <-> the a0-orbit, with deep MOND (th = s') self-dual.
pp = sp.symbols('pp', real=True)
ratio = lambda q: (1 - q) / (1 + q)
pmap = sp.solve(sp.Eq(ratio(pp), 1 / ratio(p)), pp)
ok(f"S6 th/s' = (1-p)/(1+p): Newton p=-1/2 -> {ratio(sp.Rational(-1, 2))}, MOND p=0 -> {ratio(0)}, a0-orbit p=1/2 -> {ratio(sp.Rational(1, 2))}; "
   f"swap s' <-> th sends p -> {pmap}", ratio(0) == 1 and ratio(sp.Rational(-1, 2)) * ratio(sp.Rational(1, 2)) == 1 and pmap == [-p])

# S7 relation to the conformal dilation D = n0 ^ n_inf of 32: D acts on space (x -> e^-l x) with velocities as tangent vectors of the same
#    weight when t is scaled too -- Milgrom's (r, t) -> l (r, t), i.e. p = 0 here. The spatial conformal group therefore contains only the
#    p = 0 line of the scaling group; the K-rotor (squeeze) lives in phase space, not in O(4,1) of space.
# (S7 is a statement: the spatial dilation of O(4,1) with t scaled alike is the p = 0 subgroup; the squeeze K is a phase-space rotor.)
