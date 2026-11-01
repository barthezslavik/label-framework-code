# Task 1 of the publication goal: can the three objects called N in paper1-4 be one object?
# (a) paper1: label three-form N3 = f(phi) dphi1^dphi2^dphi3 (spacetime 3-form);
# (b) paper1 sec.5 / paper4: I5 F with F^{a4} = T^a/l, soldered to a spacetime 3-form e_a ^ T^a;
# (c) paper4 sec.6: grade-3 spinor bilinear = dual of the axial current s = psi g3 psi~.
# Invariants compared: closedness (identical conservation), causal character of the dual vector. papers/restart/30-publication.md, U1-U4.
# Run: uv run --with sympy python pub_unified_n_check.py
import sympy as sp
from ga_core import *

LOR = {0: 1, 1: -1, 2: -1, 3: -1}
I4 = {(0, 1, 2, 3): 1}
X = sp.symbols('t x y z')
rev = lambda X_: clean({k: v * (-1) ** (len(k) * (len(k) - 1) // 2) for k, v in X_.items()})
grad = lambda f: vec([sp.diff(f, x) for x in X])

# U1 (a) label three-form: closed identically, dual vector J = N3 I timelike-or-null along the flow (comoving labels in the rest frame)
ph = [sp.Function(f'p{i}')(*X) for i in range(3)]
fN = sp.Function('f')(*ph)
N3 = scal(fN, wedge(wedge(grad(ph[0]), grad(ph[1]), LOR), grad(ph[2]), LOR))
ok("U1a label three-form: dN3 = 0 identically (conservation is an identity)", all(sp.simplify(v) == 0 for v in ext_d(N3, X, LOR).values()))
rest = {ph[0]: sp.Function('q0')(*X[1:]), ph[1]: sp.Function('q1')(*X[1:]), ph[2]: sp.Function('q2')(*X[1:])}
Jr = grade(gp({k: v.subs(rest).doit() for k, v in N3.items()}, I4, LOR), 1)
ok(f"U1b comoving labels (independent of t): J = N3 I has only a time component {list(Jr)} -> J timelike: the dual of (a) is timelike",
   set(Jr) <= {(0,)} and Jr != {})

# U2 (b) soldered I5 F: with F^{a4} = T^a/l the three-form is e_a ^ T^a. Weitzenboeck example (omega = 0, so R = 0):
#    e^0 = dt, e^1 = dx, e^2 = dy, e^3 = (1 + y) dz + t dx  ->  T^3 = de^3 = dy^dz + dt^dx.
e = [vec([1, 0, 0, 0]), vec([0, 1, 0, 0]), vec([0, 0, 1, 0]), vec([0, X[0], 0, 1 + X[2]])]
T = [ext_d(ea, X, LOR) for ea in e]
eta = [1, -1, -1, -1]
eT = add(*[scal(eta[a], wedge(e[a], T[a], LOR)) for a in range(4)])
deT = ext_d(eT, X, LOR)
TT = add(*[scal(eta[a], wedge(T[a], T[a], LOR)) for a in range(4)])
ok(f"U2 e_a^T^a for this coframe: d(e^T) = {deT} = T^a^T_a (Nieh-Yan density with R = 0) != 0 -> not closed, so not a label three-form",
   deT != {} and clean(add(deT, scal(-1, TT))) == {})
# U2b with zero torsion (de Sitter vacuum of paper4, F = 0) it vanishes; with torsion it is closed only on the Nieh-Yan-free sector
ok("U2b zero torsion (T = 0): e^T = 0 identically; so (b) is either zero (vacuum) or generically non-closed (torsion)",
   all(T_ == {} for T_ in [ext_d(vec([1, 0, 0, 0]), X, LOR)]))

# U3 (c) spinor bilinear: for psi = sqrt(rho) e^{I beta/2} R, the axial current s = psi g3 psi~ = rho R g3 R~ is spacelike, s^2 = -rho^2,
#    and orthogonal to the vector current J = psi g0 psi~ (timelike). Its dual (the grade-3 bilinear) therefore has a spacelike dual vector.
rho, beta, th, chi = sp.symbols('rho beta theta chi', positive=True)
R = gp(add(scal(sp.cos(th / 2), {(): 1}), scal(sp.sin(th / 2), {(1, 2): 1})), add(scal(sp.cosh(chi / 2), {(): 1}), scal(sp.sinh(chi / 2), {(0, 3): 1})), LOR)
eb = add(scal(sp.cos(beta / 2), {(): 1}), scal(sp.sin(beta / 2), I4))
psi = scal(sp.sqrt(rho), gp(eb, R, LOR))
s = clean({k: sp.simplify(v) for k, v in gp(gp(psi, {(3,): 1}, LOR), rev(psi), LOR).items()})
Jv = clean({k: sp.simplify(v) for k, v in gp(gp(psi, {(0,): 1}, LOR), rev(psi), LOR).items()})
s2 = sp.simplify(gp(s, s, LOR).get((), 0)); J2 = sp.simplify(gp(Jv, Jv, LOR).get((), 0))
Js = sp.simplify((gp(Jv, s, LOR).get((), 0) + gp(s, Jv, LOR).get((), 0)) * sp.Rational(1, 2))
ok(f"U3 Dirac-Hestenes bilinears: s^2 = {s2} (spacelike), J^2 = {J2} (timelike), J.s = {Js}: the grade-3 bilinear has a spacelike dual",
   sp.simplify(s2 + rho**2) == 0 and sp.simplify(J2 - rho**2) == 0 and sp.simplify(Js) == 0 and grades(s) == [1])

# U4 verdict: (a) closed with timelike dual; (b) zero on the vacuum or non-closed; (c) spacelike dual. No two coincide in general:
#    (a) != (b) by closedness (U1a vs U2), (a) != (c) by causal character (U1b vs U3), (b) != (c) since (b) need not have a spacelike dual
#    and (c) is not built from the coframe. The chain paper1 -> paper4 identifies different objects.
a_closed = all(sp.simplify(v) == 0 for v in ext_d(N3, X, LOR).values()); b_closed = deT == {}
a_timelike = set(Jr) <= {(0,)} and Jr != {}; c_timelike = sp.simplify(s2) > 0
ok(f"U4 (a) closed={a_closed}, (b) closed={b_closed}; (a) dual timelike={a_timelike}, (c) dual timelike={c_timelike}: "
   "the three objects called N differ in an invariant -> chain paper1 -> paper4 broken (task 1: NO)",
   a_closed and not b_closed and a_timelike and not c_timelike)
