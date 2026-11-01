# Which conditions built from Pi(w) and the grade-2 I-duality single out Euler and Maxwell (papers/restart/22-selection.md), E1-E6.
# Run: uv run --with sympy python ga_selection_check.py
import sympy as sp
from ga_core import *

LOR = {0: 1, 1: -1, 2: -1, 3: -1}
I4 = {(0, 1, 2, 3): 1}
pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
c = sp.symbols('c0:6')
Om = clean({p: ci for p, ci in zip(pairs, c)})
th = sp.symbols('theta', real=True)
rot = lambda B: add(scal(sp.cos(th), B), scal(sp.sin(th), gp(I4, B, LOR)))      # e^{I theta} B (duality rotation, A9)
w = {(0,): 1}                                                                    # rest frame of w (statement is Lorentz covariant)

# E1 first-order, quadratic in w, conformally invariant, from w, Omega and I on grade 2: w ^ e^{I theta} Omega = 0 (one-parameter family)
def solve_at(t):
    cond = wedge(w, {k: v.subs(th, t) for k, v in rot(Om).items()}, LOR)
    sol = sp.solve(list(cond.values()), list(c), dict=True)[0]
    Os = {k: sp.simplify(v.subs(sol)) for k, v in Om.items()}
    return sol, Os, sp.simplify(wedge(Os, Os, LOR).get((0, 1, 2, 3), 0)), wedge(w, Os, LOR)
gen = solve_at(th)
ok(f"E1 w ^ e^(I th) Omega = 0 (generic th): {gen[0]}; Omega^Omega = {sp.factor(gen[2])} (nonzero for generic th)", sp.simplify(gen[2]) != 0)
res = {t: solve_at(t) for t in (0, sp.pi / 2, sp.pi / 4, sp.pi / 6)}
ok("E2 Omega^Omega = 0 on all solutions only for theta = 0 (Frobenius) and theta = pi/2 (Euler)",
   res[0][2] == 0 and res[sp.pi / 2][2] == 0 and res[sp.pi / 4][2] != 0 and res[sp.pi / 6][2] != 0)
ok(f"E3 helicity w^Omega on solutions: theta = 0 -> {res[0][3]} (zero), theta = pi/2 -> {res[sp.pi / 2][3]} (free): "
   "Euler is the only member with conserved AND nonzero helicity (Pi_4 = 0, Pi_3 != 0)",
   res[0][3] == {} and res[sp.pi / 2][3] != {})
# E4 linear first-order conditions on w: only grad^w = 0 is conformally invariant (I on grade 1 scales by lambda^2, 21 G2) -> trivial
Xc = sp.symbols('t x y z'); Lc = sp.Function('L')(*Xc); wc = [sp.Function(f'u{m}')(*Xc) for m in range(4)]; et = [1, -1, -1, -1]
div = lambda lam: sp.expand(sum(sp.diff(lam**4 * lam**-2 * et[m] * wc[m], Xc[m]) for m in range(4)) / lam**4)   # (1/sqrt g) d(sqrt g g^mn w_n), g = lam^2 eta
ok("E4 linear first order: div_g w for g = L^2 eta is not a multiple of div_eta w (extra dL terms) -> grad.w = 0 needs the scale; "
   "only grad^w = 0 (Omega = 0, trivial) is conformal",
   sp.simplify(div(Lc) - div(1) / Lc**2) != 0)
# E5 linear second order from d and I on grade 2: grad^(e^{I th} grad^A) = sin(th) grad^(I F) since grad^F = 0 -> Maxwell is the only nontrivial one
X = sp.symbols('t x y z')
A = vec([sp.Function(f'A{m}')(*X) for m in range(4)])
F = ext_d(A, X, LOR)
lhs = ext_d(rot(F), X, LOR)
rhs = scal(sp.sin(th), ext_d(gp(I4, F, LOR), X, LOR))
ok("E5 grad^(e^(I th) F) = sin(th) grad^(I F) for F = grad^A: the whole family collapses to Maxwell grad^(IF) = 0 (i.e. grad.F = 0)",
   all(sp.simplify(v) == 0 for v in add(lhs, scal(-1, rhs)).values()))
# E6 no real self-dual bivector: I B = k B with real k forces B = 0 (I^2 = -1 on grade 2)
k = sp.symbols('k', real=True)
eqs = list(add(gp(I4, Om, LOR), scal(-k, Om)).values())
solsd = sp.solve(eqs, list(c), dict=True)
ok(f"E6 I B = k B (k real) has only B = 0: {solsd}", all(all(v == 0 for v in s_.values()) for s_ in solsd))
# E7 completeness in degree: all wedge monomials in w, Omega, I Omega (first order). Anything above grade 4 vanishes, and
#     (I Omega)^(I Omega) = -Omega^Omega, so the list of independent monomials is finite: w, Omega, I Omega, w^Omega, w^I Omega,
#     Omega^Omega, Omega^I Omega (grades 1, 2, 2, 3, 3, 4, 4); higher powers add nothing.
W = vec(list(sp.symbols('v0:4'))); IO = gp(I4, Om, LOR)
mons = {'w^w': wedge(W, W, LOR), 'w^Om^Om': wedge(W, wedge(Om, Om, LOR), LOR), 'w^Om^IOm': wedge(W, wedge(Om, IO, LOR), LOR),
        'Om^Om^Om': wedge(Om, wedge(Om, Om, LOR), LOR)}
ok(f"E7 grade > 4 monomials vanish: {[k for k, v in mons.items() if v == {}]}; (I Om)^(I Om) = -Om^Om: "
   f"{clean(add(wedge(IO, IO, LOR), wedge(Om, Om, LOR))) == {}} -> the first-order monomial list is complete (7 objects)",
   all(v == {} for v in mons.values()) and clean(add(wedge(IO, IO, LOR), wedge(Om, Om, LOR))) == {})
# E8 equation count: in the rest frame of w, w ^ e^{I th} Omega = 0 is a rank-3 linear system on the 6 components of Omega for every th
#     (3 independent equations, as Euler's u.Omega = 0) -> every member of the circle is a 3-equation system for w; none is over/under-determined differently
ranks = {t: sp.Matrix([[sp.diff(e_, ci) for ci in c] for e_ in wedge(w, {k: v.subs(th, t) for k, v in rot(Om).items()}, LOR).values()]).rank()
         for t in (0, sp.pi / 6, sp.pi / 4, sp.pi / 2)}
ok(f"E8 rank of w ^ e^(I th) Omega = 0 in Omega for th = 0, pi/6, pi/4, pi/2: {list(ranks.values())} (all 3)", set(ranks.values()) == {3})
# E9 where Pi_4 = 0 comes from: w pulled back from a 3-dimensional label space, w = alpha_I(phi) grad phi^I (I = 1..3, arbitrary alpha, phi),
#     has Omega^Omega = 0 identically (cf. 11 L3); conversely class <= 3 is locally such a pullback (Darboux, 20 S8). So the selection
#     requirement of E2-E3 is "matter has a 3D label space" (SDiff of labels), not an extra assumption.
Xs = sp.symbols('t x y z'); P = sp.symbols('P1:4')
phis = [sp.Function(f'phi{i}')(*Xs) for i in range(3)]
alph = [sp.Function(f'al{i}')(*P) for i in range(3)]
sub = dict(zip(P, phis))
wl = add(*[scal(alph[i].subs(sub), vec([sp.diff(phis[i], x) for x in Xs])) for i in range(3)])
Ol = ext_d(wl, Xs, LOR)
ok("E9 w = alpha_I(phi) grad phi^I from 3 labels: Omega^Omega = 0 identically -> Pi_4 = 0 is the 3-label (SDiff of labels) condition",
   all(sp.simplify(v) == 0 for v in wedge(Ol, Ol, LOR).values()))
# E10 why three: comoving labels (u.grad phi = 0) have gradients in u-perp, which is 3-dimensional for timelike u in Cl(1,3) (A11).
#     Rest frame u = e0: labels depend on x, y, z only. Any 4 of them have dphi1^...^dphi4 = 0, and any Omega built from them has Omega^Omega = 0.
Xr = sp.symbols('t x y z')
cl = [sp.Function(f'q{i}')(*Xr[1:]) for i in range(4)]
grads = [vec([sp.diff(q, x) for x in Xr]) for q in cl]
w4 = wedge(wedge(grads[0], grads[1], LOR), wedge(grads[2], grads[3], LOR), LOR)
cij = [sp.Function(f'k{i}')(*cl) for i in range(6)]
Oc = add(*[scal(cij[k_], wedge(grads[i], grads[j], LOR)) for k_, (i, j) in enumerate([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])])
ok("E10 four comoving labels: dq1^dq2^dq3^dq4 = 0; any Omega built from comoving labels has Omega^Omega = 0 -> at most 3 comoving labels (dim u-perp = 3, signature 1+3)",
   all(sp.simplify(v) == 0 for v in w4.values()) and all(sp.simplify(v) == 0 for v in wedge(Oc, Oc, LOR).values()))
# E11 two labels give no helicity: w = al_1(phi) dphi1 + al_2(phi) dphi2 -> w ^ Omega = 0; so nonzero helicity needs 3, Pi_4 = 0 allows at most 3
p2 = [sp.Function(f'r{i}')(*Xr) for i in range(2)]
a2 = [sp.Function(f'be{i}')(*p2) for i in range(2)]
w2 = add(*[scal(a2[i], vec([sp.diff(p2[i], x) for x in Xr])) for i in range(2)])
ok("E11 w from 2 labels: w ^ grad^w = 0 (no helicity): Pi_3 != 0 needs >= 3 labels, Pi_4 = 0 (E9, E10) allows <= 3 -> exactly 3",
   all(sp.simplify(v) == 0 for v in wedge(w2, ext_d(w2, Xr, LOR), LOR).values()))
# E12 continuity from labels: J = f(phi) (dphi1^dphi2^dphi3) I^-1 for 3 arbitrary labels. Then grad.J = 0 identically (cf. 11 L4)
#     and J.grad phi^i = 0 (J is tangent to the label level sets, i.e. along the flow). Continuity is not an input; only f (data on label space) is.
ph3 = [sp.Function(f'm{i}')(*Xr) for i in range(3)]
g3 = [vec([sp.diff(q, x) for x in Xr]) for q in ph3]
fN = sp.Function('fN')(*ph3)
N3 = scal(fN, wedge(wedge(g3[0], g3[1], LOR), g3[2], LOR))
J = grade(gp(N3, I4, LOR), 1)
divJ = sp.simplify(sum(LOR[m] * sp.diff(J.get((m,), 0), Xr[m]) for m in range(4)))
ok(f"E12 J = f N3 I: div J = {divJ}, J.grad phi^i = {[sp.simplify(gp(J, g_, LOR).get((), 0)) for g_ in g3]} -> continuity holds identically",
   divJ == 0 and all(sp.simplify(gp(J, g_, LOR).get((), 0)) == 0 for g_ in g3))
# E13 no third order: with only grad^ and I on grade 2 the chain A -> grad^A -> I grad^A -> grad^(I grad^A) (grade 3) ends, because
#     I on grade 3 is not conformal (21 G2) and grad^grad^ = 0. Check grad^(grad^(I grad^A)) = 0 identically.
F2 = ext_d(A, X, LOR)
ok("E13 grad^(grad^(I F)) = 0: the conformal chain built from grad^ and grade-2 I stops at second order (Maxwell); no third-order condition",
   all(sp.simplify(v) == 0 for v in ext_d(ext_d(gp(I4, F2, LOR), X, LOR), X, LOR).values()))
# E14 intermediate theta: w ^ e^{I th} Omega = 0 <=> e^{I th} Omega = w ^ b <=> Omega = e^{-I th}(w ^ b) (a pure-"electric" field w.r.t. w,
#     duality-rotated by th). Then Omega^Omega = sin(2 th) * (2 (w^b)^2 I-part) ... computed below: nonzero for th != 0, pi/2, so the helicity
#     current is not conserved: grad^(w^Omega) = Omega^Omega (20 S5) is a source with a fixed duality angle. The algebra leaves th free.
bv = vec([0] + list(sp.symbols('b1:4')))
wb = wedge(w, bv, LOR)
Omt = add(scal(sp.cos(th), wb), scal(-sp.sin(th), gp(I4, wb, LOR)))        # e^{-I th}(w^b)
cond_t = wedge(w, rot(Omt), LOR)
OOt = sp.simplify(wedge(Omt, Omt, LOR).get((0, 1, 2, 3), 0))
ok(f"E14 Omega = e^(-I th)(w^b) solves w ^ e^(I th) Omega = 0: {all(sp.simplify(v) == 0 for v in cond_t.values())}; Omega^Omega = {sp.factor(OOt)} "
   "(vanishes only at th = 0, pi/2): intermediate th = helicity source with fixed duality angle, th itself not fixed by the algebra",
   all(sp.simplify(v) == 0 for v in cond_t.values()) and sp.simplify(OOt.subs(th, sp.pi / 4)) != 0 and OOt.subs(th, 0) == 0)
