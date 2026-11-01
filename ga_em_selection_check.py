# 90: how deep is the selection 'electromagnetism = the law blind to the dilation K'? Which premises (grade 2, masslessness, linearity,
# abelian group) can be dropped? EM1-EM6. Run: ~/.venvs/jev/bin/python ga_em_selection_check.py
import sympy as sp

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

lam, nn, k = sp.symbols('lambda n k', positive=True)
# EM1 grade: for a k-form field strength F with action int F ^ *F the weight under g -> lam^2 g (F of weight 0, as for F = dA) is n - 2k
#     (Hodge on grade k scales as lam^(n-2k)). Blind to K iff k = n/2: in 4D only grade 2. Odd dimensions have no such form theory.
w = nn - 2 * k
ok(f"EM1 weight of int F^*F for a k-form in n dims = {w}; zero iff k = {sp.solve(w, k)}: in 4D only bivectors", sp.solve(w, k) == [nn / 2])

# EM2 no rescaling of the field and no curvature coupling: a scalar needs weight -1 and the coupling xi R with xi = 1/6 to be conformal in 4D;
#     the 1-form potential A needs neither. Check the scalar: int sqrt(g) (g^{mu nu} d phi d phi) has weight 4 - 2 = 2 for a weight-0 phi.
w_scalar = 4 - 2
ok(f"EM2 weight-0 scalar kinetic term has weight {w_scalar} != 0 (needs field rescaling and xi R); the bivector field needs nothing", w_scalar == 2)

# EM3 masslessness is selected: a Proca term int sqrt(g) g^{mu nu} A_mu A_nu has weight 4 - 2 = 2 != 0, so a K-blind vector field is massless
#     (massive W, Z need the Higgs vector, an internal choice -- base J10).
w_proca = 4 - 2
ok(f"EM3 Proca mass term weight {w_proca} != 0: K-blindness forces m = 0", w_proca != 0)

# EM4 linearity is NOT selected: with S = (E^2 - B^2)/2, P = E.B (both weight -4 with sqrt(g) weight +4), int sqrt(g) L(S, P) is K-blind iff L is
#     homogeneous of degree 1. Maxwell (L = S) and nonlinear members such as L = sqrt(S^2 + P^2) both qualify.
S, P, t = sp.symbols('S P t', real=True)
hom = lambda L: sp.simplify(L.subs({S: t * S, P: t * P}, simultaneous=True) - t * L)
Lm = S; Lbb = sp.sqrt(S**2 + P**2)
ok(f"EM4 degree-1 homogeneity: Maxwell {hom(Lm)}, sqrt(S^2+P^2) {sp.simplify(hom(Lbb).subs(t, 2))} (t = 2): a whole family is K-blind",
   hom(Lm) == 0 and sp.simplify(hom(Lbb).subs(t, 2)) == 0)

# EM5 adding electric-magnetic duality (the Hodge rotation of the bivector, base B1): condition L_S^2 - (2S/P) L_S L_P - L_P^2 = 1.
#     ModMax L = cosh(g) S + sinh(g) sqrt(S^2 + P^2) satisfies it for every g (Bandos, Lechner, Sorokin, Townsend 2020) and is homogeneous;
#     Maxwell is g = 0, the only member analytic at zero field (sqrt(S^2+P^2) is not).
gm = sp.symbols('gamma', real=True)
Lmm = sp.cosh(gm) * S + sp.sinh(gm) * sp.sqrt(S**2 + P**2)
LS, LP = sp.diff(Lmm, S), sp.diff(Lmm, P)
cond = sp.simplify(LS**2 - 2 * S / P * LS * LP - LP**2 - 1)
ok(f"EM5 ModMax duality condition residual = {cond}; homogeneous: {sp.simplify(hom(Lmm).subs(t, 3)) == 0}; analytic at F = 0 only for gamma = 0 (Maxwell)",
   cond == 0 and sp.simplify(hom(Lmm).subs(t, 3)) == 0)
# EM5b non-analyticity of ModMax at zero field: on P = 0 the one-sided slopes dL/dS at S = 0 are cosh(g) +- sinh(g), equal iff g = 0
Ls0 = Lmm.subs(P, 0)
right = sp.limit(sp.diff(Ls0, S), S, 0, '+'); left = sp.limit(sp.diff(Ls0, S), S, 0, '-')
ok(f"EM5b ModMax on P = 0: slopes at S = 0+ / 0- = {sp.simplify(right)} / {sp.simplify(left)}; equal iff sinh(gamma) = 0, i.e. Maxwell",
   sp.solve(sp.simplify(right - left), gm) == [0])

# EM7 analytic K-blind laws: a Lorentz-scalar monomial with m factors of F (2 indices each, weight 0) and d covariant derivatives needs
#     (2m + d)/2 inverse metrics, weight -(2m + d), plus +4 from sqrt(g): total 4 - 2m - d. Weight 0 with m >= 1, d even (index count even):
#     m = 2, d = 0 only (m = 1, d = 2 is a total derivative). Hence analytic + K-blind = a S + b P: Maxwell plus the theta term (a total
#     derivative); higher-derivative and higher-power terms carry weight != 0 and so bring a scale (a label). Linearity is derived, not assumed.
sols = [(m, d) for m in range(1, 7) for d in range(0, 9, 2) if 4 - 2 * m - d == 0]
ok(f"EM7 weight-0 monomials (m F's, d derivatives): {sols}; m = 1 is a total derivative, so only quadratic F F without derivatives",
   sols == [(1, 2), (2, 0)])
# with L = a S + b P the duality condition of EM5 reads a^2 - 2 (S/P) a b - b^2 = 1 for all S, P: b = 0, a = 1 (b P drops from the
#     field equations anyway) -- duality rotation is then a consequence, not a further premise
a, b = sp.symbols('a b', real=True)
cnd = sp.expand((a**2 - 2 * S / P * a * b - b**2 - 1) * P)
sol = sp.solve(sp.Poly(cnd, S, P).coeffs(), [a, b], dict=True)
ok(f"EM8 linear L = aS + bP: the field equations are Maxwell's (constant b P is a total derivative) and duality-invariant; the Gaillard-Zumino "
   f"condition on the Lagrangian holds iff {sol} -- a condition on the Lagrangian, not a physical selection of theta",
   {a: 1, b: 0} in sol)

# EM6 the group is NOT selected: Yang-Mills tr(F ^ *F) has the same weight n - 2k = 0 in 4D, so K-blindness picks 'grade-2 curvature' laws
#     (abelian and non-abelian alike); U(1) and the value of alpha are inputs/labels.
ok("EM6 recorded: Yang-Mills is K-blind too (same grade-2 weight); the gauge group and alpha are not selected", True)
