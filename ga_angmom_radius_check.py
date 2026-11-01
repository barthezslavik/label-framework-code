# 77: does angular-momentum conservation (a grade-2 object, L = r ^ p, conserved while baryons dissipate) put the baryon radius on the
# k = 2 level set and so give M_b ~ v^4? Structural test inside the framework (isothermal dust plateau, 75; shared mixing, 76). AM1-AM3.
# Run: ~/.venvs/jev/bin/python ga_angmom_radius_check.py
import sympy as sp

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

v, G, f, s, a, C, R = sp.symbols('v G f s a C R_i', positive=True)
# AM1 baryons collected from the plateau out to R_i ~ v^a: M_b = f M_halo(<R_i) = f (2 v^2 / G) R_i (lambda = 2 sigma^2/G, v^2 = 2 sigma^2).
#     Slope of M_b ~ v^n: n = 2 + a. a = 1 (k = 3 level set, fixed mean density: R ~ v) -> 3; a = 2 (k = 2 level set, M/R^2 fixed) -> 4.
Ri = C * v**a
Mb = f * 2 * v**2 / G * Ri
n = sp.simplify(sp.diff(sp.log(Mb), v) * v)
ok(f"AM1 M_b ~ v^({n}): k=3 level set (a=1) -> {n.subs(a, 1)}, k=2 level set (a=2) -> {n.subs(a, 2)}", n.subs(a, 1) == 3 and n.subs(a, 2) == 4)

# AM2 dissipation with conserved specific angular momentum: the material from R_i has j = s R_i v (s: spin label, initial data); it
#     settles where it is rotationally supported, R_b v = j, so R_b = s R_i. Settling rescales the radius by the label s and does not
#     touch M_b: the slope is inherited from the collection radius, n = 2 + a unchanged.
Rb = sp.solve(sp.Eq(R * v, s * Ri * v), R)[0]
ok(f"AM2 R_b = {Rb} = s R_i: angular momentum rescales the radius by a label; M_b and its slope ({n}) are unchanged", sp.simplify(Rb - s * Ri) == 0)

# AM3 NEGATIVE: the conserved bivector does not choose the level set. It would only if the spin label depended on v, s ~ v^b, which moves
#     R_b but still not M_b (collected before settling). The k = 2 radius rule has no origin in angular-momentum conservation.
ok("AM3 recorded: angular momentum (grade 2) preserves whatever collection rule the halo had; it does not put the radius on the k = 2 level set", True)
