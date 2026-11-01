# The algebra behind the symmetry list (papers/restart/36-deep-ga.md). U1-U5.
# Claim: every item of the list is built from units of square -1 (rotations: phases, conserved charges) or square +1 (boosts, dilations,
# chiral pseudoscalars: idempotent splits into two null halves). The "middles" (45 deg, mu = 1/2, p = 0) are the null directions of the
# indefinite form that a square-+1 unit defines.
# Run: uv run --with sympy python ga_deep_units_check.py
import itertools
import sympy as sp
from ga_core import *

def I_of(sig):
    return {tuple(range(len(sig))): 1}
def central(I, sig):
    return all(gp(I, {(k,): 1}, sig) == gp({(k,): 1}, I, sig) for k in range(len(sig)))

# U1 the 5D pseudoscalar is central; its square decides its nature: Cl(4,1) (conformal space): I5^2 = -1 (a phase, Cl(4,1) = C (x) Cl(1,3));
#    Cl(1,4) (de Sitter embedding) and Cl(5,0) (Euclidean S^4): I5^2 = +1 (a chiral split, P+- = (1 +- I5)/2).
sigs = {'Cl(4,1)': {0: 1, 1: 1, 2: 1, 3: 1, 4: -1}, 'Cl(1,4)': {0: 1, 1: -1, 2: -1, 3: -1, 4: -1}, 'Cl(5,0)': {k: 1 for k in range(5)}}
sq = {k: gp(I_of(s), I_of(s), s) for k, s in sigs.items()}
ok(f"U1 I5^2: " + ", ".join(f"{k}: {v[()]}" for k, v in sq.items()) + "; central in all three",
   sq['Cl(4,1)'] == {(): -1} and sq['Cl(1,4)'] == {(): 1} and sq['Cl(5,0)'] == {(): 1} and all(central(I_of(s), s) for s in sigs.values()))

# U2 the spacetime algebra sits inside Cl(4,1): gamma0 = e_+ (index 3, square +1), gamma_i = e_i I5 (i = 0,1,2: (e_i I5)^2 = -e_i^2 = -1),
#    pairwise anticommuting. With I5 central and I5^2 = -1,
#    Cl(4,1) = C (x) Cl(1,3) (the Dirac algebra).
s41 = sigs['Cl(4,1)']; I41 = I_of(s41)
g = [{(3,): 1}] + [gp({(i,): 1}, I41, s41) for i in (0, 1, 2)]
squares = [gp(a, a, s41).get((), 0) for a in g]
anti = all(add(gp(g[a], g[b], s41), gp(g[b], g[a], s41)) == {} for a, b in itertools.combinations(range(4), 2))
ok(f"U2 gamma squares {squares} (signature 1,3), pairwise anticommuting: {anti}; I5 central with I5^2 = -1 -> Cl(4,1) = C (x) Cl(1,3)",
   squares == [1, -1, -1, -1] and anti)

# U3 every square-+1 unit U (D = n0^n_inf, the squeeze K, the de Sitter boost B, the chiral I5 of Cl(5,0)) gives idempotents P+- = (1+-U)/2
#    and the half trace <P+- X>_0 = <X>_0/2 for X without a U-component in the scalar part (here: <U>_0 = 0).
units = {'D in Cl(4,1)': (wedge({(4,): sp.Rational(1, 2), (3,): -sp.Rational(1, 2)}, {(3,): 1, (4,): 1}, s41), s41),
         'K in Cl(1,1)': ({(0, 1): 1}, {0: 1, 1: -1}),
         'B in Cl(1,4)': ({(0, 4): 1}, sigs['Cl(1,4)']),
         'I5 in Cl(5,0)': (I_of(sigs['Cl(5,0)']), sigs['Cl(5,0)'])}
res = []
for name, (U, sg) in units.items():
    Pp = scal(sp.Rational(1, 2), add({(): 1}, U)); Pm = scal(sp.Rational(1, 2), add({(): 1}, scal(-1, U)))
    res.append(gp(U, U, sg) == {(): 1} and gp(Pp, Pp, sg) == Pp and gp(Pp, Pm, sg) == {} and Pp.get((), 0) == sp.Rational(1, 2))
ok(f"U3 square +1 and idempotent split for {list(units)}: {res}", all(res))

# U4 the middle as a null direction: for the plane B (Euclidean S^4, B = e0 e4) the reflection sigma(X) = B X B^-1 negates the in-plane part,
#    so X . sigma(X) = d^2 - rho^2 = -l^2 cos 2chi: 45 deg is a null direction of this indefinite form (same statement as 31/H4).
l, chi = sp.symbols('l chi', positive=True)
s50 = sigs['Cl(5,0)']
X = vec([l * sp.cos(chi), l * sp.sin(chi), 0, 0, 0])
B = {(0, 4): 1}; Binv = {(0, 4): -1}
sX = gp(gp(B, X, s50), Binv, s50)
form = sp.simplify(gp(X, sX, s50).get((), 0))
ok(f"U4 X . (B X B^-1) = {form} = -l^2 cos(2 chi): zero at chi = pi/4 (null for the split form)",
   sp.simplify(form + l**2 * sp.cos(2 * chi)) == 0)

# U5 the same for the other middles: rotor weights (alpha, beta) = (e^-lam/2, e^lam/2) of the dilation rotor -> alpha^2 - beta^2 = 0 at lam = 0
#    (mu = 1/2); the scaling-group log coordinates (s', th) -> s'^2 - th^2 = 0 on th = s' (p = 0, deep MOND).
lam, sp_, th = sp.symbols('lambda s th', real=True)
a_, b_ = sp.exp(-lam / 2), sp.exp(lam / 2)
ok(f"U5 alpha^2 - beta^2 = {sp.simplify(a_**2 - b_**2)} vanishes at lam = {sp.solve(sp.Eq(a_**2 - b_**2, 0), lam)} (mu = 1/2); "
   f"s'^2 - th^2 vanishes on th = s' (p = 0): {sp.simplify((sp_**2 - th**2).subs(th, sp_)) == 0}",
   sp.solve(sp.Eq(a_**2 - b_**2, 0), lam) == [0] and sp.simplify((sp_**2 - th**2).subs(th, sp_)) == 0)
