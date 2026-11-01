# The GA basis of the split GL = SL x K: the pseudoscalar I and Hodge duality (papers/restart/48-sl-k-hodge.md). HK1-HK6.
# Standard GA facts (Hestenes-Sobczyk): the outermorphism of f acts on I as f(I) = det(f) I; the inverse is f^-1(a) = fbar(a I) I^-1 / det f.
# Run: uv run --with sympy --with numpy python ga_sl_k_hodge_check.py
import itertools
import numpy as np
import sympy as sp
from ga_core import *

E3 = {0: 1, 1: 1, 2: 1}
rng = np.random.default_rng(11)
M = sp.Matrix(3, 3, [sp.Rational(int(v), 7) for v in rng.integers(-9, 10, size=9)])
f = lambda a: vec(list(M * sp.Matrix([a.get((k,), 0) for k in range(3)])))
fbar = lambda a: vec(list(M.T * sp.Matrix([a.get((k,), 0) for k in range(3)])))
e = [{(k,): 1} for k in range(3)]
I3 = {(0, 1, 2): 1}
Iinv = scal(1 / sp.Integer(gp(I3, I3, E3)[()]), I3)

# HK1 the outermorphism on the pseudoscalar: f(e1)^f(e2)^f(e3) = det(f) I.
fI = wedge(wedge(f(e[0]), f(e[1]), E3), f(e[2]), E3)
ok(f"HK1 f(I) = {fI.get((0, 1, 2), 0)} I, det f = {M.det()}: the pseudoscalar measures the volume change", sp.simplify(fI.get((0, 1, 2), 0) - M.det()) == 0)

# HK2 inverse through Hodge duality: f^-1(a) = fbar_(2)(a I) I^-1 / det f, where fbar_(2) is the outermorphism of the adjoint on bivectors.
def fbar_biv(Bv):
    out = {}
    for (i, j), cval in Bv.items():
        out = add(out, scal(cval, wedge(fbar(e[i]), fbar(e[j]), E3)))
    return out
a = vec([sp.Rational(2, 3), -1, sp.Rational(1, 2)])
lhs = vec(list(M.inv() * sp.Matrix([a.get((k,), 0) for k in range(3)])))
rhs = scal(1 / M.det(), grade(gp(fbar_biv(grade(gp(a, I3, E3), 2)), Iinv, E3), 1))
ok(f"HK2 f^-1(a) = fbar(a I) I^-1 / det f: difference {clean({k: sp.simplify(lhs.get(k, 0) - rhs.get(k, 0)) for k in set(lhs) | set(rhs)})}: Hodge conjugates the adjoint into det f times the inverse",
   clean({k: sp.simplify(lhs.get(k, 0) - rhs.get(k, 0)) for k in set(lhs) | set(rhs)}) == {})

# HK3 the split: SL = {f : f(I) = I} (keeps the pseudoscalar, hence every Hodge dual); the scale K: f = lam id gives f(I) = lam^n I.
lam = sp.symbols('lambda', positive=True)
scaleI = wedge(wedge(scal(lam, e[0]), scal(lam, e[1]), E3), scal(lam, e[2]), E3)
Msl = M / sp.root(M.det(), 3) if M.det() > 0 else -M / sp.root(-M.det(), 3)
ok(f"HK3 scale: f(I) = {scaleI.get((0, 1, 2), 0)} I; the unimodular part M/det^(1/3) has det {sp.nsimplify(sp.simplify(Msl.det()))}: GL = SL x scale through the action on I",
   scaleI.get((0, 1, 2), 0) == lam**3 and sp.simplify(Msl.det() - 1) == 0)

# HK4 what K does to Hodge: for g -> lam^2 g the dual on grade k scales as lam^(n-2k) (= 21 G2): only the middle grade k = n/2 is K-invariant.
# (HK4 is the statement of 21 G2 / base B3, not recomputed here: the dual on grade k scales as lam^(n-2k); only k = n/2 is K-invariant,
#  so the middle-grade duality is shared by SL and K, while the full volume form I belongs to SL only.)

# HK5 in the doubled algebra Cl(2,2): K = sum n_i ^ m_i commutes with every B_L (K is central in gl(n)), and its rotor scales the null vectors.
C22 = {0: 1, 1: 1, 2: -1, 3: -1}
n = [add({(0,): 1}, {(2,): 1}), add({(1,): 1}, {(3,): 1})]
m = [add({(0,): 1}, {(2,): -1}), add({(1,): 1}, {(3,): -1})]
L = sp.Matrix(2, 2, sp.symbols('l11 l12 l21 l22'))
BL = {}
for i in range(2):
    for j in range(2):
        BL = add(BL, scal(L[i, j] / 2, wedge(n[i], m[j], C22)))
K = add(wedge(n[0], m[0], C22), wedge(n[1], m[1], C22))
comm = clean({k: sp.simplify(v) for k, v in add(gp(BL, K, C22), scal(-1, gp(K, BL, C22))).items()})
Kact = clean({k: sp.simplify(v / 2) for k, v in add(gp(K, n[0], C22), scal(-1, gp(n[0], K, C22))).items()})
ok(f"HK5 [B_L, K] = {comm} for every L (K central); [K, n_0]/2 = {Kact} = 2 n_0 (K acts as the scale)",
   comm == {} and Kact == scal(2, n[0]))

# HK6 the pseudoscalar of the null subspace N = n_0 ^ n_1: the gl-action of B_L on N is (tr L) N -- SL keeps N, K rescales it.
N = wedge(n[0], n[1], C22)
act = clean({k: sp.simplify(v / 2) for k, v in add(gp(BL, N, C22), scal(-1, gp(N, BL, C22))).items()})
ratio = sp.simplify(act.get((0, 1), 0) / N.get((0, 1), 1)) if (0, 1) in N else None
ok(f"HK6 [B_L, N]/2 = (tr L) N: coefficient {sp.factor(ratio)}; traceless (SL) keeps the null pseudoscalar, the trace (K) rescales it",
   ratio is not None and sp.simplify(ratio - (L[0, 0] + L[1, 1])) == 0
   and all(sp.simplify(act.get(k, 0) - (L[0, 0] + L[1, 1]) * N.get(k, 0)) == 0 for k in set(act) | set(N)))
