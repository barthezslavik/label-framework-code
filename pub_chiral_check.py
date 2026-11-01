# What survives of paper2: the chiral-reading theorem in Cl(1,4) (withdrawal note, "What survives"). C1-C3.
# For generic N in grade 3 and F in grade 2: NF has odd grades only; <N F I5>_0 = <NF P+>_0 - <NF P->_0 = 2<P+ NF>_0 = -2<P- NF>_0.
# Run: uv run --with sympy python pub_chiral_check.py
import itertools
import random
import sympy as sp
from ga_core import *

S5 = {0: 1, 1: -1, 2: -1, 3: -1, 4: -1}
I5 = {(0, 1, 2, 3, 4): 1}
random.seed(7)
tri = list(itertools.combinations(range(5), 3)); biv = list(itertools.combinations(range(5), 2))
N = clean({b: sp.Rational(random.randint(-9, 9), random.randint(1, 5)) for b in tri})
F = clean({b: sp.Rational(random.randint(-9, 9), random.randint(1, 5)) for b in biv})
Pp = scal(sp.Rational(1, 2), add({(): 1}, I5)); Pm = scal(sp.Rational(1, 2), add({(): 1}, scal(-1, I5)))
NF = gp(N, F, S5)
s0 = lambda X: X.get((), 0)
ok(f"C1 I5^2 = {gp(I5, I5, S5)}, I5 central (commutes with generators); NF has grades {grades(NF)} (odd only), <NF>_0 = {s0(NF)}",
   gp(I5, I5, S5) == {(): 1} and all(gp(I5, {(k,): 1}, S5) == gp({(k,): 1}, I5, S5) for k in range(5))
   and set(grades(NF)) <= {1, 3, 5} and s0(NF) == 0)
a = s0(gp(NF, I5, S5)); b = s0(gp(NF, Pp, S5)) - s0(gp(NF, Pm, S5)); c = 2 * s0(gp(Pp, NF, S5)); d = -2 * s0(gp(Pm, NF, S5))
ok(f"C2 <NF I5>_0 = {a} = <NF P+> - <NF P-> = {b} = 2<P+ NF> = {c} = -2<P- NF> = {d} (random rational N, F)",
   a == b == c == d and a != 0)
# C3 the half-trace identity used for the 1/sqrt2 discussion: <P+- X>_0 = 1/2 <X>_0 for even X
Xe = add(gp(N, gp(N, F, S5), S5), F, {(): 3})
Xe = clean({k: v for k, v in Xe.items() if len(k) % 2 == 0})
ok(f"C3 half trace for even X: <P+ X>_0 = {s0(gp(Pp, Xe, S5))} = <X>_0/2 = {s0(Xe) / 2}", s0(gp(Pp, Xe, S5)) == s0(Xe) / 2 and s0(gp(Pm, Xe, S5)) == s0(Xe) / 2)
