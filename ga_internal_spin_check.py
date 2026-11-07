# 147: step 5 of paper13. Spin: external rotation, internal sign.
# Premises: the lock J = I of spatial and label rotations for a hedgehog (paper8, computed there); labels valued in SU(3) with the
# Wess-Zumino term of level N_c ('t Hooft matching with N_c colours); the collective-coordinate constraint of that term, right
# hypercharge Y_R = N_c B / 3 (Witten 1983; Guadagnini 1984; known, not recomputed); representation I2 of ga_internal_check.py.
# SP1 for the B = 1 soliton the allowed states are those of SU(3) irreps (p, q) with a weight of hypercharge Y = N_c/3; their spin is the
#     isospin I of that weight (lock J = I). Enumerating Gelfand-Tsetlin patterns for p + q <= 12: for N_c odd every allowed spin is
#     half-integer, for N_c even every allowed spin is integer; for N_c = 3 the lowest states are the octet (J = 1/2) and the decuplet
#     (J = 3/2);
# SP2 chain: the global SU(2) anomaly of the internal representation (1 + N_c Weyl doublets per generation, IN5) is absent iff N_c is
#     odd iff every soliton spin of SP1 is half-integer, for N_c = 1..7: the Finkelstein-Rubinstein sign of R8 equals (-1)^{N_c} and is
#     fixed by the consistency of the internal sector, given the Wess-Zumino level N_c;
# SP3 the same group pi_4(SU(2)) = Z_2 carries both the global anomaly and the Finkelstein-Rubinstein sign; for labels in S^3 = SU(2)
#     alone the Wess-Zumino 5-form vanishes (dim SU(2) = 3 < 5), so the sign is a free Z_2 angle: checked by the vanishing of
#     Tr(U^-1 dU)^5 on random SU(2) frames and its non-vanishing on SU(3).
# Run: ~/.venvs/jev/bin/python ga_internal_spin_check.py
import numpy as np
from fractions import Fraction
from itertools import permutations
from scipy.linalg import expm

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

def weights(p, q):
    """Gelfand-Tsetlin patterns of the SU(3) irrep (p, q): yield (Y, I) with Y the hypercharge, I the isospin of the multiplet."""
    m13, m23, m33 = p + q, q, 0
    for m12 in range(m23, m13 + 1):
        for m22 in range(m33, m23 + 1):
            for m11 in range(m22, m12 + 1):
                Y = Fraction(m12 + m22) - Fraction(2, 3) * (m13 + m23 + m33)
                I = Fraction(m12 - m22, 2)
                yield Y, I

def allowed_spins(Nc, pmax=12):
    out = {}
    for p in range(pmax + 1):
        for q in range(pmax + 1 - p):
            Js = {I for Y, I in weights(p, q) if Y == Fraction(Nc, 3)}
            if Js: out[(p, q)] = Js
    return out

# sanity: dimension of (p, q) from patterns
for p, q in [(1, 0), (1, 1), (3, 0), (2, 2)]:
    assert sum(1 for _ in weights(p, q)) == (p + 1) * (q + 1) * (p + q + 2) // 2

# ---------- SP1 ----------
res = {}
for Nc in range(1, 8):
    sp = allowed_spins(Nc)
    allJ = set().union(*sp.values())
    half = all(J.denominator == 2 for J in allJ); integ = all(J.denominator == 1 for J in allJ)
    res[Nc] = (half, integ, sp)
sp3 = res[3][2]
low = {k: sorted(v) for k, v in sp3.items() if sum(k) <= 3}
ok(f"SP1 N_c odd -> all soliton spins half-integer, N_c even -> all integer (N_c = 1..7, p + q <= 12); N_c = 3 lowest: "
   + ", ".join(f"({p},{q}) J={[str(j) for j in v]}" for (p, q), v in sorted(low.items())),
   all(res[N][0] for N in (1, 3, 5, 7)) and all(res[N][1] for N in (2, 4, 6))
   and low.get((1, 1)) == [Fraction(1, 2)] and low.get((3, 0)) == [Fraction(3, 2)])

# ---------- SP2 chain with the global SU(2) anomaly ----------
chain = all(((1 + N) % 2 == 0) == res[N][0] for N in range(1, 8))
ok("SP2 for N_c = 1..7: no global SU(2) anomaly (1 + N_c doublets even) <=> N_c odd <=> every soliton spin half-integer; "
   "the FR sign of R8 is (-1)^{N_c}, fixed by the internal sector given the WZ level N_c", chain)

# ---------- SP3 WZ 5-form: zero on SU(2), non-zero on SU(3) ----------
rng = np.random.default_rng(3)
def eps5(p):
    s = 1; p = list(p)
    for i in range(5):
        for j in range(i + 1, 5):
            if p[i] > p[j]: s = -s
    return s
P5 = list(permutations(range(5)))
def five_form(gens):
    """Tr of the antisymmetrised product of five left-invariant forms U^-1 dU evaluated on five tangent directions (Lie algebra)."""
    return abs(sum(eps5(p) * np.trace(gens[p[0]] @ gens[p[1]] @ gens[p[2]] @ gens[p[3]] @ gens[p[4]]) for p in P5))
su2 = [np.array([[0, 1], [1, 0]]) * 0.5j, np.array([[0, -1j], [1j, 0]]) * 0.5j, np.array([[1, 0], [0, -1]]) * 0.5j]
def rand_alg(basis):
    return sum(rng.normal() * b for b in basis)
v2 = max(five_form([rand_alg(su2) for _ in range(5)]) for _ in range(50))
gm = []
for a in range(3):
    for b in range(a + 1, 3):
        E = np.zeros((3, 3), complex); E[a, b] = E[b, a] = 1; gm.append(0.5j * E)
        E = np.zeros((3, 3), complex); E[a, b] = -1j; E[b, a] = 1j; gm.append(0.5j * E)
gm += [0.5j * np.diag([1, -1, 0]), 0.5j * np.diag([1, 1, -2]) / np.sqrt(3)]
v3 = max(five_form([rand_alg(gm) for _ in range(5)]) for _ in range(50))
ok(f"SP3 WZ 5-form Tr(U^-1 dU)^5 on five tangent directions: SU(2) max {v2:.1e} (vanishes: no WZ term, FR sign a free Z_2 angle), "
   f"SU(3) max {v3:.2f} (non-zero: WZ term with integer level)", v2 < 1e-12 and v3 > 1e-3)

print("ALL PASS")
