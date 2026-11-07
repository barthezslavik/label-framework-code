# 147: step 6 of paper12 (internal quantum numbers). Label budget of the internal sector: what S removes and what stays a label.
# Premises: I1-I3 of ga_internal_check.py; one generation for IB1, three for IB2.
# IB1 abelian charges of one generation (Q, u, d, L, e, n): with no condition five ratios are free; the anomaly conditions (SU(3)^2 U(1),
#     SU(2)^2 U(1), gravitational, U(1)^3) leave three planes through the origin, i.e. three one-parameter families of ratios; the SM
#     hypercharge and B - L lie in the same plane (U(1)_X = Y + x (B - L)); the two-dimensional algebra plane of IN1 lies in no anomaly
#     plane and meets the Y plane exactly on the S line, so S and anomaly freedom agree there and leave one point, Y:
#     S removes the last continuous charge parameter;
# IB2 physical parameters of the Yukawa (D_F) sector by orbit dimension of the flavour group at a random point: quarks 36 - 26 = 10,
#     charged leptons 18 - 15 = 3, leptons with nu_R and a Majorana matrix 48 - 27 = 21: the free values in D_F are exactly those of the
#     Standard Model with a type-I seesaw; neither S nor the algebra fixes any;
# IB3 the SM label count 3 gauge couplings + theta + mu^2 + lambda + 13 flavour values = 19 (with nu_R and Majorana: 19 - 3 + 21 = 37)
#     is unchanged by S; only the choice of group and charges moves from label to form.
# Run: ~/.venvs/jev/bin/python ga_internal_budget_check.py
import numpy as np
import sympy as sp

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

# ---------- IB1 charge freedom ----------
Q, u, d, L, e, n = sp.symbols('Q u d L e n')
lin = [2 * Q - u - d, 3 * Q + L, 6 * Q + 2 * L - 3 * u - 3 * d - e - n]   # SU(3)^2, SU(2)^2, grav (left minus right)
sol = sp.solve(lin, [L, d, n], dict=True)[0]
cub = sp.factor(sp.expand((6 * Q**3 + 2 * L**3 - 3 * u**3 - 3 * d**3 - e**3 - n**3).subs(sol)))
facs = [f for f, _ in sp.factor_list(cub)[1]]
planes = [f for f in facs if sp.Poly(f, Q, u, e).total_degree() == 1]
Ysm = {Q: sp.Rational(1, 3), u: sp.Rational(4, 3), e: -2}
BL = {Q: sp.Rational(1, 3), u: sp.Rational(1, 3), e: -1}
inY = [f for f in planes if f.subs(Ysm) == 0]; inBL = [f for f in planes if f.subs(BL) == 0]
# algebra plane (IN1): charges A = (Q 0, u 1, e -2 ...) and B = (Q -1, u -1, e 0) in the units of ga_internal_check; S line beta = -alpha/3
alg = lambda al, be: {Q: -be, u: al - be, e: -2 * al}
yS = alg(1, sp.Rational(-1, 3))
a_, b_ = sp.symbols('a b')
on_planes = [f for f in planes if sp.expand(f.subs(alg(a_, b_))) == 0]
Yplane = inY[0]
meet = sp.solve(sp.expand(Yplane.subs(alg(a_, b_))), b_)               # algebra plane meets the Y plane on a line
ok(f"IB1 linear anomaly conditions leave a 3-dim space; U(1)^3 factors as {cub}: {len(planes)} planes (one-parameter families of "
   f"ratios); the plane of SM Y also contains B-L (U(1)_X = Y + x(B-L)); the algebra plane lies in no anomaly plane and meets the Y "
   f"plane on the line b = {meet[0]}, which is the S line; S picks Y: Q {yS[Q]}, u {yS[u]}, e {yS[e]}",
   len(planes) == 3 and len(inY) == 1 and Yplane in inBL and yS == Ysm and on_planes == [] and meet == [-a_ / 3])

# ---------- IB2 physical parameters of D_F by orbit dimension ----------
rng = np.random.default_rng(11)
def herm_basis(k):
    B = []
    for i in range(k):
        E = np.zeros((k, k), complex); E[i, i] = 1j; B.append(E)
        for j in range(i + 1, k):
            E = np.zeros((k, k), complex); E[i, j] = 1; E[j, i] = -1; B.append(E)
            E = np.zeros((k, k), complex); E[i, j] = 1j; E[j, i] = 1j; B.append(E)
    return B                                                           # basis of u(k), dim k^2
U3 = herm_basis(3)
def rc():
    return rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
def count(params, actions):
    """params: list of matrices; actions: list of (group index list) -> tangent map. Returns (real dim, orbit dim, physical)."""
    dim = sum(2 * M.size if not sym else 12 for M, sym in params)
    cols = []
    for gen in actions:
        v = np.concatenate([np.concatenate([t.real.ravel(), t.imag.ravel()]) for t in gen])
        cols.append(v)
    orb = np.linalg.matrix_rank(np.array(cols).T, tol=1e-9)
    return dim, orb, dim - orb
# quarks: Y_u, Y_d with Q -> U_Q, u -> U_u, d -> U_d: Y_u -> U_Q Y_u U_u^dag, Y_d -> U_Q Y_d U_d^dag
Yu, Yd = rc(), rc()
acts = []
for g in U3: acts.append([g @ Yu, g @ Yd])                            # U_Q
for g in U3: acts.append([-Yu @ g, 0 * Yd])                           # U_u
for g in U3: acts.append([0 * Yu, -Yd @ g])                           # U_d
q = count([(Yu, False), (Yd, False)], acts)
# charged leptons only: Y_e with L, e
Ye = rc(); acts = []
for g in U3: acts.append([g @ Ye])
for g in U3: acts.append([-Ye @ g])
l0 = count([(Ye, False)], acts)
# leptons with nu_R: Y_e, Y_nu, M_R (complex symmetric): L, e, N
Yn = rc(); M = rc(); M = M + M.T
acts = []
for g in U3: acts.append([g @ Ye, g @ Yn, 0 * M])
for g in U3: acts.append([-Ye @ g, 0 * Yn, 0 * M])
for g in U3: acts.append([0 * Ye, -Yn @ g, -(g.T @ M + M @ g)])        # N -> U_N: Y_nu U_N^dag, M -> U_N^* M U_N^dag
def count_sym(acts):
    # M symmetric: 12 real parameters; project tangent onto the upper triangle
    iu = np.triu_indices(3)
    cols = []
    for t in acts:
        v = np.concatenate([np.concatenate([x.real.ravel(), x.imag.ravel()]) for x in t[:2]]
                           + [np.concatenate([t[2][iu].real, t[2][iu].imag])])
        cols.append(v)
    orb = np.linalg.matrix_rank(np.array(cols).T, tol=1e-9)
    dim = 18 + 18 + 12
    return dim, orb, dim - orb
l1 = count_sym(acts)
ok(f"IB2 physical D_F parameters (dim - orbit dim): quarks {q[0]} - {q[1]} = {q[2]}, charged leptons {l0[0]} - {l0[1]} = {l0[2]}, "
   f"leptons with nu_R and Majorana {l1[0]} - {l1[1]} = {l1[2]}: exactly the SM (+ type-I seesaw) flavour values; S fixes none",
   q[2] == 10 and l0[2] == 3 and l1[2] == 21)

# ---------- IB3 the SM label count is unchanged by S ----------
sm = 3 + 1 + 2 + q[2] + l0[2]
smnu = sm - l0[2] + l1[2]
ok(f"IB3 SM labels: 3 couplings + theta + (mu^2, lambda) + {q[2]} + {l0[2]} = {sm}; with nu_R and Majorana {smnu}; S removes 0 values "
   f"(it moves only the group and the charge ratios from label to form, IB1)", sm == 19 and smnu == 37)

print("ALL PASS")
