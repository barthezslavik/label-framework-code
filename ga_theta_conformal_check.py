# Symmetries linking de Sitter and deep MOND (papers/restart/31-theta-sphere.md, section "Symmetries"). C1-C6.
# SO(4,1) is both the isometry group of dS_4 and the conformal group of 3-space; deep MOND div(|grad phi| grad phi) = 4 pi G a0 rho is
# conformally invariant in 3D (the 3-Laplacian in 3 dimensions). Conformal GA of R^3: e1,e2,e3 (+), e+ (+), e- (-) -> R^{4,1};
# n_inf = e- + e+, n_0 = (e- - e+)/2, point P(x) = x + x^2/2 n_inf + n_0, sphere S(c, r) = P(c) - r^2/2 n_inf (S^2 = r^2 > 0: a point of dS).
# Run: uv run --with sympy python ga_theta_conformal_check.py
import random
import sympy as sp
from ga_core import *

C5 = {0: 1, 1: 1, 2: 1, 3: 1, 4: -1}
ninf = {(3,): 1, (4,): 1}; n0 = {(4,): sp.Rational(1, 2), (3,): -sp.Rational(1, 2)}
P = lambda x: add(vec(list(x) + [0, 0]), scal(sum(v**2 for v in x) / 2, ninf), n0)
S = lambda c, r: add(P(c), scal(-r**2 / 2, ninf))
sc = lambda M: sp.simplify(M.get((), 0))
inv = lambda V: scal(1 / sc(gp(V, V, C5)), V)
def point_of(Q):
    w = -sc(gp(Q, ninf, C5))                                       # -(Q . n_inf) (scalar part of the product, Q a vector)
    return [sp.simplify(Q.get((k,), 0) / w) for k in range(3)]

# C1 conformal invariance of the 3-Laplacian in 3D (deep MOND) under inversion s(x) = x/|x|^2: L3(phi o s) = |x|^-6 (L3 phi) o s;
#    the ordinary Laplacian (Newton) fails the same test.
X = sp.symbols('x y z', real=True)
def L3(f): g = [sp.diff(f, v) for v in X]; n = sp.sqrt(sum(q**2 for q in g)); return sum(sp.diff(n * g[i], X[i]) for i in range(3))
def L2(f): return sum(sp.diff(f, v, 2) for v in X)
phi = X[0]**2 * X[1] + X[2]**3 + X[0] * X[2] + 2 * X[1]
r2 = sum(v**2 for v in X); sig = [v / r2 for v in X]
pt = {X[0]: sp.Rational(3, 7), X[1]: -sp.Rational(2, 5), X[2]: sp.Rational(5, 4)}
img = {X[i]: sig[i].subs(pt) for i in range(3)}
lhs3 = sp.N(L3(phi.subs(dict(zip(X, sig)), simultaneous=True)).subs(pt), 30); rhs3 = sp.N((L3(phi).subs(img)) * r2.subs(pt)**-3, 30)
lhs2 = sp.N(L2(phi.subs(dict(zip(X, sig)), simultaneous=True)).subs(pt), 30); rhs2 = sp.N((L2(phi).subs(img)) * r2.subs(pt)**-2, 30)
ok(f"C1 inversion: 3-Laplacian {lhs3:.12g} vs |x|^-6 (L3 phi)(s x) {rhs3:.12g} (equal); Laplacian {lhs2:.6g} vs {rhs2:.6g} (not)",
   abs(lhs3 - rhs3) < 1e-20 and abs(lhs2 - rhs2) > 1e-3)

# C2 a dS point = a sphere; the reflection Q -> S Q S^-1 (geodesic symmetry of dS at S) is the inversion in that sphere.
c = [sp.Rational(1, 2), -1, sp.Rational(3, 2)]; r = sp.Rational(7, 3); x = [2, sp.Rational(1, 3), -1]
Q = gp(gp(S(c, r), P(x), C5), inv(S(c, r)), C5)
dx = [x[i] - c[i] for i in range(3)]; d2 = sum(v**2 for v in dx)
ok(f"C2 S(c,r)^2 = {sc(gp(S(c, r), S(c, r), C5))} = r^2 (spacelike: a point of dS); S P(x) S^-1 is the point {point_of(Q)} "
   f"= c + r^2 (x-c)/|x-c|^2", sc(gp(S(c, r), S(c, r), C5)) == r**2 and point_of(Q) == [sp.simplify(c[i] + r**2 * dx[i] / d2) for i in range(3)])

# C3 the dS boost B (static time of the accelerated observers) is the dilation D = n_0 ^ n_inf = e- e+ of 3-space: D^2 = 1, and
#    exp(-D lam/2) P(x) exp(D lam/2) is the point e^{-lam} x: a pure scaling of x.
D = wedge(n0, ninf, C5)
lam = sp.symbols('lambda', real=True)
Rr = add({(): sp.cosh(lam / 2)}, scal(-sp.sinh(lam / 2), D))
Rrev = add({(): sp.cosh(lam / 2)}, scal(sp.sinh(lam / 2), D))
img3 = point_of(gp(gp(Rr, P(x), C5), Rrev, C5))
ratio = [sp.simplify(img3[i] / x[i]) for i in range(3)]
ok(f"C3 D = n0^n_inf = {D}, D^2 = {gp(D, D, C5)}; the D-rotor maps x to {ratio[0]} x (pure dilation): dS static time = 3D scaling",
   gp(D, D, C5) == {(): 1} and len(set(ratio)) == 1 and sp.simplify(ratio[0] - sp.exp(-lam)) == 0)

# C4 the 45-degree condition in 3D language: for the sphere S(c, r) and the dilation about the origin,
#    <D S D S^-1>_0 = -cos(2 chi) with tan^2 chi = |c|^2/(r^2 - |c|^2): 45 deg <=> r^2 = 2 |c|^2; a sphere centred on the dilation centre is chi = 0.
cc, rr = sp.symbols('c r', positive=True)
Sx = S([cc, 0, 0], rr)
q = sp.simplify(sc(gp(gp(gp(D, Sx, C5), D, C5), inv(Sx), C5)))
chi = sp.atan(cc / sp.sqrt(rr**2 - cc**2))
ok(f"C4 <D S D S^-1>_0 = {q} = -cos(2 chi), tan^2 chi = c^2/(r^2-c^2): zero at r^2 = {sp.solve(q, rr)[0]**2}; centred sphere (c=0) gives {q.subs(cc, 0)} (chi = 0)",
   sp.simplify(q + sp.cos(2 * chi)) == 0 and sp.simplify(sp.solve(q, rr)[0]**2 - 2 * cc**2) == 0 and q.subs(cc, 0) == -1)

# C5 deep MOND around a point mass: inversion in the MOND sphere r_M = sqrt(GM/a0) gives g(r) g(r_M^2/r) = a0^2 and g_N <-> a0^2/g_N --
#    the same form as a <-> H^2/a. But that sphere is centred on the mass (the dilation centre of the solution), i.e. it is a chi = 0 point
#    (C4), the Milgrom point T = T_GH, not the 45-degree knee.
G, M, a0, R_ = sp.symbols('G M a0 R', positive=True)
rM2 = G * M / a0
g = lambda s: sp.sqrt(G * M * a0) / s; gN = lambda s: G * M / s**2
ok(f"C5 g(r) g(r_M^2/r) = {sp.simplify(g(R_) * g(rM2 / R_))}; g_N(r) g_N(r_M^2/r) = {sp.simplify(gN(R_) * gN(rM2 / R_))}; MOND sphere centred on the mass -> chi = 0",
   sp.simplify(g(R_) * g(rM2 / R_) - a0**2) == 0 and sp.simplify(gN(R_) * gN(rM2 / R_) - a0**2) == 0)

# C6 deep MOND is dilation invariant (phi(x) -> phi(lam x): L3 scales as lam^3 on both sides with rho -> lam^3 rho, a0 untouched),
#    so SO(4,1) cannot fix a0: a0 enters only through the Newtonian regime, which breaks the symmetry (Laplacian scales as lam^2, not lam^3).
lm = sp.symbols('l', positive=True)
f = sp.log(sum(v**2 for v in X)) / 2 + X[0]                         # any test function
s3 = sp.simplify(L3(f.subs({v: lm * v for v in X}, simultaneous=True)) / L3(f).subs({v: lm * v for v in X}, simultaneous=True))
s2 = sp.simplify(L2(f.subs({v: lm * v for v in X}, simultaneous=True)) / L2(f).subs({v: lm * v for v in X}, simultaneous=True))
ok(f"C6 under x -> l x: 3-Laplacian scales as {s3}, Laplacian as {s2}; deep MOND has no scale, a0 is fixed only where Newton takes over",
   sp.simplify(s3 - lm**3) == 0 and sp.simplify(s2 - lm**2) == 0)
