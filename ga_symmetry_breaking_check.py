# How Newton breaks the SO(4,1) of deep MOND, and where a0 sits (papers/restart/32-symmetries.md). B1-B6.
# Conformal GA of R^3 as in ga_theta_conformal_check: e1,e2,e3,e+ (+), e- (-); n_inf = e- + e+, n_0 = (e- - e+)/2, n_inf . n_0 = -1.
# Run: uv run --with sympy python ga_symmetry_breaking_check.py
import sympy as sp
from ga_core import *

C5 = {0: 1, 1: 1, 2: 1, 3: 1, 4: -1}
ninf = {(3,): 1, (4,): 1}; n0 = {(4,): sp.Rational(1, 2), (3,): -sp.Rational(1, 2)}
P = lambda x: add(vec(list(x) + [0, 0]), scal(sum(v**2 for v in x) / 2, ninf), n0)
S = lambda c, r: add(P(c), scal(-r**2 / 2, ninf))
sc = lambda M: sp.simplify(M.get((), 0))
simp = lambda M: {k: w for k, w in ((k, sp.simplify(v)) for k, v in M.items()) if w != 0}
rev = lambda M: {k: (v if len(k) % 4 in (0, 1) else -v) for k, v in M.items()}
act = lambda V, Y: simp(gp(gp(V, Y, C5), rev(V), C5))               # versor action (rotors)
X = sp.symbols('x y z', real=True)
r2 = sum(v**2 for v in X); sig = [v / r2 for v in X]
def L3(f): g = [sp.diff(f, v) for v in X]; n = sp.sqrt(sum(q**2 for q in g)); return sum(sp.diff(n * g[i], X[i]) for i in range(3))
def L2(f): return sum(sp.diff(f, v, 2) for v in X)
pt = {X[0]: sp.Rational(3, 7), X[1]: -sp.Rational(2, 5), X[2]: sp.Rational(5, 4)}
img = {X[i]: sig[i].subs(pt) for i in range(3)}
phi = X[0]**2 * X[1] + X[2]**3 + X[0] * X[2] + 2 * X[1]
N = lambda e: sp.N(e, 30)

# B1 conformal weights. Mass conservation under inversion needs rho' = |x|^-6 rho o s (d^3x' = |x|^-6 d^3x).
#    Deep MOND: L3(phi o s) = |x|^-6 (L3 phi) o s  -> potential weight 0, mass conserved.
#    Newton: the Kelvin transform K phi = |x|^-1 phi o s gives L2(K phi) = |x|^-5 (L2 phi) o s -> mass changes by |x|: not conserved.
kel = (phi.subs(dict(zip(X, sig)), simultaneous=True)) / sp.sqrt(r2)
newton_w = N(L2(kel).subs(pt) / (L2(phi).subs(img)))
mond_w = N(L3(phi.subs(dict(zip(X, sig)), simultaneous=True)).subs(pt) / (L3(phi).subs(img)))
rr = N(r2.subs(pt))
ok(f"B1 inversion weights: deep MOND {mond_w:.10g} = |x|^-6 = {rr**-3:.10g}; Newton (Kelvin) {newton_w:.10g} = |x|^-5 = {rr**sp.Rational(-5, 2):.10g} "
   f"!= |x|^-6: with mass conserved only deep MOND carries the inversion", abs(mond_w - rr**-3) < 1e-20 and abs(newton_w - rr**sp.Rational(-5, 2)) < 1e-20)

# B2 Newton's symmetry is the stabiliser of the point at infinity [n_inf]: translations T_a = 1 + a n_inf/2 fix n_inf, the dilation rotor
#    rescales it, rotations fix it; the inversion (reflection in e+) sends n_inf to 2 n_0. Stab([n_inf]) = Sim(3).
a1 = sp.Rational(2, 3)
Ta = add({(): 1}, scal(sp.Rational(1, 2), gp(vec([a1, 0, 0]), ninf, C5)))
lam = sp.symbols('lambda', real=True)
D = wedge(n0, ninf, C5)
Rd = add({(): sp.cosh(lam / 2)}, scal(-sp.sinh(lam / 2), D))
Rrot = add({(): sp.cos(lam / 2)}, scal(-sp.sin(lam / 2), {(0, 1): 1}))
ep = {(3,): 1}
inv_ninf = simp(scal(-1, gp(gp(ep, ninf, C5), ep, C5)))              # -e+ n_inf e+^-1 (reflection in e+)
dil = act(Rd, ninf)
ok(f"B2 T_a n_inf T_a~ = {act(Ta, ninf)}, rotation: {act(Rrot, ninf)}, dilation: {dil} (= e^-lam n_inf: rescaled, fixed as a point), "
   f"inversion: {inv_ninf} = 2 n_0",
   act(Ta, ninf) == ninf and act(Rrot, ninf) == ninf and sp.simplify(dil.get((3,), 0) - dil.get((4,), 0)) == 0
   and simp(add(inv_ninf, scal(-2, n0))) == {})

# B3 a0 is a spurion of the broken dilation: with GM fixed, x -> l x maps MOND(a0) to MOND(a0/l^2); deep part (weight 0) and Newton part
#    (weight 1) can only coexist if a0 moves. Check on g_MOND(r) = g_N + sqrt(g_N a0) (any interpolation of this type).
G, M, a0, r, l = sp.symbols('G M a0 r l', positive=True)
gN = G * M / r**2
gm = lambda a, s: gN.subs(r, s) / 2 + sp.sqrt(gN.subs(r, s)**2 / 4 + gN.subs(r, s) * a)        # 'simple' mu: g = g_N/2 + sqrt(g_N^2/4 + g_N a0)
# the dilated field must be g'(r) = l^-2 g(r/l) for the Newton part; it is a MOND field with a0' = a0/l^2:
ok(f"B3 l^-2 g(r/l; a0) = g(r; a0/l^2): {sp.simplify(gm(a0, r / l) / l**2 - gm(a0 / l**2, r)) == 0}: dilations act on a0 with weight 2 (a0 ~ 1/length^2 at fixed GM)",
   sp.simplify(gm(a0, r / l) / l**2 - gm(a0 / l**2, r)) == 0)

# B4 dictionary: the galaxy's two special points (mass at n_0, infinity n_inf) are the null eigenvectors of the dilation D; spheres centred on
#    the mass lie in span(n_0, n_inf) = the B-plane, i.e. on the chi = 0 worldline, and the dilation moves them along it (radius r -> e^-lam r).
ev0 = simp(gp(D, n0, C5)); evinf = simp(gp(D, ninf, C5))
Sc = S([0, 0, 0], r)
moved = act(Rd, Sc)
wgt = -sc(gp(moved, ninf, C5))                                      # -(S' . n_inf): normalisation of the moved sphere
rad2 = sp.simplify(sc(gp(moved, moved, C5)) / wgt**2)               # radius^2 of the moved sphere
ok(f"B4 D n_0 = {ev0}, D n_inf = {evinf} (null eigenvectors: mass point and infinity); S(0,r) has only n_0, n_inf parts (chi = 0); "
   f"the D-rotor moves it to a centred sphere of radius^2 {rad2} (along the chi = 0 worldline)",
   (simp(add(ev0, n0)) == {} or simp(add(ev0, scal(-1, n0))) == {}) and (simp(add(evinf, ninf)) == {} or simp(add(evinf, scal(-1, ninf))) == {})
   and set(simp(Sc)) <= {(3,), (4,)} and set(moved) <= {(3,), (4,)} and sp.simplify(rad2 - r**2 * sp.exp(-2 * lam)) == 0)

# B5 the anchor: the MOND inversion sphere is S(0, r_M) (B4: chi = 0), so in the dictionary the galaxy is the inertial dS observer and its
#    natural temperature is T_GH (Milgrom's point). A 45-degree point would be a sphere of radius sqrt2 |c| about an off-centre point c,
#    which the point-mass problem does not supply.
ok("B5 MOND sphere centred on the mass -> chi = 0 (C4 of ga_theta_conformal_check); the 45-degree sphere r = sqrt2 |c| needs an off-centre c",
   set(simp(S([0, 0, 0], sp.sqrt(G * M / a0)))) <= {(3,), (4,)})

# B6 (statement, no computation) no scale from the group: SO(4,1) of R^3 is scale-free, so matching it with the isometries of dS_l relates
#    no length to l. Only an extra identification of galactic lengths with dS lengths gives a0 = k c^2/l, and k stays free.
