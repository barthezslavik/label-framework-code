# 131: is G a curvature of one extra dimension? Three known routes (equations borrowed, every step recomputed here):
# GX1 Kaluza-Klein on a circle of radius R: R^(5) = R^(4) for a constant radius, so 1/(16 pi G4) = 2 pi R/(16 pi G5): G4 = G5/(2 pi R);
#     the circle is flat, G4 is set by its size and by G5.
# GX2 Randall-Sundrum II: the warp metric e^{-2|y|/l} eta + dy^2 solves the 5D Einstein equation with Lambda5 = -6/l^2 (AdS5, curvature
#     radius l); the zero mode gives 1/(16 pi G4) = l/(16 pi G5): G4 = G5/l, and Lambda4 = 0 needs the brane tension tuned.
# GX3 MacDowell-Mansouri (the de Sitter group SO(1,4), the Cl(1,4) phase of this project; 4D spacetime, the fifth dimension only embeds dS): F = R - (1/l^2)(g g - g g); the Gauss-Bonnet
#     density of F is GB(R) - (4/l^2)(R - 6/l^2), so S = k int GB(F) is Einstein-Hilbert with Lambda = 3/l^2 and k = -l^2/(64 pi G).
#     The radius of the extra dimension is fixed by Lambda, not by G; the only dimensionless coupling is alpha = G Lambda/3 = (l_P/l)^2.
# GX4 label count (an argument over GX1-GX3, not a computation): in every route G needs one more label besides the length: G5 or alpha stays.  Run: ~/.venvs/jev/bin/python ga_g_curvature_check.py
import itertools
import sympy as sp

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

def geometry(g, X):
    n = len(X)
    gi = g.inv()
    Gam = [[[sp.simplify(sum(gi[i, l] * (sp.diff(g[l, j], X[k]) + sp.diff(g[l, k], X[j]) - sp.diff(g[j, k], X[l])) for l in range(n)) / 2)
             for k in range(n)] for j in range(n)] for i in range(n)]
    def riem(a, b, c, d):                     # R^a_{bcd}
        return sp.diff(Gam[a][b][d], X[c]) - sp.diff(Gam[a][b][c], X[d]) + sum(Gam[a][c][e] * Gam[e][b][d] - Gam[a][d][e] * Gam[e][b][c] for e in range(n))
    Rup = {}
    for a, b, c, d in itertools.product(range(n), repeat=4):
        Rup[a, b, c, d] = sp.simplify(riem(a, b, c, d)) if c < d else None
    def Rud(a, b, c, d):
        if c == d: return 0
        return Rup[a, b, c, d] if c < d else -Rup[a, b, d, c]
    Ric = sp.Matrix(n, n, lambda b, d: sp.simplify(sum(Rud(a, b, a, d) for a in range(n))))
    Rs = sp.simplify(sum(gi[i, j] * Ric[i, j] for i in range(n) for j in range(n)))
    return gi, Rud, Ric, Rs

# ---------- GX1 Kaluza-Klein: FLRW x circle of constant radius ----------
t, x, y, z, th = sp.symbols('t x y z theta', real=True)
a = sp.Function('a')(t)
Rc, G5 = sp.symbols('R_c G_5', positive=True)
_, _, _, R4 = geometry(sp.diag(-1, a**2, a**2, a**2), [t, x, y, z])
_, _, _, R5 = geometry(sp.diag(-1, a**2, a**2, a**2, Rc**2), [t, x, y, z, th])
G4kk = sp.solve(sp.Eq(1 / (16 * sp.pi * sp.Symbol('G4')), 2 * sp.pi * Rc / (16 * sp.pi * G5)), sp.Symbol('G4'))[0]
ok(f"GX1 Kaluza-Klein: R^(5) - R^(4) = {sp.simplify(R5 - R4)} for a constant radius; integrating the circle gives G4 = {G4kk}: set by the "
   "size of a flat circle and by G5", sp.simplify(R5 - R4) == 0 and sp.simplify(G4kk - G5 / (2 * sp.pi * Rc)) == 0)
# a varying radius R_c(t) adds derivative terms: the radius is a field (radion) whose value is initial data unless stabilised
Rt = sp.Function('R')(t)
_, _, _, R5v = geometry(sp.diag(-1, a**2, a**2, a**2, Rt**2), [t, x, y, z, th])
extra = sp.simplify(R5v - R4)
ok(f"GX1b a time-dependent radius adds {extra}: the radius is a dynamical modulus, its value is a label", extra != 0)

# ---------- GX2 Randall-Sundrum II warp metric (bulk y > 0) ----------
l = sp.symbols('l', positive=True)
Y = sp.symbols('y5', positive=True)
w = sp.exp(-2 * Y / l)
gRS = sp.diag(-w, w, w, w, 1)
XRS = [t, x, y, z, Y]
giRS, _, RicRS, RsRS = geometry(gRS, XRS)
Lam5 = -6 / l**2
EinRS = sp.simplify(RicRS - RsRS / 2 * gRS + Lam5 * gRS)
ok(f"GX2 warp metric e^(-2y/l) eta + dy^2: G_AB + Lambda5 g_AB = 0 with Lambda5 = -6/l^2 (all entries zero: {EinRS == sp.zeros(5, 5)}); "
   f"R^(5) = {RsRS}, i.e. AdS5 of curvature radius l", EinRS == sp.zeros(5, 5) and sp.simplify(RsRS + 20 / l**2) == 0)
G4s = sp.Symbol('G4')
# zero mode h_mn(x) in ds^2 = e^{-2|y|/l} (eta + h) + dy^2: sqrt(-g5) R5 contains e^{-4|y|/l} e^{+2|y|/l} R4[eta + h] -> weight e^{-2|y|/l}
weight = 2 * sp.integrate(sp.exp(-2 * Y / l), (Y, 0, sp.oo))
G4rs = sp.solve(sp.Eq(1 / (16 * sp.pi * G4s), weight / (16 * sp.pi * G5)), G4s)[0]
ok(f"GX2b zero-mode weight int e^(-2|y|/l) dy = {weight} (both sides of the brane): G4 = {G4rs}; G4 is set by the bulk curvature radius l and G5",
   sp.simplify(G4rs - G5 / l) == 0)
# Lambda4 = 0 needs the Israel jump of the extrinsic curvature to match the tension: K_mn jump = -(2/l) g_mn on both sides
sig = sp.symbols('sigma', positive=True)
# junction: [K_mn] - g [K] = -8 pi G5 sigma (-g_mn) for a pure-tension brane; K_mn = -(1/l) g_mn on each side, [K_mn] = -(2/l) g_mn, [K] = -8/l
jump = sp.Rational(-2) / l - (-8 / l)                       # coefficient of g_mn in [K_mn] - g_mn [K]
sig_crit = sp.solve(sp.Eq(jump, 8 * sp.pi * G5 * sig), sig)[0]
ok(f"GX2c flat brane (Lambda4 = 0) needs the tension sigma = {sig_crit}: the curvature radius l and the tension are tied by one tuned "
   "condition; a detuned tension gives Lambda4 != 0", sp.simplify(sig_crit - 3 / (4 * sp.pi * G5 * l)) == 0)

# ---------- GX3 MacDowell-Mansouri: Gauss-Bonnet density of F = R - (1/l^2)(g g - g g) ----------
# check on a curved 4D metric with all curvature components present: FLRW with spatial curvature
K = sp.symbols('K', real=True)
r = sp.symbols('r', positive=True)
g4 = sp.diag(-1, a**2 / (1 - K * r**2), a**2 * r**2, a**2 * r**2 * sp.sin(y)**2)
X4 = [t, r, y, z]
gi4, Rud4, Ric4, Rs4 = geometry(g4, X4)
n = 4
def lower(T):                                               # R_{abcd} = g_ae R^e_bcd
    return {(a_, b, c, d): sp.simplify(sum(g4[a_, e] * T(e, b, c, d) for e in range(n))) for a_, b, c, d in itertools.product(range(n), repeat=4)}
Rl = lower(Rud4)
Gl = {(a_, b, c, d): g4[a_, c] * g4[b, d] - g4[a_, d] * g4[b, c] for a_, b, c, d in itertools.product(range(n), repeat=4)}
def gb(T):
    Ric_ = sp.Matrix(n, n, lambda b, d: sum(gi4[a_, c] * T[a_, b, c, d] for a_ in range(n) for c in range(n)))
    S_ = sum(gi4[i, j] * Ric_[i, j] for i in range(n) for j in range(n))
    ric2 = sum(gi4[i, k] * gi4[j, m] * Ric_[i, j] * Ric_[k, m] for i, j, k, m in itertools.product(range(n), repeat=4))
    # metric is diagonal: raise each index with the diagonal inverse
    riem2 = sum(T[a_, b, c, d]**2 * gi4[a_, a_] * gi4[b, b] * gi4[c, c] * gi4[d, d] for a_, b, c, d in itertools.product(range(n), repeat=4))
    return sp.simplify(S_**2 - 4 * ric2 + riem2)
Fl = {k_: Rl[k_] - Gl[k_] / l**2 for k_ in Rl}
diff_gb = sp.simplify(gb(Fl) - gb(Rl) + 4 / l**2 * (Rs4 - 6 / l**2))
ok(f"GX3 MacDowell-Mansouri: GB(F) = GB(R) - (4/l^2)(R - 6/l^2) on curved FLRW (residual {diff_gb})", diff_gb == 0)
Gs, Lam = sp.symbols('G Lambda', positive=True)
kMM = sp.symbols('k', real=True)
sol = sp.solve([sp.Eq(-4 * kMM / l**2, 1 / (16 * sp.pi * Gs)), sp.Eq(2 * Lam, 6 / l**2)], [kMM, Lam], dict=True)[0]
alpha = sp.simplify(Gs * sol[Lam] / 3)
ok(f"GX3b matching -4k/l^2 (R - 6/l^2) to (R - 2 Lambda)/(16 pi G): Lambda = {sol[Lam]}, k = {sol[kMM]}; the radius l of the de Sitter "
   f"extra dimension is fixed by Lambda; the one dimensionless coupling is G Lambda/3 = {alpha} = (l_P/l)^2",
   sp.simplify(sol[Lam] - 3 / l**2) == 0 and sp.simplify(sol[kMM] + l**2 / (64 * sp.pi * Gs)) == 0 and sp.simplify(alpha - Gs / l**2) == 0)
import math
LamlP2 = 2.9e-122                                           # Lambda l_P^2 today (Planck 2018, as a dimensionless label)
ok(f"GX3c with today's Lambda: G = alpha l^2 needs alpha = Lambda l_P^2/3 = {LamlP2 / 3:.1e}; G is the inverse curvature l^2 of de Sitter "
   "(an embedding dimension, not a physical one) times a label of order 1e-122", LamlP2 / 3 < 1e-120)

# ---------- GX4 label count ----------
routes = {'Kaluza-Klein': ('R_c (size, flat)', 'G5'), 'Randall-Sundrum II': ('l (AdS5 curvature)', 'G5 and the tension tuning'),
          'MacDowell-Mansouri': ('l = sqrt(3/Lambda) (dS curvature)', 'alpha = G Lambda/3')}
for k_, (geo, lab) in routes.items():
    print(f"   {k_}: G4 from the length {geo} and the label {lab}")
ok("GX4 (parameter count over GX1-GX3, an argument, not a computation) in all three routes G4 is fixed only together with one more label (G5, or alpha = G Lambda/3) besides the length; "
   "the budget does not shrink: G is traded for G5 or for alpha", True)
