# 74: the native line of 71 (FC1-FC16), without imported laws: the dust label blade around a centre factorises, N3 = d psi ^ N2, into a
#     radial k = 1 label and the angular k = 2 blade N2 (= the Gauss field blade, FC15). LF1-LF4. Structural; no data.
# Run: ~/.venvs/jev/bin/python ga_label_factor_check.py
import sympy as sp

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

r, th, ph = sp.symbols('r theta phi', positive=True)
psi = sp.Function('psi')
# spatial metric in spherical coordinates; label densities n_k = sqrt(det of the Gram matrix of the label gradients) x measure
gs = sp.diag(1, r**2, r**2 * sp.sin(th)**2); gi = gs.inv(); X = [r, th, ph]
def dens(labels, f=1):
    J = sp.Matrix(len(labels), 3, lambda a, m: sp.diff(labels[a], X[m]))
    dp, st = sp.symbols('dp st', positive=True)                          # psi' > 0 (labels increase outward); sin(theta) > 0
    D = (J * gi * J.T).det().subs(sp.Derivative(psi(r), r), dp).subs(sp.sin(th), st)
    return sp.simplify(f * sp.sqrt(sp.simplify(D))).subs(st, sp.sin(th)).subs(dp, sp.Derivative(psi(r), r))
n1 = dens([psi(r)])                       # radial label density |grad psi|
n2 = sp.simplify(dens([th, ph], sp.sin(th)))           # angular blade with the uniform sphere measure
n3 = dens([psi(r), th, ph], sp.sin(th))   # full dust density

# LF1 factorisation: n3 = n1 n2 (orthogonal labels), with n2 = 1/r^2 universal.
ok(f"LF1 n3 = n1 * n2: n1 = {n1}, n2 = {n2}, n3 = {n3}", sp.simplify(n3 - n1 * n2) == 0 and sp.simplify(n2 - 1 / r**2) == 0)

# LF2 two uniformities: uniform at the k = 3 level (n3 = const) <=> n1 ~ r^2 (psi ~ r^3: ball, rigid rotation); uniform at the k = 1
#     level (n1 = const) <=> n3 ~ 1/r^2 (psi ~ r: plateau). One structure, the split 3 = 1 + 2, gives both regimes.
c1, c3 = sp.symbols('c1 c3', positive=True)
ball = sp.dsolve(sp.Eq(n3, c3)); plat = sp.dsolve(sp.Eq(n1, c1))
ok(f"LF2 n3 = c3 -> {ball}; n1 = c1 -> {plat}, n3 = {sp.simplify(n3.subs(psi(r), plat.rhs).doit())}",
   sp.simplify(sp.diff(ball.rhs, r) - c3 * r**2) == 0 and sp.simplify(n3.subs(psi(r), plat.rhs).doit() - c1 / r**2) == 0)

# LF3 the radial label is the enclosed mass: Gauss in forms, *g = -G M(r) N2 and d(*g) = -G M'(r) dr ^ N2 = -4 pi G rho vol; comparing
#     with the dust 3-form rho vol = (1/4pi) d psi ^ N2 (unit mass per label volume) gives psi = M(r) up to a constant: no choice of label.
G = sp.symbols('G', positive=True); M = sp.Function('M')
rho_from_gauss = sp.diff(M(r), r) / (4 * sp.pi * r**2)                  # from d(*g) = -4 pi G rho vol with vol = r^2 dr ^ N2
rho_from_label = (n3 / (4 * sp.pi)).subs(psi(r), M(r)).doit()
ok(f"LF3 Gauss: rho = M'/(4 pi r^2); dust with radial label psi = M: rho = {sp.simplify(rho_from_label)} -- identical",
   sp.simplify(rho_from_gauss - rho_from_label) == 0)

# LF4 where the two uniformities meet, n3_ball = n3_plateau: c3 = c1 n2, i.e. on a level set of the k = 2 density, n2 = c3/c1.
#     The switch between k = 3- and k = 1-uniformity is a condition on the remaining factor, the k = 2 blade: the rung the BTFR slope
#     selected in 71 (FC12-FC13). Weighting N2 by the flux M_b (the field blade) and requiring continuity gives FC14.
rc = sp.solve(sp.Eq(c3, c1 * n2), r)
ok(f"LF4 the two uniformities meet where n2 = c3/c1, r_c = {rc}: the switch sits on the k = 2 rung", len(rc) == 1 and sp.simplify(rc[0] - sp.sqrt(c1 / c3)) == 0)
