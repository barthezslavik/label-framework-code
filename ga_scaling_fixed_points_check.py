# One scaling group, two roles (papers/restart/34-sdiff-rotation.md, "One group"): profiles inside a galaxy are fixed points of the
# one-parameter subgroups mu = lambda^p of the Vlasov-Poisson scaling (x -> lambda x, v -> mu v, M -> lambda mu^2 M); populations across
# galaxies are orbits of the same subgroups. G1-G4.
# Run: uv run --with sympy python ga_scaling_fixed_points_check.py
import sympy as sp
from ga_core import ok

r, p, q, lam = sp.symbols('r p q lambda', positive=True)
pp = sp.symbols('p', real=True)

# G1 fixed points: a profile v ~ r^p is invariant under mu = lambda^p, with M(r) = v^2 r/G ~ r^(1+2p).
#    p = 1: solid body (M ~ r^3, uniform core); p = 0: flat (M ~ r, isothermal); p = -1/2: Kepler (M fixed, point mass).
names = {1: 'solid body, M ~ r^3', 0: 'flat, M ~ r', sp.Rational(-1, 2): 'Kepler, M = const'}
Mexp = {k: 1 + 2 * k for k in names}
ok(f"G1 v ~ r^p invariant under mu = lambda^p; M ~ r^(1+2p): " + "; ".join(f"p = {k}: {Mexp[k]} ({v})" for k, v in names.items()),
   Mexp[1] == 3 and Mexp[0] == 1 and Mexp[sp.Rational(-1, 2)] == 0)

# G2 RAR slope between two fixed points: observed profile p, baryons q: g ~ r^(2p-1), so d ln g_obs / d ln g_bar = (2p-1)/(2q-1).
#    Inner (p = q): slope 1 (Newton). Outer: observed flat (p = 0) over Keplerian baryons (q = -1/2): slope 1/2 -- the 'deep-MOND' sqrt law
#    follows from flat curves + finite baryons, and its normalisation per galaxy is v_f^4/(G M_b), i.e. the BTFR.
s = lambda p_, q_: (2 * p_ - 1) / (2 * q_ - 1)
ok(f"G2 RAR slope (2p-1)/(2q-1): p = q -> {sp.simplify(s(pp, pp))}; flat over Kepler -> {s(0, sp.Rational(-1, 2))}",
   sp.simplify(s(pp, pp)) == 1 and s(0, sp.Rational(-1, 2)) == sp.Rational(1, 2))

# G3 populations: galaxies spread along mu = lambda^p give M ~ v^((1+2p)/p): p = 1 (fixed density, halos formed at one epoch: Lambda-CDM
#    virial M ~ V^3), p = 1/2 (fixed acceleration/surface density: Freeman, slope 4, invariant 'a0'), observed 3.62 -> p = 0.617.
slope = (1 + 2 * pp) / pp
ok(f"G3 population slope (1+2p)/p: p = 1 -> {slope.subs(pp, 1)}, p = 1/2 -> {slope.subs(pp, sp.Rational(1, 2))}, slope 3.62 -> p = "
   f"{float(sp.solve(sp.Eq(slope, sp.Rational(362, 100)), pp)[0]):.3f}",
   slope.subs(pp, 1) == 3 and slope.subs(pp, sp.Rational(1, 2)) == 4)

# G4 what the group cannot give: Newtonian self-gravity + cosmology has G, H (and rho_crit = 3H^2/8piG, the same scale); an acceleration
#    needs a length in addition (G rho_crit L ~ H^2 L). With c the only one is c H; otherwise the length comes from baryon physics.
G, H, c, L = sp.symbols('G H c L', positive=True)
ok(f"G4 accelerations from (G, H, L): G*rho_crit*L ~ {sp.simplify(G * 3 * H**2 / (8 * sp.pi * G) * L)}; from (H, c): c H. "
   "No acceleration from G and H alone", sp.simplify(G * 3 * H**2 / (8 * sp.pi * G) * L - sp.Rational(3, 8) / sp.pi * H**2 * L) == 0)
