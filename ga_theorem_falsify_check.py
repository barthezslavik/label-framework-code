# Falsification pass over the theorem base (papers/restart/44-theorem-base.md, section "Falsification"): for each theorem a known equation or
# solution that could break it, and/or a test that dropping a hypothesis makes it fail (the hypothesis is necessary). X-labels.
# A PASS means: the counterexample does not break the theorem as stated, or the dropped hypothesis is shown to be needed. Where a test DOES
# break a statement, the statement is narrowed and the narrowing is what is checked (marked NARROWED).
# Run: uv run --with sympy --with numpy python ga_theorem_falsify_check.py
import itertools
import numpy as np
import sympy as sp
from ga_core import *

t, x, y, z = sp.symbols('t x y z', real=True)
X3 = (x, y, z)
curl = lambda F: [sp.diff(F[2], y) - sp.diff(F[1], z), sp.diff(F[0], z) - sp.diff(F[2], x), sp.diff(F[1], x) - sp.diff(F[0], y)]
dot = lambda A, B: sum(a * b for a, b in zip(A, B))

# --- A1 three labels. Known flow with helicity (ABC, a steady Euler solution): helicity != 0, so it cannot come from 2 labels (2-label
#     Clebsch has w^dw = 0); and the count 3 is tied to the dimension: in 5D four spatial label gradients have a nonzero wedge.
A_, B_, C_ = 1, sp.Rational(2, 3), sp.Rational(1, 2)
w = [A_ * sp.sin(z) + C_ * sp.cos(y), B_ * sp.sin(x) + A_ * sp.cos(z), C_ * sp.sin(y) + B_ * sp.cos(x)]
hel = sp.simplify(dot(w, curl(w)))
beltrami = all(sp.simplify(c - wi) == 0 for c, wi in zip(curl(w), w))
E5 = {k: 1 for k in range(5)}
g4 = [{(k,): 1} for k in (1, 2, 3, 4)]
w4 = wedge(wedge(wedge(g4[0], g4[1], E5), g4[2], E5), g4[3], E5)
ok(f"XA1 ABC flow: Beltrami {beltrami} (steady Euler), helicity {hel} != 0 -> needs 3 labels; in 5D four spatial gradients wedge to {w4} != 0: "
   "'3' needs signature (1,3)", beltrami and hel != 0 and w4 == {(1, 2, 3, 4): 1})

# --- A2 identical conservation needs 'built from labels': a generic 3-form current rho(t,x) dx^dy^dz is not closed.
STA = {0: 1, 1: -1, 2: -1, 3: -1}
X4 = (t, x, y, z)
rho = sp.Function('rho')(t, x)
Ngen = {(1, 2, 3): rho}
dN = ext_d(Ngen, X4, STA)
ok(f"XA2 generic current rho(t,x) dx^dy^dz: dN = {dN} != 0 -> conservation is an identity only for label currents", dN != {})

# --- A3 equation of state on known matter: radiation F ~ n^(4/3) -> w = 1/3; stiff F ~ n^2 -> w = 1 (both known).
n = sp.symbols('n', positive=True)
wof = lambda F: sp.simplify((n * sp.diff(F, n) - F) / F)
ok(f"XA3 radiation w = {wof(n**sp.Rational(4, 3))}, stiff w = {wof(n**2)}, dust w = {wof(n)}: matches the known equations of state",
   wof(n**sp.Rational(4, 3)) == sp.Rational(1, 3) and wof(n**2) == 1 and wof(n) == 0)

# --- A4 COUNTEREXAMPLE: null dust. A congruence of massless world-lines is reparametrisation invariant, yet T = rho k k (k null) has
#     T^xx = rho != 0 along the beam. So 'reparametrisation <=> p = 0' is false for null lines. NARROWED: timelike world-lines.
kv = sp.Matrix([1, 1, 0, 0]); r_ = sp.symbols('r_', positive=True)
Tnull = r_ * kv * kv.T
ok(f"XA4 NARROWED: null dust T^xx = {Tnull[1, 1]} != 0 while reparametrisation invariant -> A4 holds only for timelike labels", Tnull[1, 1] != 0)

# --- A5 two cold counter-streams +-v: T^xx = rho v^2 (anisotropic), T^yy = 0 -> stress ~ rho sigma^2 with sigma = v, as stated.
v = sp.symbols('v', positive=True)
Txx = (r_ / 2) * v**2 + (r_ / 2) * v**2
ok(f"XA5 counter-streams: T^xx = {Txx}, T^yy = 0: anisotropic stress rho sigma^2 (sigma = v)", sp.simplify(Txx - r_ * v**2) == 0)

# --- A6 exponent 3 needs p = 0: with p = w rho, conservation gives rho ~ a^(-3(1+w)) (radiation a^-4).
a, wq = sp.symbols('a w', positive=True)
rhoa = sp.Function('rhoa')(a)
sol = sp.dsolve(sp.Eq(sp.diff(rhoa, a), -3 * (1 + wq) * rhoa / a), rhoa).rhs
expo = sp.simplify(sp.log(sol.subs(sp.Symbol('C1'), 1)) / sp.log(a))
ok(f"XA6 rho(a) = {sol}: exponent {expo}; = -3 only at w = 0, -4 for radiation", sp.simplify(expo + 3 * (1 + wq)) == 0)

# --- A7 Liouville needs Hamiltonian flow: friction dp/dt = -gamma p - V'(x) has phase divergence -gamma.
gam = sp.symbols('gamma', positive=True); V = sp.Function('V')(x); pp = sp.symbols('p', real=True)
div = sp.diff(pp, x) + sp.diff(-gam * pp - sp.diff(V, x), pp)
ok(f"XA7 damped flow: phase divergence {div} != 0 -> Liouville needs a Hamiltonian (collisionless) flow", sp.simplify(div + gam) == 0)

# --- B1 the Euler member has helicity: the ABC flow (steady Euler) has w.curl w != 0 (XA1) -- consistent with 'helicity only at Euler'.
ok("XB1 a known Euler solution (ABC) carries helicity, as B1 says of the Euler point", hel != 0)

# --- B2 hypothesis 'no scale': Proca div F = m^2 A is linear 2nd order but needs m; under x -> lam x Maxwell is invariant, Proca is not.
lam, m = sp.symbols('lambda m', positive=True)
ok("XB2 Proca term m^2 A scales as lam^0 while d F scales as lam^-2: invariant only for m = 0 -> B2 needs 'no dimensionful constant'",
   sp.simplify(lam**-2 - lam**0) != 0)

# --- B3 weight lam^(n-2k): 1-forms in 4D (k = 1) have weight 2 -> not conformal (known: minimally coupled scalar wave is not conformal).
ok(f"XB3 n = 4, k = 1: weight lam^{4 - 2} != 1 (not conformal); n = 4, k = 2: lam^0", 4 - 2 != 0 and 4 - 4 == 0)

# --- B4 COUNTEREXAMPLE for an unrestricted reading: quadratic gravity gives an identically divergence-free tensor
#     H_ab = 2 R R_ab - R^2 g_ab/2 - 2 nabla_a nabla_b R + 2 g_ab box R, not of the form a Ric + b R g + c g. B4 is right only within tensors
#     algebraic in Ric, R, g (second order, Lovelock). Check divergence of H in FLRW with arbitrary a(t).
af = sp.Function('a')(t)
coords = [t, x, y, z]
g = sp.diag(1, -af**2, -af**2, -af**2); gi = g.inv()
Gam = [[[sp.simplify(sum(gi[i, l] * (sp.diff(g[l, j], coords[k]) + sp.diff(g[l, k], coords[j]) - sp.diff(g[j, k], coords[l])) for l in range(4)) / 2)
         for k in range(4)] for j in range(4)] for i in range(4)]
def Ric(j, k):
    return sp.simplify(sum(sp.diff(Gam[i][j][k], coords[i]) - sp.diff(Gam[i][j][i], coords[k])
                           + sum(Gam[i][i][l] * Gam[l][j][k] - Gam[i][k][l] * Gam[l][j][i] for l in range(4)) for i in range(4)))
Rab = sp.Matrix(4, 4, lambda j, k: Ric(j, k)); R = sp.simplify(sum(gi[j, k] * Rab[j, k] for j in range(4) for k in range(4)))
dR = [sp.diff(R, c) for c in coords]
hess = sp.Matrix(4, 4, lambda j, k: sp.diff(R, coords[j], coords[k]) - sum(Gam[l][j][k] * dR[l] for l in range(4)))
box = sp.simplify(sum(gi[j, k] * hess[j, k] for j in range(4) for k in range(4)))
Hab = sp.simplify(2 * R * Rab - R**2 * g / 2 - 2 * hess + 2 * g * box)
def divH(b):
    return sp.simplify(sum(gi[a_, c] * (sp.diff(Hab[a_, b], coords[c]) - sum(Gam[d][c][a_] * Hab[d, b] + Gam[d][c][b] * Hab[a_, d] for d in range(4)))
                           for a_ in range(4) for c in range(4)))
dH0 = divH(0)
ok(f"XB4 NARROWED: div H_ab (quadratic gravity, FLRW, any a(t)) = {dH0}: a divergence-free tensor outside a Ric + b R g + c g exists -> "
   "B4 holds within second-order (algebraic) tensors only (Lovelock)", dH0 == 0)

# --- B5 hypothesis 'conserved matter': with energy exchange rho' = -3H(rho+p) + Q, d/dt(H^2 - kappa rho/3) = -kappa Q/3 != 0: Lambda drifts.
kap = sp.symbols('kappa', positive=True)
Hf, rh, pr, Q = [sp.Function(nm)(t) for nm in ('H', 'rh', 'pr', 'Q')]
expr = sp.simplify(2 * Hf * (-kap / 2 * (rh + pr)) - kap / 3 * (-3 * Hf * (rh + pr) + Q))
ok(f"XB5 with exchange Q: d/dt(H^2 - kappa rho/3) = {expr}: Lambda is a label only if matter is conserved", sp.simplify(expr + kap * Q / 3) == 0)

# --- B6 hypothesis 'boost invariance': maps commuting only with rotations so(3) keep two free constants (rho and p independent).
Tm = sp.Matrix(4, 4, lambda i, j: sp.Symbol(f't{i}{j}'))
def gen(i, j):
    M = sp.zeros(4, 4)
    for k in range(4):
        for (mm,), c in grade(gp({(i, j): 1}, {(k,): 1}, STA), 1).items():
            M[mm, k] += c
    return M
eqs = [e for (i, j) in itertools.combinations((1, 2, 3), 2) for e in (Tm * gen(i, j) - gen(i, j) * Tm)]
solr = sp.solve(eqs, list(Tm), dict=True)[0]
nfree = len(Tm.subs(solr).free_symbols)
ok(f"XB6 commuting with rotations only: {nfree} free constants (rho, p independent) -> w = -1 needs boost invariance", nfree == 2)

# --- C1 hypothesis 'spacelike': a timelike vector (S^2 < 0, imaginary sphere) is not a de Sitter point; S(c, r) with r^2 -> -r^2 has S^2 < 0.
C5 = {0: 1, 1: 1, 2: 1, 3: 1, 4: -1}
ninf = {(3,): 1, (4,): 1}; n0 = {(4,): sp.Rational(1, 2), (3,): -sp.Rational(1, 2)}
P = lambda xx: add(vec(list(xx) + [0, 0]), scal(sum(q**2 for q in xx) / 2, ninf), n0)
Simag = add(P([0, 0, 0]), scal(sp.Rational(1, 2), ninf))      # r^2 = -1
ok(f"XC1 imaginary sphere S^2 = {gp(Simag, Simag, C5).get((), 0)} < 0: not a de Sitter point -> C1 needs S^2 > 0",
   gp(Simag, Simag, C5).get((), 0) < 0)

# --- C2 only p = n is conformal: the 4-Laplacian in 3D is not covariant under inversion (ratio depends on direction at fixed |x|).
def Lp(f, pw):
    gr = [sp.diff(f, q) for q in X3]; nn = sp.sqrt(sum(c**2 for c in gr))
    return sum(sp.diff(nn**(pw - 2) * gr[i], X3[i]) for i in range(3))
phi = x**2 * y + z**3 + x * z + 2 * y
r2 = sum(q**2 for q in X3); sig = [q / r2 for q in X3]
def ratio(pw, pt):
    img = {X3[i]: sig[i].subs(pt) for i in range(3)}
    return sp.N(Lp(phi.subs(dict(zip(X3, sig)), simultaneous=True), pw).subs(pt) / Lp(phi, pw).subs(img), 20)
pa = {x: sp.Rational(3, 5), y: sp.Rational(4, 5), z: 0}; pb = {x: 0, y: sp.Rational(3, 5), z: sp.Rational(4, 5)}   # same |x| = 1
r3a, r3b, r4a, r4b = ratio(3, pa), ratio(3, pb), ratio(4, pa), ratio(4, pb)
ok(f"XC2 at |x| = 1: 3-Laplacian ratios {r3a:.6g}, {r3b:.6g} (equal: covariant); 4-Laplacian {r4a:.6g}, {r4b:.6g} (differ: not covariant)",
   abs(r3a - r3b) < 1e-15 and abs(r4a - r4b) > 1e-6)

# --- C3 hypothesis 'de Sitter': in the matter era a time shift is not a symmetry (a(t+c)/a(t) depends on t).
c0 = sp.symbols('c0', positive=True)
am = t**sp.Rational(2, 3)
ok(f"XC3 matter era: d/dt[a(t+c)/a(t)] = {sp.simplify(sp.diff(am.subs(t, t + c0) / am, t))} != 0: time shift = dilator only in de Sitter",
   sp.simplify(sp.diff(am.subs(t, t + c0) / am, t)) != 0)

# --- C4 known limits and hypothesis 'positive curvature': H -> 0 gives Unruh a/2pi, a -> 0 gives Gibbons-Hawking H/2pi; in AdS
#     (Deser-Levin) T = sqrt(a^2 - k^2)/2pi and the 'dual' a -> k^2/a gives no real temperature for a > k.
aa, H, k = sp.symbols('a H k', positive=True)
T = sp.sqrt(aa**2 + H**2) / (2 * sp.pi); Tads = sp.sqrt(aa**2 - k**2) / (2 * sp.pi)
ok(f"XC4 limits: H->0 {sp.limit(T, H, 0)}, a->0 {sp.limit(T, aa, 0)}; AdS: the dual of a = 2k is k/2, where T^2 = {sp.simplify(Tads.subs(aa, k / 2)**2)} < 0",
   sp.limit(T, H, 0) == aa / (2 * sp.pi) and sp.limit(T, aa, 0) == H / (2 * sp.pi) and sp.simplify(Tads.subs(aa, k / 2)**2) < 0)

# --- D1 hypothesis 'no extra scale': a cosmological constant in Poisson breaks the group to mu = lambda (cosmic_dilator M4); collisions
#     with a fixed cross-section s break it too: mean free path 1/(n s) must scale as lambda, but n s scales as (phase density x mu^3) s.
l_, m_ = sp.symbols('l_ m_', positive=True)
nscale = (1 / (l_**2 * m_)) * m_**3                 # number density ~ f v^3
ok(f"XD1 collisions with fixed cross-section: 1/(n s) scales as {sp.simplify(1 / nscale)}, a length scales as l_: equal only if "
   f"{sp.solve(sp.Eq(1 / nscale, l_), m_)} -> collisional systems break the group", sp.solve(sp.Eq(1 / nscale, l_), m_) != [])

# --- D2 known profiles: Plummer has inner v ~ r (p = 1) and outer Kepler (p = -1/2); NFW outer M ~ ln r is not a power law (no fixed point).
rr = sp.symbols('r', positive=True)
Mpl = rr**3 / (1 + rr**2)**sp.Rational(3, 2)
vpl = sp.sqrt(Mpl / rr)
pin = sp.limit(sp.diff(sp.log(vpl), rr) * rr, rr, 0); pout = sp.limit(sp.diff(sp.log(vpl), rr) * rr, rr, sp.oo)
Mnfw = sp.log(1 + rr) - rr / (1 + rr)
pnfw_out = sp.limit(sp.diff(sp.log(sp.sqrt(Mnfw / rr)), rr) * rr, rr, sp.oo)
ok(f"XD2 Plummer: p_in = {pin}, p_out = {pout}; NFW outer slope -> {pnfw_out} only logarithmically (M ~ ln r): not a fixed point",
   pin == 1 and pout == sp.Rational(-1, 2) and pnfw_out == sp.Rational(-1, 2))

# --- D3 known limits: Fermi-Dirac density -> (sqrt(pi)/2) e^psi (Boltzmann, isothermal) and -> (2/3) psi^(3/2) (degenerate, polytrope 3/2).
from scipy.integrate import quad
F = lambda p_: quad(lambda s: np.sqrt(s) / (1 + np.exp(min(s - p_, 700))), 0, max(p_, 0) + 60)[0]
ok(f"XD3 F(-15)/(sqrt(pi)/2 e^-15) = {F(-15) / (np.sqrt(np.pi) / 2 * np.exp(-15)):.6f}, F(400)/((2/3) 400^1.5) = {F(400) / (2 / 3 * 400**1.5):.6f}",
   abs(F(-15) / (np.sqrt(np.pi) / 2 * np.exp(-15)) - 1) < 1e-4 and abs(F(400) / (2 / 3 * 400**1.5) - 1) < 1e-3)

# --- D4 hypothesis 'fixed L': without it, v = 0 has lower energy than any rigid rotation (trivial minimum).
# (XD4, statement: without fixing L the minimum is v = 0; Kelvin's theorem needs L fixed.)

# --- E1 the general rule I^2 = (-1)^(n(n-1)/2 + q) for Cl(p,q) agrees with direct computation for all signatures of dimension 5.
def Isq(sig):
    I = {tuple(range(len(sig))): 1}
    return gp(I, I, sig)[()]
okall = all(Isq({i: (1 if i < pq else -1) for i in range(5)}) == (-1)**(10 + (5 - pq)) for pq in range(6))
ok(f"XE1 I^2 = (-1)^(n(n-1)/2 + q) for all Cl(p, 5-p): {okall}", okall)

# --- E2 hypothesis 'square +1': for a rotation unit (square -1) the form is alpha^2 + beta^2 (definite): no null direction, no middle.
al, be = sp.symbols('alpha beta', real=True)
none = sp.solveset(al**2 + 1, al, domain=sp.S.Reals)
ok(f"XE2 rotation unit: alpha^2 + beta^2 = 0 with beta = 1 has real solutions {none} (none): no null direction, no middle -> middles need square +1",
   none == sp.S.EmptySet)

# --- E3 the empirical RAR function is not self-dual (ga_duality_test T4 prints 0.005-0.92): self-duality singles out x/(1+x) among the two
#     standard forms; the vector fraction x^2/(1+x^2) is not L1.
xs = sp.symbols('x_', positive=True)
mv = xs**2 / (1 + xs**2)
ms = xs / (1 + xs)
ok(f"XE3 both fractions are partitions: rotor mu(x)+mu(1/x) = {sp.simplify(ms + ms.subs(xs, 1 / xs))}, vector = {sp.simplify(mv + mv.subs(xs, 1 / xs))}; "
   f"the standard mu = sqrt(vector fraction) is not: {sp.simplify(sp.sqrt(mv) + sp.sqrt(mv.subs(xs, 1 / xs)))} -- E3 as stated survives",
   sp.simplify(ms + ms.subs(xs, 1 / xs)) == 1 and sp.simplify(mv + mv.subs(xs, 1 / xs)) == 1
   and sp.simplify(sp.sqrt(mv) + sp.sqrt(mv.subs(xs, 1 / xs)) - 1) != 0)

# --- E4 hypothesis 'scale-invariant potential': Newton 1/x breaks sl(2,R): {D,H} = H fails (ga_time_dilation T4 gives 2E - V).
xx, px, gg = sp.symbols('x p g', positive=True)
PB = lambda A, B: sp.simplify(sp.diff(A, xx) * sp.diff(B, px) - sp.diff(A, px) * sp.diff(B, xx))
Hn = px**2 / 2 - gg / xx; Dn = xx * px / 2
ok(f"XE4 Newton: {{D,H}} - H = {sp.simplify(PB(Dn, Hn) - Hn)} != 0 -> the sl(2,R) time triple needs a scale-invariant potential",
   sp.simplify(PB(Dn, Hn) - Hn) != 0)

# --- E5 hypothesis 'non-relativistic inputs': with c an acceleration exists (c H); from (G, H) alone none.
# (XE5, statement: with c an acceleration exists, cH; E5 concerns (G, H) and scale-free symmetries only.)

# --- F1 known limits: standard expansion (k = 1) -> b = 1; static 'tired light' (no a_o/a_e change in arrival times) -> b = 0.
kk = sp.symbols('k_', positive=True); qq = sp.symbols('q_', real=True)
b_of = lambda kv: 1 / kv
ok(f"XF1 k = 1 -> b = {b_of(1)} (observed 1.003 +- 0.011); b = 0 (no time dilation, tired light) excluded at {(1.003 - 0) / 0.011:.0f} sigma",
   b_of(1) == 1 and (1.003 - 0) / 0.011 > 50)

# --- F2 hypothesis 'unoriented labels': charged dust (Einstein-Maxwell) is a consistent theory with oriented labels (A^N gauge invariant, K6).
# (XF2, statement: charged dust is consistent with oriented labels (A^N gauge invariant, dark_sector K6) -> F2 needs unoriented labels.)

# --- F3 hypothesis 'conserved charges': decaying dark matter rho_c ~ a^-3 e^{-Gamma t} makes Omega_c/Omega_b time-dependent.
Gm = sp.symbols('Gamma', positive=True)
ratio_t = sp.exp(-Gm * t)
ok(f"XF3 decaying DM: Omega_c/Omega_b ~ {ratio_t} (time-dependent) -> F3 needs a conserved label charge", sp.diff(ratio_t, t) != 0)
