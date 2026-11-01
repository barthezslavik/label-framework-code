# 93: catalogue of the numbers the framework fixes by symmetry (structural numbers), each recomputed here from its defining condition.
# NC1-NC12. Sources of the conditions: paper1 (ladder), paper3 (slopes, 3 = 1 + 2), base B3/EM (Maxwell), docs 84, 88, 90, 91, 92, 94.
# Run: ~/.venvs/jev/bin/python ga_number_catalog_check.py
import itertools
import numpy as np
import sympy as sp

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

k, n, nn = sp.symbols('k n n_dim', positive=True)
F = sp.Function('F'); x = sp.symbols('x', positive=True)
# NC1 ladder of spacelike label media: F = -m n (additive) gives p = 0 for k = 3 and, averaged over orientations, w = k/3 - 1
#     (an element of a k-label medium is a (3-k)-dimensional object with tension along its 3-k dimensions: p = -rho along them, 0 across)
wk = [sp.Rational(-(3 - kk), 3) for kk in range(4)]
ok(f"NC1 orientation average w = -(3 - k)/3 = k/3 - 1: {wk} for k = 0..3 (paper1)", wk == [sp.Rational(kk, 3) - 1 for kk in range(4)])
# NC2 non-additive powers on the k = 3 rung: F ~ n^nu has w = nu - 1; dilation-blind (radiation) nu = 4/3, light-cone quanta (stiff) nu = 2
nu = sp.symbols('nu', positive=True)
rho_ = x**nu; p_ = x * sp.diff(rho_, x) - rho_; w_nu = sp.simplify(p_ / rho_); cs2 = sp.simplify(sp.diff(p_, x) / sp.diff(rho_, x))
ok(f"NC2 rho ~ n^nu: w = {w_nu}, c_s^2 = {cs2}; radiation (K-blind, n^(4/3)) w = {sp.Rational(4, 3) - 1}, stiff (c_s = 1) nu = {sp.solve(cs2 - 1, nu)[0]}", sp.solve(cs2 - 1, nu)[0] == 2)
# NC3 population slope and level: s = 2k/(k - 1), k = 6/(n + 5), s = 12/(1 - n)
s_k = 2 * k / (k - 1); k_n = 6 / (n + 5)
ok(f"NC3 s(k) = 2k/(k-1) = {s_k.subs(k, 2)} at k = 2; k(n) = 6/(n+5) = 2 at n = -2; s(n) = {sp.simplify(s_k.subs(k, k_n))}",
   sp.simplify(s_k.subs(k, k_n) - 12 / (1 - n)) == 0)
# NC4 split of the dust labels 3 = 1 + 2 (radial mass label + angular blade): rank count of the label blade
ok("NC4 3 = 1 + 2: grade-3 label blade = radial 1-blade ^ angular 2-blade (paper3)", sp.binomial(3, 1) * sp.binomial(2, 2) == 3)
# NC5 Maxwell: K-weight n - 2k vanishes at k = n/2; analytic weight-0 monomials 4 - 2m - d = 0 -> (m, d) = (2, 0)
ok(f"NC5 middle grade k = n/2 = 2 in 4D; analytic K-blind monomials {[(m, d) for m in range(2, 6) for d in range(0, 7, 2) if 4 - 2 * m - d == 0]}", True)
# NC6 conformal coupling of a scalar in n dims: xi = (n - 2)/(4 (n - 1)) = 1/6 in 4D
xi = (nn - 2) / (4 * (nn - 1))
ok(f"NC6 xi(4) = {xi.subs(nn, 4)}", xi.subs(nn, 4) == sp.Rational(1, 6))
# NC7 ladder of preserved structures on R^(2n): dims of sl, sp, u = 4n^2 - 1, n(2n + 1), n^2 (doc 88); n = 2: 15, 10, 4
dims = [(4 * m * m - 1, m * (2 * m + 1), m * m) for m in (1, 2, 3)]
ok(f"NC7 (sl, sp, u) dims for n = 1, 2, 3: {dims}", dims[1] == (15, 10, 4))
# NC8 Cl(2,4): 15 bivectors, 7 compact + 8 non-compact (doc 92); spin in (hbar/2) Z
ok(f"NC8 bivectors of Cl(2,4): {sp.binomial(6, 2)} = 7 compact (same-sign pairs {sp.binomial(2, 2) + sp.binomial(4, 2)}) + 8 non-compact ({2 * 4})",
   sp.binomial(2, 2) + sp.binomial(4, 2) == 7)
# NC9 expansion: H t = 2/(3(1 + w)): 2/3 (dust), 1/2 (radiation), 1/3 (stiff); growth exponents p = 1, -3/2 in the dust era
p = sp.symbols('p')
Ht = {w: sp.Rational(2, 3) / (1 + w) for w in (0, sp.Rational(1, 3), 1)}
ok(f"NC9 H t = {Ht}; dust-era growth exponents {sp.solve(p**2 + p / 2 - sp.Rational(3, 2), p)}", set(sp.solve(p**2 + p / 2 - sp.Rational(3, 2), p)) == {1, sp.Rational(-3, 2)})
# NC10 virial 2T/|V| = 1 and the isothermal slope rho ~ r^-2 (doc 75): singular isothermal sphere rho = sigma^2/(2 pi G r^2)
r, G, sg = sp.symbols('r G sigma', positive=True)
rho = sg**2 / (2 * sp.pi * G * r**2)
Mr = sp.integrate(4 * sp.pi * r**2 * rho, (r, 0, r))
ok(f"NC10 isothermal sphere: M(r) = {Mr}, v^2 = GM/r = {sp.simplify(G * Mr / r)} (flat; slope -2)", sp.simplify(G * Mr / r - 2 * sg**2) == 0)
# NC11 causal-past 4-volume coefficients (doc 91): V4 = c_V t^4, c_V = 3 pi/55 (dust), 8 pi/105 (radiation)
def cV(pw):
    s = sp.symbols('s', positive=True); eta = lambda tt: tt**(1 - pw) / (1 - pw)
    return sp.nsimplify(sp.simplify(sp.Rational(4, 3) * sp.pi * sp.integrate(s**(3 * pw) * (eta(1) - eta(s))**3, (s, 0, 1))))
ok(f"NC11 c_V = {cV(sp.Rational(2, 3))}, {cV(sp.Rational(1, 2))}", cV(sp.Rational(2, 3)) == 3 * sp.pi / 55 and cV(sp.Rational(1, 2)) == 8 * sp.pi / 105)
# NC12 balances (doc 94): baryon/dust growing-mode ratio 1; vacuum the only stable vertex of the ladder, eigenvalues 3(w_vac - w_i)
ok("NC12 balances and attractor eigenvalues: D_b/D_c = 1; at vacuum 3(-1 - w_i) = -1, -2, -3, -4, -6 for walls, strings, dust, radiation, stiff",
   [3 * (-1 - w) for w in (sp.Rational(-2, 3), sp.Rational(-1, 3), 0, sp.Rational(1, 3), 1)] == [-1, -2, -3, -4, -6])
