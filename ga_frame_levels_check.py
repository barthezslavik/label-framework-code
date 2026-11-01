# Two levels of one object, the frame map h(a): spinor level (rotation + dilation) vs general level (adds shear).
# Checks against GR and CDM: papers/restart/23-frame-levels.md, F1-F8.
# Run: uv run --with sympy python ga_frame_levels_check.py
import sympy as sp
from ga_core import *

LOR = {0: 1, 1: -1, 2: -1, 3: -1}
I4 = {(0, 1, 2, 3): 1}
rev = lambda X: clean({k: v * (-1) ** (len(k) * (len(k) - 1) // 2) for k, v in X.items()})
X = sp.symbols('t x y z')
eta = sp.diag(1, -1, -1, -1)

# F1 spinor frame: psi = sqrt(rho) e^{I beta/2} R. For every vector a: psi a psi~ = rho R a R~ (the duality angle beta drops out on vectors)
rho, beta, th, ph = sp.symbols('rho beta theta phi', real=True)
R = gp(add(scal(sp.cos(th / 2), {(): 1}), scal(sp.sin(th / 2), {(1, 2): 1})), add(scal(sp.cosh(ph / 2), {(): 1}), scal(sp.sinh(ph / 2), {(0, 1): 1})), LOR)
eb = add(scal(sp.cos(beta / 2), {(): 1}), scal(sp.sin(beta / 2), I4))
psi = scal(sp.sqrt(rho), gp(eb, R, LOR))
av = vec(list(sp.symbols('a0:4')))
lhs = gp(gp(psi, av, LOR), rev(psi), LOR)
rhs = scal(rho, gp(gp(R, av, LOR), rev(R), LOR))
ok("F1 psi a psi~ = rho R a R~ for psi = sqrt(rho) e^(I beta/2) R: beta drops out, the spinor frame is rotation x dilation",
   all(sp.simplify(v) == 0 for v in add(lhs, scal(-1, rhs)).values()))
# F2 the frame e_a = psi gamma_a psi~ has e_a . e_b = rho^2 eta_ab: the metric it defines is conformally flat (only the scale rho survives)
ea = [gp(gp(psi, {(m,): 1}, LOR), rev(psi), LOR) for m in range(4)]
G = sp.Matrix(4, 4, lambda i, j: sp.simplify(gp(ea[i], ea[j], LOR).get((), 0) / 2 + gp(ea[j], ea[i], LOR).get((), 0) / 2))
ok(f"F2 e_a . e_b = rho^2 eta_ab: {sp.simplify(G - rho**2 * sp.diag(*[LOR[m] for m in range(4)])) == sp.zeros(4)}",
   sp.simplify(G - rho**2 * sp.diag(*[LOR[m] for m in range(4)])) == sp.zeros(4))
# F3 degree count of a general frame map (polar decomposition h = R S, S symmetric): 16 = 6 (rotation) + 1 (dilation) + 9 (traceless symmetric = shear)
ok("F3 16 = 6 (Lorentz rotation, rotor R) + 1 (dilation, rho) + 9 (traceless symmetric shear): spinor level = 7, general level adds the 9 shear components",
   6 + 1 + 9 == 16 and sp.binomial(4, 2) == 6 and 4 * 5 // 2 - 1 == 9)

# full curvature of a metric
def curvature(g):
    gi = g.inv()
    Gm = [[[sp.simplify(sum(gi[r, s] * (sp.diff(g[s, m], X[k]) + sp.diff(g[s, k], X[m]) - sp.diff(g[m, k], X[s])) for s in range(4)) / 2)
            for k in range(4)] for m in range(4)] for r in range(4)]
    def Riem(r, s, m, k):
        return sp.diff(Gm[r][k][s], X[m]) - sp.diff(Gm[r][m][s], X[k]) + sum(Gm[r][m][l] * Gm[l][k][s] - Gm[r][k][l] * Gm[l][m][s] for l in range(4))
    Ric = sp.Matrix(4, 4, lambda s, k: sp.simplify(sum(Riem(r, s, r, k) for r in range(4))))
    Rs = sp.simplify(sum(gi[i, j] * Ric[i, j] for i in range(4) for j in range(4)))
    Ein = sp.simplify(Ric - Rs * g / 2)
    def Weyl(a_, b_, c_, d_):
        Rl = sum(g[a_, r] * Riem(r, b_, c_, d_) for r in range(4))
        return sp.simplify(Rl - (g[a_, c_] * Ric[b_, d_] - g[a_, d_] * Ric[b_, c_] - g[b_, c_] * Ric[a_, d_] + g[b_, d_] * Ric[a_, c_]) / 2
                           + Rs * (g[a_, c_] * g[b_, d_] - g[a_, d_] * g[b_, c_]) / 6)
    return Ric, Rs, Ein, Weyl
prs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
allweyl = lambda W: [W(a_, b_, c_, d_) for i_, (a_, b_) in enumerate(prs) for (c_, d_) in prs[i_:]]

# F4 spinor level in GR: g = h(x)^2 eta (arbitrary h): Weyl = 0 identically, R = -6 box(h)/h^3 -> only the Ricci ("matter") part of curvature
h = sp.Function('h')(*X)
Ric4, R4, Ein4, W4 = curvature(h**2 * eta)
boxh = sum(eta[m, m] * sp.diff(h, X[m], 2) for m in range(4))
ok("F4 g = h^2 eta: all Weyl components vanish, R = -6 box(h)/h^3: the spinor (dilation) level carries only Ricci curvature",
   all(w_ == 0 for w_ in allweyl(W4)) and sp.simplify(R4 + 6 * boxh / h**3) == 0)

# F5 CDM background: FLRW in conformal time, g = a(t)^2 eta, is the spinor level with h = a. Einstein tensor = perfect fluid with
#    rho = 3 a'^2/a^4, p = (2 a a'' - a'^2)/a^4 ... (units 8 pi G = 1); p = 0 (dust, CDM) <=> a ~ t^2, and then rho ~ a^-3
a = sp.Function('a')(X[0])
_, _, EinF, WF = curvature(a**2 * eta)
mixed = (a**2 * eta).inv() * EinF                                          # G^mu_nu
rhoF, pF = sp.simplify(mixed[0, 0]), sp.simplify(-mixed[1, 1])
offdiag = all(sp.simplify(mixed[i, j]) == 0 for i in range(4) for j in range(4) if i != j)
iso = sp.simplify(mixed[1, 1] - mixed[2, 2]) == 0 and sp.simplify(mixed[2, 2] - mixed[3, 3]) == 0
tau = X[0]; C = sp.symbols('C', positive=True)
dust = {a: C * tau**2}
p_dust = sp.simplify(pF.subs(a, C * tau**2).doit())
rho_dust = sp.simplify(rhoF.subs(a, C * tau**2).doit())
ok(f"F5 FLRW = a(t)^2 eta (spinor level): G is a perfect fluid (diagonal {offdiag}, isotropic {iso}), rho = {rhoF}, p = {pF}; "
   f"a = C t^2 gives p = {p_dust}, rho = {rho_dust} ~ a^-3 (CDM matter era)",
   offdiag and iso and p_dust == 0 and sp.simplify(sp.diff(rho_dust * (C * tau**2)**3, tau)) == 0 and rho_dust != 0)

# F6 a static dilation-only inhomogeneity: g = a(t)^2 (1 + 2 eps f(x,y,z)) eta. To first order the spatial traceless stress
#    G_ij - delta_ij G_kk/3 is proportional to d_i d_j f - delta_ij lap f /3, nonzero for any f that is not linear+radial-quadratic.
#    Dust has no anisotropic stress, so this ansatz is inconsistent with dust; whether clustering dust in general needs the shear level
#    is not decided here.
eps = sp.symbols('epsilon')
f = sp.Function('f')(*X[1:])
gp_ = a**2 * (1 + 2 * eps * f) * eta
_, _, EinP, _ = curvature(gp_)
dE = EinP.applyfunc(lambda e: sp.simplify(sp.diff(e, eps).subs(eps, 0)))
trS = sum(dE[i, i] for i in (1, 2, 3))
aniso = sp.Matrix(3, 3, lambda i, j: sp.simplify(dE[i + 1, j + 1] - (trS / 3 if i == j else 0)))
target = sp.Matrix(3, 3, lambda i, j: sp.diff(f, X[i + 1], X[j + 1]) - (sum(sp.diff(f, X[k], 2) for k in (1, 2, 3)) / 3 if i == j else 0))
ratio = sp.simplify(aniso[0, 1] / target[0, 1])
ok(f"F6 dilation-only inhomogeneity: anisotropic stress = {ratio} * (d_i d_j f - delta lap f/3) != 0 -> this static dilation-only ansatz is inconsistent with dust (no anisotropic stress); general clustering is not addressed",
   sp.simplify(aniso - ratio * target) == sp.zeros(3) and ratio != 0)

# F7 the shear level in GR: weak field around a mass (Newtonian gauge, Phi = Psi, as for CDM with no anisotropic stress):
#    g = eta + 2 Phi diag(1, 1, 1, 1). Its trace-free part relative to eta is Phi diag(3, 1, 1, 1) != 0 (shear), and the linear Weyl tensor
#    is the tidal tensor d_i d_j Phi - delta lap Phi/3, nonzero for any non-trivial Phi; Ricci is the density (lap Phi).
Phi = sp.Function('Phi')(*X[1:])
dg = 2 * Phi * sp.eye(4)
tr = sum(eta[m, m] * dg[m, m] for m in range(4))
shear = dg - tr / 4 * eta
ok(f"F7a Newtonian-gauge perturbation: trace part {sp.simplify(-tr/4)} * eta, shear part diag{[sp.simplify(shear[m, m]/Phi) for m in range(4)]} * Phi != 0",
   sp.simplify(shear[1, 1] - Phi) == 0 and sp.simplify(shear[0, 0] - 3 * Phi) == 0)
def lin_riem(hm):   # R_{mu nu rho sigma} = 1/2 (d_nu d_rho h_mu sigma + d_mu d_sigma h_nu rho - d_mu d_rho h_nu sigma - d_nu d_sigma h_mu rho)
    d2 = lambda e, i, j: sp.diff(e, X[i], X[j])
    return lambda m, n, r, s: (d2(hm[m, s], n, r) + d2(hm[n, r], m, s) - d2(hm[n, s], m, r) - d2(hm[m, r], n, s)) / 2
def lin_weyl(hm):
    Rl = lin_riem(hm)
    Ric = sp.Matrix(4, 4, lambda n, s: sum(eta[m, m] * Rl(m, n, m, s) for m in range(4)))
    Rs = sum(eta[m, m] * Ric[m, m] for m in range(4))
    return lambda a_, b_, c_, d_: sp.simplify(Rl(a_, b_, c_, d_) - (eta[a_, c_] * Ric[b_, d_] - eta[a_, d_] * Ric[b_, c_] - eta[b_, c_] * Ric[a_, d_] + eta[b_, d_] * Ric[a_, c_]) / 2
                                              + Rs * (eta[a_, c_] * eta[b_, d_] - eta[a_, d_] * eta[b_, c_]) / 6), Ric
Wn, Ricn = lin_weyl(dg)
C0101 = Wn(0, 1, 0, 1)
tidal = sp.diff(Phi, X[1], 2) - sum(sp.diff(Phi, X[k], 2) for k in (1, 2, 3)) / 3
ok(f"F7b linear Weyl C_0101 = {sp.factor(sp.simplify(C0101 / tidal))} * (d_x d_x Phi - lap Phi/3): the tidal field = shear level; Ricci_00 = {sp.simplify(Ricn[0, 0])}",
   sp.simplify(C0101 / tidal).free_symbols <= set() and C0101 != 0)
# F8 a pure-trace (dilation) perturbation 2 phi eta has zero linear Weyl for any phi: Weyl sees only the shear part
phf = sp.Function('varphi')(*X)
Wc, _ = lin_weyl(2 * phf * eta)
ok("F8 dilation perturbation 2 phi eta: linear Weyl = 0 for any phi(t,x,y,z) -> Weyl (free gravity, tides, waves) lives only in the shear part",
   all(w_ == 0 for w_ in allweyl(Wc)))
