# 143: the four-label vacuum and q-theory (Klinkhamer-Volovik 2008, 0711.3170; 4D-brane realization 1604.06060; two-measure theory of
# Guendelman-Kaganovich, gr-qc/9905029). Equations borrowed, every step recomputed here.
# VQ1 the label 4-form is exact with a composite potential: det(d phi) = d_mu(phi^1 C^mu_1) (C = cofactor matrix), and the Piola identity
#     d_mu C^mu_a = 0;  VQ2 stress of sqrt(-g) F(n), n = |det d phi|/sqrt(-g), by numerical metric variation: T = (n F' - F) g, equal to the
#     q-theory form (eps - q eps') g with eps = -F, q = n;  VQ3 field equation: d_mu(F' C^mu_a) = C^mu_a d_mu F' (no metric), det C = (det J)^3,
#     so F'(n) = mu is a constant of integration where the gradients are non-degenerate;  VQ4 no local excitations: the quadratic part of the
#     Jacobian is a total divergence, the quadratic action is F''(n0)/2 (div xi)^2 up to a total derivative, the kinetic matrix k_a k_b has
#     rank 1, so every non-constant solution of the linear equation has div xi = 0 (a relabelling, delta n = 0); relabellings with unit
#     Jacobian leave the stress unchanged and a perturbation that changes n violates the field equation (de Sitter background);
#     VQ5 status of Lambda: rho_V(mu) takes every value on the solution family; the q-theory equilibrium (zero external pressure,
#     rho_V = 0) selects one mu; with eps = 1/5 + sin^2(q)/2 it reproduces q0 = 3.20479 and q0^2 eps''(q0) = 10.1887 of 0711.3170;
#     F proportional to n gives T = 0 (the linear eps discarded in 0711.3170);  VQ6 periodic labels on a 4-torus: the total label count
#     int det J d^4x = (2 pi)^4 deg is quantised by the degree of the map;  VQ7 five labels in 4D: the 5-form vanishes (rank <= 4); five
#     labels with X.X = -l^2 in R^{1,4} are four labels on a curved label space; the induced metric of the hyperboloid is de Sitter with
#     R = +-12/l^2 (sign by curvature convention) in the hyperboloid's own geometry.  Run: ~/.venvs/jev/bin/python ga_vacuum_q_check.py
import itertools
import numpy as np
import sympy as sp
from scipy.optimize import brentq

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

rng = np.random.default_rng(143)
X = sp.symbols('x0:4')

# ---------- VQ1 exactness of the label 4-form and the Piola identity ----------
def rand_poly(seed):
    r = np.random.default_rng(seed)
    terms = [sp.Integer(int(r.integers(-3, 4))) * X[i] * X[j] ** int(r.integers(0, 2))
             for i in range(4) for j in range(4)]
    return sum(terms) + X[seed % 4]
phi = [rand_poly(10 + a) for a in range(4)]
J = sp.Matrix(4, 4, lambda a, m: sp.diff(phi[a], X[m]))            # J[a, mu] = d_mu phi^a
detJ = sp.expand(J.det())
C = J.adjugate().T                                                  # C[a, mu] = cofactor, sum_mu C[a,mu] J[b,mu] = det J delta_ab
piola = [sp.expand(sum(sp.diff(C[a, m], X[m]) for m in range(4))) for a in range(4)]
exact = sp.expand(sum(sp.diff(phi[0] * C[0, m], X[m]) for m in range(4)) - detJ)
cof_ok = sp.simplify(C * J.T - detJ * sp.eye(4)) == sp.zeros(4)
ok(f"VQ1 label 4-form exact: det(d phi) - d_mu(phi^1 C^mu_1) = {exact}; Piola d_mu C^mu_a = {piola}; cofactor identity {cof_ok} "
   f"(random polynomial labels): the vacuum 4-form is F = dA with the composite potential A = phi^1 dphi^2^dphi^3^dphi^4",
   exact == 0 and all(p == 0 for p in piola) and cof_ok)

# ---------- VQ2 stress tensor by numerical metric variation ----------
def lorentz_metric(r):
    L = np.eye(4) + 0.3 * r.standard_normal((4, 4))
    return L.T @ np.diag([1., -1., -1., -1.]) @ L                   # signature (+,-,-,-)
Fs = {'chaplygin': (lambda n: -np.sqrt(1.3 + 0.7 * n**2), lambda n: -0.7 * n / np.sqrt(1.3 + 0.7 * n**2)),
      'cubic':     (lambda n: 0.4 * n**3 - n,                  lambda n: 1.2 * n**2 - 1.0),
      'log':       (lambda n: np.log(n) + 0.2 * n**2,           lambda n: 1.0 / n + 0.4 * n)}
def lag(ginv, Jm, F):
    g = np.linalg.inv(ginv); sg = np.sqrt(-np.linalg.det(g)); n = abs(np.linalg.det(Jm)) / sg
    return sg * F(n), n, sg
worst = 0.0; worst_q = 0.0
for trial in range(6):
    g = lorentz_metric(rng); ginv = np.linalg.inv(g); Jm = np.eye(4) + 0.4 * rng.standard_normal((4, 4))
    for name, (F, Fp) in Fs.items():
        _, n, sg = lag(ginv, Jm, F)
        T = np.zeros((4, 4)); h = 1e-6
        for m in range(4):
            for v in range(m, 4):
                E = np.zeros((4, 4)); E[m, v] = E[v, m] = 1.0
                d = (lag(ginv + h * E, Jm, F)[0] - lag(ginv - h * E, Jm, F)[0]) / (2 * h)
                fac = 1.0 if m == v else 0.5                         # symmetric variation counts the pair twice
                T[m, v] = T[v, m] = 2.0 / sg * d * fac
        target = (n * Fp(n) - F(n)) * g
        worst = max(worst, np.abs(T - target).max() / np.abs(target).max())
        eps = lambda q: -F(q); epsp = lambda q: -Fp(q)               # q-theory: S = int sqrt(-g)(R/16piG - eps), eps = -F, q = n
        worst_q = max(worst_q, abs((eps(n) - n * epsp(n)) - (n * Fp(n) - F(n))))
ok(f"VQ2 stress of sqrt(-g) F(n) by numerical metric variation (6 random Lorentzian metrics x 3 actions): T = (n F' - F) g, max rel. dev "
   f"{worst:.1e}; equals the q-theory vacuum energy eps - q eps' with eps = -F, q = n (max dev {worst_q:.1e})", worst < 1e-6 and worst_q < 1e-12)

# ---------- VQ3 field equation without the metric ----------
fx = sp.Function('f')(*X)                                            # f stands for F'(n) at each point; the identity holds for any f
lhs = [sp.expand(sum(sp.diff(fx * C[a, m], X[m]) for m in range(4))) for a in range(4)]
rhs = [sp.expand(sum(C[a, m] * sp.diff(fx, X[m]) for m in range(4))) for a in range(4)]
same = all(sp.expand(l - r) == 0 for l, r in zip(lhs, rhs))
Jn = sp.Matrix(4, 4, sp.symbols('j0:16'))
detC = sp.simplify(Jn.adjugate().det() - Jn.det() ** 3) == 0
ok(f"VQ3 Euler-Lagrange d_mu(F'(n) C^mu_a) = C^mu_a d_mu F'(n) by Piola ({same}); det C = (det J)^3 ({detC}): where det J != 0 the equation "
   f"is d_mu F'(n) = 0, so F'(n) = mu is a constant of integration; the equation contains no metric, on every background", same and detC)

# ---------- VQ4 no local excitations ----------
xi = [rand_poly(30 + a) for a in range(4)]
H = sp.Matrix(4, 4, lambda a, m: sp.diff(xi[a], X[m]))
e2 = sp.expand((H.trace()**2 - (H * H).trace()) / 2)
div = sp.expand(sum(sp.diff((xi[m] * sum(sp.diff(xi[b], X[b]) for b in range(4))
                              - sum(xi[b] * sp.diff(xi[m], X[b]) for b in range(4))) / 2, X[m]) for m in range(4)))
eps_ = sp.symbols('eps_')
second = sp.expand((sp.eye(4) + eps_ * H).det()).coeff(eps_, 2)
nullL = sp.expand(e2 - div) == 0 and sp.expand(second - e2) == 0
k = rng.standard_normal(4); M = np.outer(k, k)                      # kinetic matrix of F''(n0)/2 (k.xi)^2 per plane wave
rank = np.linalg.matrix_rank(M)
# de Sitter background, flat slicing: sqrt(-g) = a^3, a = e^{Ht}; labels phi^0 = n0 int a^3 dt, phi^i = x^i give n = n0
Hh, n0 = 0.7, 1.3
def labels(t, xs, pert):
    a3 = np.exp(3 * Hh * t)
    p0 = n0 * (a3 - 1) / (3 * Hh); p = [p0, xs[0], xs[1], xs[2]]
    return pert(p)
def n_at(t, xs, pert, h=1e-5):
    J4 = np.zeros((4, 4)); pt = [t] + list(xs)
    for m in range(4):
        e = np.zeros(4); e[m] = h
        up = labels(*(lambda q: (q[0], q[1:]))(np.array(pt) + e), pert); dn = labels(*(lambda q: (q[0], q[1:]))(np.array(pt) - e), pert)
        J4[:, m] = (np.array(up) - np.array(dn)) / (2 * h)
    return abs(np.linalg.det(J4)) / np.exp(3 * Hh * t)
relabel = lambda p: [p[0] + 0.3 * np.sin(p[1] + 2 * p[2]), p[1] + 0.2 * np.cos(p[3]), p[2], p[3] + 0.1 * np.sin(p[2])]   # shears: unit Jacobian
squeeze = lambda p: [p[0] * (1 + 0.2 * np.sin(p[1])), p[1], p[2], p[3]]
pts = [(rng.uniform(0, 2), rng.uniform(-1, 1, 3)) for _ in range(20)]
dev_rel = max(abs(n_at(t, xs, relabel) - n0) for t, xs in pts)
dev_sq = max(abs(n_at(t, xs, squeeze) - n0) for t, xs in pts)
ok(f"VQ4 no local excitations: the quadratic part of det(1 + d xi) is the total divergence d_mu[(xi^mu div xi - xi^nu d_nu xi^mu)/2] ({nullL}), "
   f"so the quadratic action is F''(n0)/2 (div xi)^2 up to a boundary term; per plane wave the kinetic matrix k_a k_b has rank {rank}: "
   f"every solution with k != 0 has k.xi = 0, i.e. delta n = 0, a relabelling. On de Sitter (H = {Hh}) a unit-Jacobian relabelling keeps "
   f"n = n0 (max dev {dev_rel:.1e}) and so the stress (n F' - F) g; a perturbation that changes n (max dev {dev_sq:.2f}) has d F'(n) != 0 and "
   f"violates the field equation where F'' != 0. The four-label vacuum has no propagating mode and does not relax",
   nullL and rank == 1 and dev_rel < 1e-6 and dev_sq > 0.05)

# ---------- VQ5 status of Lambda: initial data or equilibrium ----------
eps_kv = lambda q: 0.2 + 0.5 * np.sin(q)**2
deps = lambda q: np.sin(q) * np.cos(q)
d2eps = lambda q: np.cos(2 * q)
P = lambda q: -eps_kv(q) + q * deps(q)                             # zero external pressure: P = 0  <=>  rho_V = eps - q eps' = 0
q0 = brentq(P, 3.0, 3.4); chi_inv = q0**2 * d2eps(q0)
F_kv = lambda n: -eps_kv(n); Fp_kv = lambda n: -deps(n)
rhoV = lambda n: n * Fp_kv(n) - F_kv(n)
grid = np.linspace(2.6, 3.6, 201); vals = rhoV(grid)
lin = lambda n: 2.5 * n
Tlin = [(n * 2.5 - lin(n)) for n in (0.3, 1.0, 7.0)]
ok(f"VQ5 on the solution family n = n0 (F'(n0) = mu) the gravitating density rho_V = n F' - F = eps - q eps' takes values from "
   f"{vals.min():.2f} to {vals.max():.2f} (eps = 1/5 + sin^2(q)/2 of 0711.3170): Lambda is fixed by the initial value mu; the q-theory "
   f"equilibrium of an isolated vacuum (P = 0) selects one value, q0 = {q0:.5f} with q0^2 eps''(q0) = {chi_inv:.4f} (0711.3170: 3.20479, "
   f"10.1887), an extra premise; F proportional to n gives T = 0 ({Tlin}), the linear eps excluded in 0711.3170",
   abs(q0 - 3.20479) < 1e-5 and abs(chi_inv - 10.1887) < 1e-3 and vals.min() < 0 < vals.max() and all(abs(t) < 1e-12 for t in Tlin))

# ---------- VQ6 degree quantisation of the total label count ----------
Ng = 24; s = np.linspace(0, 2 * np.pi, Ng, endpoint=False); dv = (2 * np.pi / Ng)**4
G = np.meshgrid(s, s, s, s, indexing='ij')
def total(w, amp):
    # phi^a = w_a x^a + amp * periodic(x); exact derivatives of the trigonometric perturbation
    J4 = np.zeros(G[0].shape + (4, 4))
    for a in range(4):
        J4[..., a, a] = w[a]
    J4[..., 0, 1] += amp * np.cos(G[1] + G[2]); J4[..., 0, 2] += amp * np.cos(G[1] + G[2])
    J4[..., 1, 3] += amp * 0.7 * np.cos(G[3] - G[0]); J4[..., 1, 0] -= amp * 0.7 * np.cos(G[3] - G[0])
    J4[..., 2, 0] += amp * 0.5 * np.cos(2 * G[0])
    J4[..., 3, 2] += amp * 0.4 * np.cos(G[2] + G[0]); J4[..., 3, 0] += amp * 0.4 * np.cos(G[2] + G[0])
    return np.linalg.det(J4).sum() * dv / (2 * np.pi)**4
res = [(w, total(w, 0.6)) for w in [(1, 1, 1, 1), (2, 1, 1, 1), (1, -1, 3, 1), (2, 2, 1, 1)]]
okd = all(abs(v - np.prod(w)) < 1e-9 for w, v in res)
ok(f"VQ6 periodic labels on a 4-torus: int det J d^4x / (2 pi)^4 = degree of the map, an integer, for perturbed winding maps "
   f"{[(w, round(v, 10)) for w, v in res]}: the total label count of a closed region is quantised; with n = n0 the 4-volume is N/n0", okd)

# ---------- VQ7 five labels ----------
J5 = rng.standard_normal((5, 4))
minors5 = [np.linalg.matrix_rank(J5)]
tt, ell = sp.symbols('t ell', positive=True); xs3 = sp.symbols('y1:4', real=True)
r2 = sum(v**2 for v in xs3); e = sp.exp(tt / ell)
Xh = [ell * sp.sinh(tt / ell) + r2 * e / (2 * ell)] + [e * v for v in xs3] + [ell * sp.cosh(tt / ell) - r2 * e / (2 * ell)]
eta5 = sp.diag(1, -1, -1, -1, -1)
constraint = sp.simplify(sum(eta5[A, A] * Xh[A]**2 for A in range(5)).rewrite(sp.exp))
Y = [tt] + list(xs3)
gi_ = sp.Matrix(4, 4, lambda i, j: sp.expand(sum(eta5[A, A] * sp.diff(Xh[A], Y[i]) * sp.diff(Xh[A], Y[j]) for A in range(5)).rewrite(sp.exp)))
aS = sp.exp(tt / ell); gflat = sp.diag(1, -aS**2, -aS**2, -aS**2)
induced_ok = all(sp.simplify(sp.expand(gi_[i, j] - gflat[i, j])) == 0 for i in range(4) for j in range(4))
gin = gflat.inv()
Gam = [[[sp.simplify(sum(gin[i, l] * (sp.diff(gflat[l, j], Y[k]) + sp.diff(gflat[l, k], Y[j]) - sp.diff(gflat[j, k], Y[l])) for l in range(4)) / 2)
         for k in range(4)] for j in range(4)] for i in range(4)]
def ric(i, j):
    return sp.simplify(sum(sp.diff(Gam[k][i][j], Y[k]) - sp.diff(Gam[k][i][k], Y[j])
                           + sum(Gam[k][k][l] * Gam[l][i][j] - Gam[k][j][l] * Gam[l][i][k] for l in range(4)) for k in range(4)))
Ric = sp.Matrix(4, 4, lambda i, j: ric(i, j))
Rs = sp.simplify(sum(gin[i, j] * Ric[i, j] for i in range(4) for j in range(4)))
einst = sp.simplify(Ric - Rs / 2 * gflat - 3 / ell**2 * gflat) == sp.zeros(4)
einst2 = sp.simplify(Ric - Rs / 2 * gflat + 3 / ell**2 * gflat) == sp.zeros(4)
ok(f"VQ7 five labels in four dimensions: the gradient matrix has rank {minors5[0]} <= 4, so dX^0^...^dX^4 = 0 and the five-label density "
   f"vanishes; with the constraint X.X = {constraint} in R^(1,4) the five labels are four labels on a curved label space, the hyperboloid, "
   f"whose induced metric is diag(1, -e^(2t/l), -e^(2t/l), -e^(2t/l)) ({induced_ok}) with R = {Rs} and G_ab = +-3/l^2 g_ab "
   f"({einst or einst2}): de Sitter of radius l. Equating this induced metric with the "
   f"spacetime metric is the premise of embedding (Regge-Teitelboim) gravity, not of the label media",
   minors5[0] <= 4 and constraint == -ell**2 and induced_ok and sp.simplify(Rs**2 - 144 / ell**4) == 0 and (einst or einst2))
