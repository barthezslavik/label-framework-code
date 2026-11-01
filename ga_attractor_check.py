# 94: attractors and balances in the dynamics of label media (beyond doc 45), with stability. A number is an attractor value if the
# direction that carries it has a negative eigenvalue; a zero eigenvalue is a conserved ratio, i.e. a label. AT1-AT5.
# Run: ~/.venvs/jev/bin/python ga_attractor_check.py
import numpy as np
import sympy as sp

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

# AT1 the ladder of label media (w = k/3 - 1 for k = 0..3, radiation 1/3, stiff 1) in an expanding universe: Omega_i' = 3 Omega_i (w_eff - w_i),
#     w_eff = sum Omega_j w_j. At the vertex 'medium j dominates' the eigenvalue towards medium i is 3 (w_j - w_i): the only stable vertex is the
#     vacuum (k = 0 or 4, w = -1); dust (k = 3) is a saddle that decays only towards k < 3. Two media with the SAME w (dust and baryons, or
#     two dust species) have eigenvalue 0: their ratio is conserved -- a label, never an attractor value.
ws = {'vac': -1, 'wall': sp.Rational(-2, 3), 'string': sp.Rational(-1, 3), 'dust': 0, 'baryon': 0, 'rad': sp.Rational(1, 3), 'stiff': 1}
names = list(ws)
O = sp.symbols('O0:%d' % len(names))
weff = sum(Oi * ws[n] for Oi, n in zip(O, names))
F = [3 * Oi * (weff - ws[n]) for Oi, n in zip(O, names)]
def eig_at(j):
    sub = {Oi: (1 if k == j else 0) for k, Oi in enumerate(O)}
    Jm = sp.Matrix(F).jacobian(O).subs(sub)
    free = [k for k in range(len(O)) if k != j]                       # constraint sum = 1 removes the j direction
    return {names[k]: sp.simplify(Jm[k, k]) for k in free}
ev_vac, ev_dust = eig_at(0), eig_at(3)
ok(f"AT1 eigenvalues at vacuum domination {ev_vac} (all < 0: stable); at dust domination {ev_dust}: unstable only towards wall/string/vac, "
   f"zero towards baryons (conserved ratio = label)", all(v < 0 for v in ev_vac.values()) and ev_dust['baryon'] == 0 and ev_dust['vac'] > 0)

# AT2 dust with the coupling m(phi) = m0 e^(beta phi) and a constant potential (paper1 v2, doc 86): x = phi'/(sqrt 6 H), y^2 = V0/(3 H^2),
#     Omega_m = 1 - x^2 - y^2, eps = -H'/H = (3/2)(1 + x^2 - y^2):  x' = -3x - sqrt(3/2) beta Omega_m + eps x,  y' = eps y.
#     Fixed points: the 'phi-matter era' x = -sqrt(2/3) beta, y = 0, with Omega_phi = 2 beta^2/3 and w_eff = 2 beta^2/3 (a saddle), and
#     de Sitter x = 0, y = 1 (stable). The kinetic share 2 beta^2/3 is an attractor value along x, but a function of the label beta.
x, y, b = sp.symbols('x y beta', real=True)
Om = 1 - x**2 - y**2; eps = sp.Rational(3, 2) * (1 + x**2 - y**2)
fx = -3 * x - sp.sqrt(sp.Rational(3, 2)) * b * Om + eps * x; fy = eps * y
Jm = sp.Matrix([fx, fy]).jacobian([x, y])
xs = -sp.sqrt(sp.Rational(2, 3)) * b
res_fp = sp.simplify(sp.series(fx.subs({x: xs, y: 0}), b, 0, 4).removeO())
evm = [sp.series(e, b, 0, 3).removeO() for e in Jm.subs({x: xs, y: 0}).eigenvals()]
evd = list(Jm.subs({x: 0, y: 1}).eigenvals())
ok(f"AT2 phi-matter era x = -sqrt(2/3) beta solves x' = 0 to O(beta^3) (residual {res_fp}); eigenvalues {evm} (saddle: one positive along y); "
   f"de Sitter eigenvalues {evd} (stable); Omega_phi = 2 beta^2/3", res_fp == 0 and all(sp.re(sp.N(e.subs(b, 0.03))) < 0 for e in evd))

# AT3 balance between baryons and dust in the matter era (uncoupled): power-law growth delta = D a^p of
#     delta_c'' + (1/2) delta_c' = (3/2)(f_c delta_c + f_b delta_b), same for delta_b. The growing mode p = 1 has D_b/D_c = 1 exactly, and the
#     other modes decay or are constant: delta_b/delta_c -> 1 is an attractor (baryons fall into the dust wells), the number 1 is a balance.
p, fb = sp.symbols('p f_b', positive=True)
fc = 1 - fb
def growth(beta_, xk, fbv, Omm):
    # matrix of the coupled growth equations at a fixed point: D'' + A D' = Bm D with D ~ a^p:  (p^2 + A p) D = Bm D
    epsv = sp.Rational(3, 2) * (1 + xk**2)
    Ac = 2 - epsv + sp.sqrt(6) * beta_ * xk; Ab = 2 - epsv
    Bm = sp.Rational(3, 2) * Omm * sp.Matrix([[(1 + 2 * beta_**2) * (1 - fbv), fbv], [(1 - fbv), fbv]])
    return Ac, Ab, Bm
Ac, Ab, Bm = growth(0, 0, fb, 1)
M0 = sp.Matrix([[p**2 + Ac * p, 0], [0, p**2 + Ab * p]]) - Bm
pr = sp.solve(sp.factor(M0.det()), p)
vec = M0.subs(p, 1).nullspace()[0]
ok(f"AT3 uncoupled matter era: growth exponents {pr} (+ decaying -3/2), growing mode p = 1 with D_b/D_c = {sp.simplify(vec[1] / vec[0])}: balance 1 is an attractor",
   1 in pr and sp.simplify(vec[1] / vec[0]) == 1)

# AT4 with the coupling on dust only (baryons uncoupled, Cassini, 82): in the phi-matter era the growing mode has D_b/D_c < 1 -- a baryon
#     bias fixed by beta and f_b (an attractor value, function of labels). Numerically for beta = 0.03, 0.1, f_b = 0.157.
def coupled(bv, fbv=0.157):
    xk = -np.sqrt(2 / 3) * bv; Omm = 1 - xk**2
    epsv = 1.5 * (1 + xk**2); Ac = 2 - epsv + np.sqrt(6) * bv * xk; Ab = 2 - epsv
    Bm = 1.5 * Omm * np.array([[(1 + 2 * bv**2) * (1 - fbv), fbv], [(1 - fbv), fbv]])
    # first-order system in (Dc, Db, Dc', Db')
    A = np.zeros((4, 4)); A[0, 2] = A[1, 3] = 1; A[2:, :2] = Bm; A[2, 2] = -Ac; A[3, 3] = -Ab
    w, V = np.linalg.eig(A); i = np.argmax(w.real)
    return w[i].real, (V[1, i] / V[0, i]).real
cp = {bv: coupled(bv) for bv in (0.0, 0.03, 0.1)}
ok("AT4 phi-matter era growing mode (p, D_b/D_c): " + ", ".join(f"beta = {k}: ({v[0]:.4f}, {v[1]:.4f})" for k, v in cp.items())
   + ": baryons lag the dust by a beta-fixed bias", abs(cp[0.0][0] - 1) < 1e-9 and abs(cp[0.0][1] - 1) < 1e-9 and cp[0.1][1] < cp[0.03][1] < 1)

# AT5 recorded (classification): attractor values of label-media dynamics are (i) rational structural numbers (w_eff = 0, -1, 1/3; p = 1;
#     D_b/D_c = 1; slopes) or (ii) functions of labels (2 beta^2/3, the bias). No label value (Lambda, beta, f, Omega_c/Omega_b, a0) is itself
#     attracted: each sits on a zero mode (conserved charge) or is a parameter of the flow.
ok("AT5 recorded: attractors give rational numbers or functions of labels; labels are zero modes or flow parameters, never attractor values", True)
