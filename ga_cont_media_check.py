# 155/156: paper14, step 2. Three spacelike labels phi^a with action F(B), B^{ab} = g^{mu nu} d_mu phi^a d_nu phi^b, signature (+,-,-,-),
# as in paper1; H = -B is positive definite. The internal group G of label maps fixes which functions of H are allowed:
#   SDiff (fluid):       all phi -> f(phi) with unit Jacobian;                     F(n), n = sqrt(det H)
#   G_S (smectic):       phi^3 -> phi^3 + c, (phi^1, phi^2) -> area-preserving maps that may depend on phi^3;   F(n, H^33)
#   G_C (columnar):      (phi^1, phi^2) -> rigid motions of the plane, phi^3 -> phi^3 + g(phi^1, phi^2);     F(n, tr H_2, det H_2)
#   ISO(3) (isotropic solid): rigid motions of label space;                        F(n, two shape invariants)
#   cubic point group x translations (cubic crystal).
# CM1 G acts on the gradients at a point through its Jacobian M (J -> J M^T, H -> M H M^T). Each F is invariant under random Jacobians
#     of its G and not under a generic unit-Jacobian one (except the fluid). The number of independent invariants of H is
#     6 - (rank of the Lie-algebra action at a generic H): 1, 2, 3, 3 for SDiff, G_S, G_C, ISO(3). Reflections of phi^1 or phi^3 leave every
#     test action unchanged.
# CM2 stress at the background phi^a = x^a: p_ij = F delta_ij - 2 dF/dH_ij. Fluid and isotropic solid: one pressure; smectic and columnar:
#     two (uniaxial), with p_zz - p_perp = -2 dF/dH33 for the smectic.
# CM3 sound modes from the quadratic action: (propagating, omega = 0) counts by direction of k: fluid (1,2) everywhere; smectic (2,1)
#     generic and (1,2) along and across the layer normal; columnar (3,0) generic, (2,1) for k in the plane, (1,2) along the columns;
#     isotropic solid and cubic crystal (3,0).
# CM4 the omega = 0 modes at a generic direction are the local relabellings of G: displacements xi e^{ikx} whose Jacobian xi k^T lies in
#     the local Lie algebra of G (fluid: sl(3); smectic: upper block in sl(2), free third column, zero third row; columnar: only the
#     third row, with zero third entry). The dimensions 2, 1, 0 match CM3.
# CM5 smectic second sound: the product of the two omega^2 is proportional to k_z^2 k_perp^2, i.e. to sin^2 cos^2 of the angle to the
#     layer normal (known form, Martin-Parodi-Pershan 1972), the slow branch vanishes at 0 and 90 degrees.
# CM6 fluid c^2 = n F''/F' = dp/drho; a superfluid (one timelike label, P(X), paper1 scalar medium, Son 2002) with the same equation of
#     state has the same c^2 and one mode only, no transverse modes.
# CM8 with a layer stress (p_zz != p_perp, set by dF/dH33 = -tau) the in-plane undulation gets omega^2 proportional to the stress: stiff for
#     one sign, unstable for the other (known: Helfrich-Hurault); without it the undulation is soft at this order (known: de Gennes).
# CM7 relabellings are gauge only where they are symmetries: a non-rigid unit-Jacobian relabelling leaves the fluid action unchanged and
#     changes the solid action (strain energy); in a solid the labels are physical (reference positions).
# Run: ~/.venvs/jev/bin/python ga_cont_media_check.py
import numpy as np
import sympy as sp

rng = np.random.default_rng(14)

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

Jt = sp.Matrix(1, 3, sp.symbols('t1:4'))                 # d_t phi^a
Js = sp.Matrix(3, 3, sp.symbols('s1:10'))                # d_i phi^a (row i, column a)
H = Js.T * Js - Jt.T * Jt                                # H = -B
VARS = list(Jt) + list(Js)
BG = {**{v: 0 for v in Jt}, **{Js[i, j]: (1 if i == j else 0) for i in range(3) for j in range(3)}}

def n_of(Hm): return sp.sqrt(Hm.det())
def shape(Hm):                                           # unimodular part, its traceless square
    Hh = Hm / Hm.det() ** sp.Rational(1, 3)
    return (Hh * Hh).trace() - Hh.trace() ** 2 / 3, Hh

def F_fluid(Hm):
    n = n_of(Hm); return -n - sp.Rational(3, 10) * n**2
def F_smectic(Hm, tau=0):
    n = n_of(Hm); h = Hm[2, 2]
    return -n - sp.Rational(3, 10) * n**2 - tau * h - sp.Rational(1, 10) * (h - 1)**2 + sp.Rational(1, 20) * (h - 1) * (n - 1)
def F_columnar(Hm, tau=0):
    n = n_of(Hm); H2 = Hm[:2, :2]; t, d = H2.trace(), H2.det()
    return -n - sp.Rational(3, 10) * n**2 - tau * t - sp.Rational(1, 40) * (t - 2)**2 - sp.Rational(1, 20) * (t**2 - 4 * d) / 4
F_smectic_st = lambda Hm: F_smectic(Hm, sp.Rational(1, 5))
F_columnar_st = lambda Hm: F_columnar(Hm, sp.Rational(1, 5))
def F_iso(Hm):
    n = n_of(Hm); s, _ = shape(Hm); return -n - sp.Rational(3, 10) * n**2 - sp.Rational(1, 10) * s
def F_cubic(Hm):
    n = n_of(Hm); s, Hh = shape(Hm); tr = Hh.trace()
    return -n - sp.Rational(3, 10) * n**2 - sp.Rational(1, 10) * s - sp.Rational(1, 40) * sum((Hh[a, a] - tr / 3)**2 for a in range(3))

MEDIA = [('fluid (SDiff)', F_fluid), ('smectic (G_S)', F_smectic), ('columnar (G_C)', F_columnar), ('isotropic solid (ISO(3))', F_iso),
         ('cubic crystal', F_cubic)]

# ---------- CM1: invariance and number of invariants ----------
Hsym = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f'h{min(i, j)}{max(i, j)}'))
def rand_sl(k):
    A = rng.normal(size=(k, k)); A /= np.cbrt(np.linalg.det(A)) if k == 3 else np.sign(np.linalg.det(A)) * np.sqrt(abs(np.linalg.det(A)))
    return A
def rot3():
    q, r = np.linalg.qr(rng.normal(size=(3, 3))); q *= np.sign(np.diag(r)); return q * np.sign(np.linalg.det(q))
def rot2():
    a = rng.uniform(0, 2 * np.pi); return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
def M_SDiff(): return rand_sl(3)
def M_S():
    M = np.eye(3); M[:2, :2] = rand_sl(2); M[:2, 2] = rng.normal(size=2); return M
def M_C():
    M = np.eye(3); M[:2, :2] = rot2(); M[2, :2] = rng.normal(size=2); return M
def M_iso(): return rot3()
def M_cubic():
    P = np.eye(3)[rng.permutation(3)]; S = np.diag(rng.choice([-1, 1], 3)); return P @ S
GROUPS = {'fluid (SDiff)': M_SDiff, 'smectic (G_S)': M_S, 'columnar (G_C)': M_C, 'isotropic solid (ISO(3))': M_iso, 'cubic crystal': M_cubic}

def F_num(F, Hn):
    Hm = sp.Matrix(Hn.tolist()); return float(F(Hm))
H0 = rng.normal(size=(3, 3)); H0 = H0 @ H0.T + 2 * np.eye(3)
inv_ok, msg = True, []
for name, F in MEDIA:
    base = F_num(F, H0)
    dev_in = max(abs(F_num(F, (M := GROUPS[name]()) @ H0 @ M.T) - base) for _ in range(4))
    dev_out = abs(F_num(F, (M := rand_sl(3)) @ H0 @ M.T) - base)
    good = dev_in < 1e-9 and (dev_out > 1e-3 or name.startswith('fluid'))
    inv_ok &= good; msg.append(f"{name.split()[0]}: in-G {dev_in:.0e}, generic SDiff {dev_out:.1e}")

def lie_rank(gens, Hn):
    vecs = [(X @ Hn + Hn @ X.T)[np.triu_indices(3)] for X in gens]
    return np.linalg.matrix_rank(np.array(vecs), tol=1e-9)
E = lambda i, j: np.eye(3)[:, [i]] @ np.eye(3)[[j], :]
g_sl3 = [E(i, j) for i in range(3) for j in range(3) if i != j] + [E(0, 0) - E(1, 1), E(1, 1) - E(2, 2)]
g_S = [E(0, 1), E(1, 0), E(0, 0) - E(1, 1), E(0, 2), E(1, 2)]
g_C = [E(0, 1) - E(1, 0), E(2, 0), E(2, 1)]
g_so3 = [E(0, 1) - E(1, 0), E(0, 2) - E(2, 0), E(1, 2) - E(2, 1)]
ninv = {k: 6 - lie_rank(g, H0) for k, g in [('SDiff', g_sl3), ('G_S', g_S), ('G_C', g_C), ('ISO(3)', g_so3)]}
refl = max(abs(F_num(F, R @ H0 @ R.T) - F_num(F, H0)) for _, F in MEDIA for R in (np.diag([-1., 1, 1]), np.diag([1., 1, -1])))
ok("CM1 each F is invariant under its G and (except the fluid) not under a generic unit-Jacobian map: " + "; ".join(msg) +
   f". Independent invariants of H: {ninv}. Reflections of phi^1 or phi^3 change no test action (max {refl:.0e})",
   inv_ok and list(ninv.values()) == [1, 2, 3, 3] and refl < 1e-9)

# ---------- quadratic action at the background ----------
def quad(F):
    L = F(H)
    g = [sp.diff(L, v) for v in VARS]
    C = np.array([[float(sp.diff(gi, v).subs(BG)) for v in VARS] for gi in g])
    grad = np.array([float(gi.subs(BG)) for gi in g])
    return float(L.subs(BG)), grad, C
QUAD = {name: quad(F) for name, F in MEDIA}
QUAD['smectic stressed+'] = quad(F_smectic_st)
QUAD['smectic stressed-'] = quad(lambda Hm: F_smectic(Hm, -sp.Rational(1, 5)))

def stress(F):
    Hm = sp.Matrix(3, 3, lambda i, j: Hsym[i, j]); L = F(Hm)
    sub = {Hsym[i, j]: (1 if i == j else 0) for i in range(3) for j in range(3)}
    Fv = float(L.subs(sub))
    P = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            dF = float(sp.diff(L, Hsym[i, j]).subs(sub)) * (1 if i == j else 0.5)
            P[i, j] = Fv * (i == j) - 2 * dF
    return P

# ---------- CM2 ----------
pr, good2 = [], True
STRESSED = {'smectic (G_S)': F_smectic_st, 'columnar (G_C)': F_columnar_st}
for name, F0_ in MEDIA:
    F = STRESSED.get(name, F0_)
    P = stress(F); ev = np.sort(np.linalg.eigvalsh(P)); nd = len(set(np.round(ev, 9)))
    offd = np.abs(P - np.diag(np.diag(P))).max()
    want = 2 if name.split()[0] in ('smectic', 'columnar') else 1
    good2 &= nd == want and offd < 1e-12
    pr.append(f"{name.split()[0]} p = {', '.join(f'{x:.3f}' for x in sorted(set(np.round(ev, 9))))}")
n_, h_ = sp.symbols('n h', positive=True)
Fs = -n_ - sp.Rational(3, 10) * n_**2 - sp.Rational(1, 5) * h_ - sp.Rational(2, 5) * (h_ - 1)**2 + sp.Rational(1, 4) * (h_ - 1) * (n_ - 1)
Psm = stress(F_smectic_st); aniso = Psm[2, 2] - Psm[0, 0]; pred = float(-2 * sp.diff(Fs, h_).subs({n_: 1, h_: 1}))
ok("CM2 principal stresses at the background (smectic and columnar with a layer or column stress, tau = 0.2): " + "; ".join(pr) + f"; smectic p_zz - p_perp = {aniso:.4f} = -2 dF/dH33 = {pred:.4f}",
   good2 and abs(aniso - pred) < 1e-12)

# ---------- CM3 modes ----------
def modes(name, k):
    Fv, grad, C = QUAD[name]
    Ctt = C[:3, :3]
    Css = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            Css[a, b] = sum(C[3 + 3 * i + a, 3 + 3 * j + b] * k[i] * k[j] for i in range(3) for j in range(3))
    Cts = C[:3, 3:]
    assert np.abs(Cts).max() < 1e-12
    T = Ctt                                              # kinetic matrix (L2 = 1/2 pi_t T pi_t - 1/2 dpi K dpi)
    K = -Css
    w2, V = np.linalg.eig(np.linalg.solve(T, K))
    return np.sort(w2.real), V.real, T, K
def kvec(theta, phi=0.7):
    return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
def count(w2, tol=1e-10): return int((w2 > tol).sum()), int((np.abs(w2) <= tol).sum())
expect = {'fluid (SDiff)': {0: (1, 2), 1: (1, 2), 2: (1, 2)}, 'smectic (G_S)': {0: (1, 2), 1: (2, 1), 2: (1, 2)},
          'columnar (G_C)': {0: (1, 2), 1: (3, 0), 2: (2, 1)}, 'isotropic solid (ISO(3))': {0: (3, 0), 1: (3, 0), 2: (3, 0)},
          'cubic crystal': {0: (3, 0), 1: (3, 0), 2: (3, 0)}}
thetas = {0: 0.0, 1: 0.6, 2: np.pi / 2}
good3, rep = True, []
for name, _ in MEDIA:
    cs = {}
    for key, th in thetas.items():
        w2, V, T, K = modes(name, kvec(th))
        cs[key] = count(w2)
        good3 &= bool(np.all(np.linalg.eigvalsh(T) > 0)) and bool(np.all(w2 > -1e-10)) and bool(np.all(w2 < 1))
    good3 &= cs == expect[name]
    rep.append(f"{name.split()[0]} (along normal/column, generic, in plane) = {cs[0]}, {cs[1]}, {cs[2]}")
w2i, Vi, _, _ = modes('isotropic solid (ISO(3))', kvec(0.6))
cL, cT = np.sqrt(w2i[-1]), np.sqrt(w2i[0])
ok("CM3 (propagating, omega = 0) by direction: " + "; ".join(rep) + f"; kinetic matrix positive, 0 <= c^2 < 1 (stable and causal); isotropic solid c_L = {cL:.4f}, "
   f"c_T = {cT:.4f} (twice)", good3 and abs(w2i[0] - w2i[1]) < 1e-10 and cT < cL)

# ---------- CM4 omega = 0 modes = local relabellings ----------
def local_dim(alg_test, k):
    # dimension of {xi : xi k^T in the local algebra}; alg_test returns the linear constraints on the 3x3 matrix X = xi k^T
    rows = []
    for c in alg_test:
        rows.append([c(np.outer(np.eye(3)[a], k)) for a in range(3)])
    return 3 - np.linalg.matrix_rank(np.array(rows), tol=1e-10) if rows else 3
loc = {'fluid (SDiff)': [lambda X: np.trace(X)],
       'smectic (G_S)': [lambda X: X[2, 0], lambda X: X[2, 1], lambda X: X[2, 2], lambda X: X[0, 0] + X[1, 1]],
       'columnar (G_C)': [lambda X: X[0, 0], lambda X: X[0, 1], lambda X: X[0, 2], lambda X: X[1, 0], lambda X: X[1, 1],
                          lambda X: X[1, 2], lambda X: X[2, 2]]}
good4, rep4 = True, []
for name, cons in loc.items():
    k = kvec(0.6); d = local_dim(cons, k); w2, V, T, K = modes(name, k)
    z = [V[:, i] for i in range(3) if abs(w2[i]) < 1e-10] if False else None
    nz = count(w2)[1]
    # check that the local-relabelling displacements are zero modes of K
    xis = []
    for xi in np.eye(3):
        pass
    null = []
    A = np.array([[c(np.outer(np.eye(3)[a], k)) for a in range(3)] for c in cons])
    _, s, vt = np.linalg.svd(A); ns = vt[np.sum(s > 1e-10):]
    res = max([np.linalg.norm(K @ v) for v in ns], default=0.0)
    good4 &= d == nz and res < 1e-10
    rep4.append(f"{name.split()[0]}: local relabellings {d}, omega = 0 modes {nz}, |K xi| = {res:.0e}")
ok("CM4 at a generic direction the omega = 0 modes are the local relabellings of G: " + "; ".join(rep4) +
   ". The extra soft modes along special directions (smectic undulation in the plane, column tilt along the columns) come from rotations",
   good4)

# ---------- CM5 smectic second sound ----------
prods, slow = [], []
for th in [0.2, 0.5, 0.8, 1.1, 1.4]:
    w2, _, _, _ = modes('smectic (G_S)', kvec(th)); nzv = w2[w2 > 1e-10]
    prods.append(np.prod(nzv) / (np.sin(th)**2 * np.cos(th)**2)); slow.append(nzv.min())
w2_0 = modes('smectic (G_S)', kvec(1e-3))[0]; w2_90 = modes('smectic (G_S)', kvec(np.pi / 2 - 1e-3))[0]
spread = (max(prods) - min(prods)) / np.mean(prods)
ok(f"CM5 smectic: omega_1^2 omega_2^2 / (sin^2 cos^2) constant to {spread:.0e} over five angles (k = 1); slow branch omega^2 = "
   f"{w2_0[w2_0 > 1e-14].min():.1e} at 0.001 rad and {w2_90[w2_90 > 1e-14].min():.1e} at 0.001 rad from 90 degrees: second sound vanishes "
   "along and across the normal", spread < 1e-9 and w2_0[w2_0 > 1e-14].min() < 1e-5 and w2_90[w2_90 > 1e-14].min() < 1e-5)

# ---------- CM6 fluid vs superfluid ----------
nn = sp.Symbol('n', positive=True); gam = sp.Rational(5, 3); a = sp.Rational(2, 5)
Ff = -(nn + a * nn**gam)
rho = -Ff; p = Ff - nn * sp.diff(Ff, nn); mu = -sp.diff(Ff, nn)
c2_formula = (nn * sp.diff(Ff, nn, 2) / sp.diff(Ff, nn))
c2_eos = sp.diff(p, nn) / sp.diff(rho, nn)
n0 = sp.Rational(7, 10)
cf = float(c2_formula.subs(nn, n0)); ce = float(c2_eos.subs(nn, n0))
w2f = modes('fluid (SDiff)', kvec(0.6))[0]
F0, Fp, Fpp = [float(sp.diff(-nn - sp.Rational(3, 10) * nn**2, nn, k).subs(nn, 1)) for k in range(3)]
# superfluid: P(X) = p(mu) with mu = sqrt(X); c^2 = P_X / (P_X + 2 X P_XX), evaluated by the chain rule at mu(n0)
mu0 = float(mu.subs(nn, n0))
dp_dmu = float((sp.diff(p, nn) / sp.diff(mu, nn)).subs(nn, n0))                 # = n
d2p_dmu2 = float((sp.diff(sp.diff(p, nn) / sp.diff(mu, nn), nn) / sp.diff(mu, nn)).subs(nn, n0))
X0 = mu0**2; PX = dp_dmu / (2 * mu0); PXX = (d2p_dmu2 - dp_dmu / mu0) / (4 * X0)
c2_sf = PX / (PX + 2 * X0 * PXX)
ok(f"CM6 fluid F = -(n + 0.4 n^(5/3)) at n = 0.7: c^2 = nF''/F' = {cf:.6f} = dp/drho = {ce:.6f}; superfluid P(X) with the same equation of "
   f"state: c^2 = {c2_sf:.6f}, one mode; the background fluid of CM3 has c^2 = {w2f[-1]:.6f} = nF''/F' = {Fpp / Fp:.6f} and two omega = 0 "
   "transverse modes", abs(cf - ce) < 1e-12 and abs(c2_sf - cf) < 1e-9 and abs(w2f[-1] - Fpp / Fp) < 1e-10)

# ---------- CM7 gauge only where symmetric ----------
Msh = np.array([[1, 0.3, 0], [0, 1, 0], [0, 0, 1.0]])   # simple shear of labels, unit Jacobian, not rigid
Hb = np.eye(3)
dF = {name.split()[0]: F_num(F, Msh @ Hb @ Msh.T) - F_num(F, Hb) for name, F in MEDIA}
ok("CM7 a shear relabelling with unit Jacobian changes F by: " + ", ".join(f"{k} {v:+.4f}" for k, v in dF.items()) +
   "; it is a symmetry (gauge under the series premise) only for the fluid; in the smectic it is a symmetry only within the layers, "
   "in the solids it is a strain", abs(dF['fluid']) < 1e-12 and all(abs(dF[k]) > 1e-3 for k in ('columnar', 'isotropic', 'cubic')))

# ---------- CM8 layer stress: undulation stiffened or unstable ----------
w2p = modes('smectic stressed+', kvec(np.pi / 2))[0]; w2m = modes('smectic stressed-', kvec(np.pi / 2))[0]
w20 = modes('smectic (G_S)', kvec(np.pi / 2))[0]
ok(f"CM8 smectic, k in the layer plane: without layer stress omega^2 = {', '.join(f'{x:.3f}' for x in w20)} (undulation soft); with "
   f"p_zz - p_perp = +0.4: {', '.join(f'{x:.3f}' for x in w2p)} (undulation stiff); with -0.4: {', '.join(f'{x:.3f}' for x in w2m)} "
   "(undulation unstable, the Helfrich-Hurault sign): the extra soft mode needs isotropic background stress",
   count(w20) == (1, 2) and count(w2p) == (2, 1) and w2m.min() < -1e-3)

print("ALL PASS")
