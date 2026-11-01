# 136: reduction of postulates 3 (unoriented labels), 4 (Lorentz-invariant vacuum) and 6 (quantum level 3) to the label medium.
# VC1 k labels in 4D with L = -F(n): T_mu nu = (n F' - F) g_mu nu for every configuration and every F only for k = 4; for k = 1..3 T has a
#     part along the flow/sheet and changes under boosts. A Lorentz-invariant vacuum = the label medium whose label number equals the dimension;
# VC1b for k < 4, T ~ g only for constant F (a medium that does not depend on its labels);
# VC3 every function of the three matter labels is comoving (df ^ dX^1 ^ dX^2 ^ dX^3 = 0) and composites carry no own degree of freedom: a
#     label-dependent vacuum medium needs four labels of its own; a constant F (Lambda) needs no field;
# VC2 the k = 4 vacuum is boost-invariant for every F (w = -1), the k = 3 medium is not;
# OR1 n^2 = |det g^-1(dX, dX)| is even in the label current J: a reflection of label space flips J and keeps n, F(n) and T;
# OR2 a term linear in J (A . J, a label charge) flips sign: an example of a term odd in J, which sees orientation;
# OR3 SDiff (det = +1) does not contain reflections (det = -1); the maps keeping the density, |det| = 1, form a larger group. Unoriented labels
#     = this larger symmetry; every F(n) action has it automatically (OR1), and only terms odd in J break it (OR2);
# OR4 Lorentz scalars odd in J exist without any other field: for the current of three labels in 4D (periodic box) tr((dJ)^3) is a total
#     derivative (mean 0), but tr((dJ)^3)(dJ.dJ) (odd in J) has a nonzero mean: it is not a total derivative, so J-odd actions built from
#     the labels alone exist; the absence of such terms (unoriented labels) is an independent choice of the action class;
# QS1 Madelung: psi = J_nu(k r) e^{i nu phi} solves the free radial Schroedinger equation for every real nu (computed for nu = 0.5, 1, 1.37);
#     the fluid (rho = |psi|^2, v = hbar nu/(m r)) is single-valued and smooth for every nu > 0; the circulation 2 pi hbar nu/m is integer
#     in units 2 pi hbar/m only if psi is single-valued (Wallstrom 1994, known): the local label-fluid equations do not quantise;
# QS2 the quantum-pressure term (hbar^2/8m) (grad rho)^2/rho depends on the labels only through n: it is relabelling-invariant, a gradient term
#     of the label medium with the coefficient hbar (a label); the Bohm force it gives equals -rho grad Q, Q = -(hbar^2/2m) lap sqrt(rho)/sqrt(rho).
# Run: ~/.venvs/jev/bin/python ga_postulates_reduction_check.py
import itertools
import numpy as np
from scipy.special import jv, jvp

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

rng = np.random.default_rng(1360)
eta = np.diag([1.0, -1.0, -1.0, -1.0])

def T_numeric(P, F, g, eps=1e-6):
    # S = int sqrt|g| (-F(n)), n^2 = |det(P g^-1 P^T)|; T_mu nu = (2/sqrt|g|) dS/dg^{mu nu}  (sign convention: T_00 = energy density)
    def lag(ginv):
        gl = np.linalg.inv(ginv)
        n = np.sqrt(abs(np.linalg.det(P @ ginv @ P.T)))
        return np.sqrt(abs(np.linalg.det(gl))) * (-F(n))
    gi0 = np.linalg.inv(g); T = np.zeros((4, 4))
    for m in range(4):
        for k in range(m, 4):
            E = np.zeros((4, 4)); E[m, k] = E[k, m] = eps
            d = (lag(gi0 + E) - lag(gi0 - E)) / (2 * eps)
            T[m, k] = T[k, m] = -2 * d / (2 if m != k else 1) / np.sqrt(abs(np.linalg.det(g)))
    return T

def boost(b):
    c, s = np.cosh(b), np.sinh(b)
    L = np.eye(4); L[0, 0] = L[1, 1] = c; L[0, 1] = L[1, 0] = s
    return L

Fs = {'n^2': (lambda n: n**2, lambda n: 2 * n), 'n^1.5': (lambda n: n**1.5, lambda n: 1.5 * n**0.5), 'exp': (lambda n: np.exp(n), np.exp)}
dev4, aniso = 0.0, {}
for k in (1, 2, 3, 4):
    worst = 0.0
    for _ in range(20):
        if k < 4:   # comoving labels: gradients orthogonal to a timelike flow (rest frame of a random boost), plus noise in that subspace
            P = np.hstack([np.zeros((k, 1)), rng.normal(size=(k, 3))]) @ boost(rng.normal() * 0.5)
        else:
            P = rng.normal(size=(4, 4))
        for name, (F, dF) in Fs.items():
            T = T_numeric(P, F, eta)
            n = np.sqrt(abs(np.linalg.det(P @ np.linalg.inv(eta) @ P.T)))
            rel = np.abs(T - (n * dF(n) - F(n)) * eta).max() / max(abs(F(n)), 1e-12)
            worst = max(worst, rel)
    aniso[k] = worst
print("   max |T - (nF' - F) g| / F for k = 1..4: " + ", ".join(f"{aniso[k]:.1e}" for k in (1, 2, 3, 4)))
ok("VC1 T = (nF' - F) g for every configuration and every F (n^2, n^1.5, e^n) only for k = 4 labels; k = 1, 2, 3 give a stress not "
   "proportional to g", aniso[4] < 1e-5 and min(aniso[k] for k in (1, 2, 3)) > 1e-2)

# VC1b: for k < 4, T ~ g only for a constant F (no dependence on the labels): T = (nF' - F) g + nF' (h-part), so T ~ g iff n F' = 0
Pk3 = np.hstack([np.zeros((3, 1)), rng.normal(size=(3, 3))])
Tc = T_numeric(Pk3, lambda n: 0.7 + 0 * n, eta)
Tl = T_numeric(Pk3, lambda n: 0.7 + 1e-3 * n, eta)
dev_c = np.abs(Tc - (-0.7) * eta).max(); dev_l = np.abs(Tl - (Tl[0, 0] / eta[0, 0]) * eta).max()
ok(f"VC1b for k = 3 a constant F gives T = -F g (deviation {dev_c:.1e}), but any label dependence (F = 0.7 + 0.001 n) breaks T ~ g ({dev_l:.1e}): "
   "four labels are the only label number at which a medium that depends on its labels has T ~ g", dev_c < 1e-8 and dev_l > 1e-5)

P4 = rng.normal(size=(4, 4)); P3 = np.hstack([np.zeros((3, 1)), rng.normal(size=(3, 3))])
Lb = boost(0.8)
T4, T3 = T_numeric(P4, Fs['n^2'][0], eta), T_numeric(P3, Fs['n^2'][0], eta)
dT4 = np.abs(Lb.T @ T4 @ Lb - T4).max() / np.abs(T4).max(); dT3 = np.abs(Lb.T @ T3 @ Lb - T3).max() / np.abs(T3).max()
ok(f"VC2 under a boost (rapidity 0.8) the k = 4 stress is unchanged (relative change {dT4:.1e}, w = -1 for every F) and the k = 3 stress "
   f"changes ({dT3:.2f}): a Lorentz-invariant vacuum is exactly the medium of four labels", dT4 < 1e-5 and dT3 > 0.1)

# ---------- VC3 the fourth label cannot be built from the three comoving labels ----------
worst_dep, best_clock = 0.0, np.inf
for _ in range(200):
    P3r = np.hstack([np.zeros((3, 1)), rng.normal(size=(3, 3))]) @ boost(rng.normal() * 0.5)
    grad_f = rng.normal(size=3)                                               # df = (df/dX^a) dX^a for any smooth f(X^1, X^2, X^3)
    df = grad_f @ P3r
    worst_dep = max(worst_dep, abs(np.linalg.det(np.vstack([P3r, df]))))
    u_ = np.linalg.svd(P3r)[2][-1]
    clock = rng.normal(size=4); clock = clock + (1.0 - clock @ u_) * u_ / (u_ @ u_)   # u . d tau = 1: not comoving
    best_clock = min(best_clock, abs(np.linalg.det(np.vstack([P3r, clock]))))
# where there is no matter the label Jacobian has rank < 3: n = 0 and every composite of the matter labels vanishes or is undefined
P_empty = np.vstack([rng.normal(size=(2, 4)), np.zeros((1, 4))])
n_empty = np.sqrt(abs(np.linalg.det(P_empty @ np.linalg.inv(eta) @ P_empty.T)))
ok(f"VC3 for every f(X^1, X^2, X^3) df ^ dX^1 ^ dX^2 ^ dX^3 = 0 (max |det| {worst_dep:.1e}), while a non-comoving field (u . d tau != 0) gives a "
   f"nonzero 4-form (min |det| {best_clock:.2f}); composites of the derivatives (e.g. n) carry no degree of freedom of their own and n = "
   f"{n_empty:.1e} where matter is absent: a label-dependent vacuum medium needs four labels of its own (the field of axiom D); a constant F "
   "(Lambda, VC1b) is Lorentz invariant without any extra field", worst_dep < 1e-12 and best_clock > 1e-6 and n_empty < 1e-12)

# ---------- orientation ----------
def current(P3_):
    return np.array([(-1)**mu * np.linalg.det(np.delete(P3_, mu, axis=1)) for mu in range(4)])
R = np.diag([-1.0, 1.0, 1.0])                                                   # reflection of label space
Pr = R @ P3
J, Jr = current(P3), current(Pr)
n_of = lambda P_: np.sqrt(abs(np.linalg.det(P_ @ np.linalg.inv(eta) @ P_.T)))
Tr = T_numeric(Pr, Fs['n^2'][0], eta)
ok(f"OR1 a reflection of label space flips the label current (J_r = -J: {np.allclose(Jr, -J)}) and keeps n ({n_of(P3):.6f} = {n_of(Pr):.6f}) and "
   "T (max change %.1e): every F(n) medium is blind to label orientation" % np.abs(Tr - T3).max(),
   np.allclose(Jr, -J) and abs(n_of(P3) - n_of(Pr)) < 1e-12 and np.abs(Tr - T3).max() < 1e-6)
A = rng.normal(size=4)
ok(f"OR2 a coupling linear in J, A.J = {A @ eta @ J:.4f}, becomes {A @ eta @ Jr:.4f}: a term odd in the label current sees orientation "
   "(example: a label charge); every F(n) term does not (OR1)", np.isclose(A @ eta @ Jr, -(A @ eta @ J)))
ok(f"OR3 det R = {np.linalg.det(R):.0f}: a reflection is not in SDiff (det = +1) but keeps the density (|det| = 1); unoriented labels mean the "
   "larger group of density-preserving maps; every F(n) medium has it automatically (OR1), only a term odd in J breaks it (OR2): postulate 3 "
   "is equivalent to the absence of J-odd terms, not a consequence of SDiff", np.isclose(abs(np.linalg.det(R)), 1) and np.linalg.det(R) < 0)


# ---------- OR4 J-odd Lorentz scalars of the label current, 4D ----------
Nb = 16
kb = np.fft.fftfreq(Nb, 1 / Nb)
K4 = np.meshgrid(kb, kb, kb, kb, indexing='ij')
band = (K4[0]**2 + K4[1]**2 + K4[2]**2 + K4[3]**2) <= 2.5                       # band-limited fields: products are alias-free
def d4(f, mu):
    return np.fft.ifftn(1j * K4[mu] * np.fft.fftn(f)).real
perms = list(itertools.permutations(range(4)))
psign = [np.linalg.det(np.eye(4)[list(q)]) for q in perms]
ratios_td, ratios_odd, divs = [], [], []
for seed in (11, 12, 13):
    r = np.random.default_rng(seed)
    dX = np.zeros((3, 4) + (Nb,) * 4)
    for a_ in range(3):
        pert = np.fft.ifftn(np.fft.fftn(r.normal(size=(Nb,) * 4)) * band).real
        pert *= 0.3 / np.abs(pert).max()
        for mu in range(4):
            dX[a_, mu] = d4(pert, mu) + (1.0 if mu == a_ + 1 else 0.0)      # labels X^a = x^a + periodic, flow along x^0
    Jc = np.zeros((4,) + (Nb,) * 4)
    for q, sg in zip(perms, psign):
        Jc[q[0]] += sg * dX[0, q[1]] * dX[1, q[2]] * dX[2, q[3]]           # J^mu = eps^{mu nu rho sigma} dX1 dX2 dX3
    divs.append(np.abs(sum(d4(Jc[mu], mu) for mu in range(4))).max())
    D = np.array([[d4(Jc[m], n) for n in range(4)] for m in range(4)])      # D[m, n] = d_n J^m
    tr3 = np.einsum('ab...,bc...,ca...->...', D, D, D)                        # Lorentz scalar, odd in J
    dd = np.einsum('mn...,pq...,nq,mp->...', D, D, eta, eta)                  # d_n J_m d^n J^m, even
    odd = tr3 * dd
    ratios_td.append(tr3.mean() / np.abs(tr3).mean()); ratios_odd.append(odd.mean() / np.abs(odd).mean())
print("   4D label currents: max div J " + ", ".join(f"{x:.0e}" for x in divs) + "; <tr(dJ^3)>/<|.|> " + ", ".join(f"{x:.0e}" for x in ratios_td)
      + "; <tr(dJ^3)(dJ.dJ)>/<|.|> " + ", ".join(f"{x:.1e}" for x in ratios_odd))
ok("OR4 for the current of three labels in 4D (periodic box, div J = 0 to 1e-14): the Lorentz scalar tr((dJ)^3) = d_n J^m d_r J^n d_m J^r has mean "
   "0 (a total derivative), but tr((dJ)^3)(d_n J_m d^n J^m), odd in J, has a nonzero mean: J-odd Lorentz-invariant actions built from the labels "
   "alone exist; unoriented labels are an independent choice of the action class",
   max(divs) < 1e-12 and max(abs(x) for x in ratios_td) < 1e-12 and min(abs(x) for x in ratios_odd) > 1e-3)

# ---------- quantum level 3: local part vs global part ----------
kk = 1.3
r = np.linspace(0.05, 12, 24001); dr = r[1] - r[0]
res = {}
for nu in (0.5, 1.0, 1.37):
    f = jv(nu, kk * r)
    f1 = kk * jvp(nu, kk * r, 1); f2 = kk**2 * jvp(nu, kk * r, 2)        # analytic derivatives
    radial = f2 + f1 / r + (kk**2 - nu**2 / r**2) * f
    res[nu] = np.abs(radial).max() / np.abs(f).max()
    single = np.isclose(np.exp(2j * np.pi * nu), 1)
    print(f"   nu = {nu}: radial residual {res[nu]:.1e}, psi single-valued: {single}, circulation = {nu} x 2 pi hbar/m")
ok("QS1 psi = J_nu(kr) e^{i nu phi} solves the free Schroedinger equation locally for nu = 0.5, 1, 1.37; the fluid (rho, v = hbar nu/(m r)) is "
   "single-valued for every nu, psi only for integer nu: the local label-fluid equations allow any circulation, integrality is a global "
   "postulate (Wallstrom 1994, known; recomputed here)", max(res.values()) < 1e-10)

# QS2: Bohm force from the gradient term of n
x = np.linspace(-6, 6, 4001); dx = x[1] - x[0]
rho = np.exp(-x**2) * (1 + 0.3 * np.cos(2 * x))
hb, m = 1.0, 1.0
s = np.sqrt(rho)
Q = -(hb**2 / (2 * m)) * np.gradient(np.gradient(s, dx), dx) / s
F_bohm = -rho * np.gradient(Q, dx)
# the same force from the variational derivative of E_q = int (hbar^2/8m) rho'^2/rho: force = -rho d/dx (dE_q/drho)
def dEq(rho_):
    r1 = np.gradient(rho_, dx)
    return (hb**2 / (8 * m)) * (-(r1 / rho_)**2 - 2 * np.gradient(r1 / rho_, dx))
F_var = -rho * np.gradient(dEq(rho), dx)
inner = slice(400, -400)
dev = np.abs(F_bohm[inner] - F_var[inner]).max() / np.abs(F_bohm[inner]).max()
ok(f"QS2 the gradient term (hbar^2/8m)(rho')^2/rho of a medium depending on the labels only through n gives the Bohm force -rho grad Q "
   f"(relative deviation {dev:.1e}): the local part of level 3 is a relabelling-invariant gradient term of the label medium with the label hbar",
   dev < 1e-3)
