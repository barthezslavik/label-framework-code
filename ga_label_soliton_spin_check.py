# 134: fermions from a compact label space, stage 2. If the label space is S^3 (unit sphere in R^4) and labels go to a fixed point at
# spatial infinity, the label current N is the pull-back of the volume form of S^3; the label map is a Skyrme-type field and the
# sextic term of the BPS Skyrme model (Adam, Sanchez-Guillen, Wereszczynski 2010), quadratic in this current, is a fluid of such labels with
# the full SDiff of label space (its potential term breaks SDiff, doc 135 CL6). Equations borrowed, steps recomputed.
# LS1 the number of labels int N = degree x Vol(S^3) = 2 pi^2 for the hedgehog, unchanged by a non-spherical deformation: the label
#     number is a topological degree (protected), unlike the non-compact sigma model of doc 125;
# LS2 dust labels (F = m |n|): static energy = m int |n| >= m |degree| 2pi^2, equality iff the Jacobian has one sign (no folds, no
#     lump-antilump pairs); for such configurations all sizes are degenerate (scaling test); a folded profile has more energy; the stiff
#     step F ~ n^2 scales as lambda^-3 and needs one more term to fix a size;
# LS3 hedgehog: a spatial rotation equals a rotation of label space, X(R x) = diag(1, R) X(x);
# LS4 collective coordinate A in SU(2): label rotations act from the left, spatial rotations from the right; on the matrix elements of
#     spin j both Casimirs equal j(j+1) (computed for j = 1/2 and 1): J = I; Psi(-A) = (-1)^(2j) Psi(A);
# LS5 if label rotations are gauge, I = 0 forces J = 0 (spin-0 lump); if they are global (physical label charges), the Finkelstein-
#     Rubinstein sign Psi(-A) = -Psi(A) allows exactly the half-integer J = I (1/2, 3/2, ...); with the rigid-rotor energy
#     J(J+1)/(2 Theta), Theta = int m n r^2 of the rotating dust lump (moment of inertia > 0), the lowest state is J = I = 1/2;
# LS6 (known, not recomputed) pi_1 of degree-1 maps is pi_4(S^3) = Z2; the 2 pi rotation is its generator and the exchange of two
#     lumps is homotopic to it (Finkelstein-Rubinstein 1968, Williams 1970): the FR sign makes lumps fermions with J = 1/2.
# Run: ~/.venvs/jev/bin/python ga_label_soliton_spin_check.py
import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

def prof(r, a=1.0):
    return np.pi * np.exp(-r / a)                                     # f(0) = pi, f(inf) = 0
def hedgehog(x, a=1.0, eps=0.0):
    r = np.linalg.norm(x, axis=-1, keepdims=True)
    f = prof(r, a)
    xh = np.where(r > 0, x / np.maximum(r, 1e-300), 0.0)
    X = np.concatenate([np.cos(f), np.sin(f) * xh], axis=-1)
    if eps:                                                            # non-spherical, compactly supported deformation, renormalised
        bump = np.exp(-np.sum((x - np.array([0.7, 0.2, -0.3]))**2, axis=-1, keepdims=True) / 0.3)
        X = X + eps * bump * np.array([0.3, 1.0, -0.5, 0.8])
        X = X / np.linalg.norm(X, axis=-1, keepdims=True)
    return X
def density(Xf, a=1.0, M=(160, 80, 160)):
    # degree as the integral of the pulled-back 3-form in coordinates (s, theta, phi), r = a s/(1 - s): no metric factor is needed
    s = (np.arange(M[0]) + 0.5) / M[0]; th = (np.arange(M[1]) + 0.5) * np.pi / M[1]; ph = np.arange(M[2]) * 2 * np.pi / M[2]
    S, T, P = np.meshgrid(s, th, ph, indexing='ij')
    R = a * S / (1 - S)
    x = np.stack([R * np.sin(T) * np.cos(P), R * np.sin(T) * np.sin(P), R * np.cos(T)], axis=-1)
    X = Xf(x)
    ds, dth, dph = 1 / M[0], np.pi / M[1], 2 * np.pi / M[2]
    d0 = np.gradient(X, ds, axis=0); d1 = np.gradient(X, dth, axis=1)
    d2 = (np.roll(X, -1, axis=2) - np.roll(X, 1, axis=2)) / (2 * dph)   # periodic in phi
    n = np.linalg.det(np.stack([X, d0, d1, d2], axis=-1))
    return n, ds * dth * dph, x
# ---------- LS1 label number = degree ----------
radial = 4 * np.pi * quad(lambda r: -np.sin(prof(r))**2 * (-np.pi * np.exp(-r)) , 0, np.inf)[0]
n0, h, x = density(lambda x: hedgehog(x))
n1, _, _ = density(lambda x: hedgehog(x, eps=0.6))
I0, I1 = abs(n0.sum()) * h, abs(n1.sum()) * h
print(f"   radial integral 4 pi int sin^2 f |f'| dr = {radial:.6f} (2 pi^2 = {2 * np.pi**2:.6f}); grid: hedgehog {I0:.4f}, deformed {I1:.4f}")
ok(f"LS1 compact label space S^3: int N = {I0 / (2 * np.pi**2):.4f} x 2 pi^2 for the hedgehog and {I1 / (2 * np.pi**2):.4f} x 2 pi^2 after a "
   "non-spherical deformation: the number of labels is the degree of the label map, an integer protected by topology",
   abs(radial - 2 * np.pi**2) < 1e-6 and abs(I0 / (2 * np.pi**2) - 1) < 0.02 and abs(I1 / (2 * np.pi**2) - 1) < 0.02)

# ---------- LS2 dust energy is topological; stiff step needs a size term ----------
E_dust, E_stiff = [], []
for a in (0.7, 1.0, 1.4):
    n_, w_, x_ = density(lambda x: hedgehog(x, a=a), a=a)
    E_dust.append(abs(n_.sum()) * w_)
    # stiff energy int n_phys^2 d^3x = int n_coord^2 / |J_coord| over coordinates; J_coord = r^2 sin(theta) dr/ds
    S_ = np.linalg.norm(x_, axis=-1); sv = S_ / (a + S_); Jc = S_**2 * np.sin(np.arccos(np.clip(x_[..., 2] / np.maximum(S_, 1e-300), -1, 1))) * a / (1 - sv)**2
    E_stiff.append((n_**2 / np.maximum(Jc, 1e-300)).sum() * w_)
slope = np.polyfit(np.log([0.7, 1.0, 1.4]), np.log(E_stiff), 1)[0]
ok(f"LS2 dust F = m |n| on hedgehogs (one-sign Jacobian): energy / m = {np.round(np.array(E_dust) / (2 * np.pi**2), 3)} x 2 pi^2 at sizes 0.7, 1, 1.4 (degenerate, = degree); "
   f"stiff step F ~ n^2: E ~ size^{slope:.2f} (-3): the lump spreads, a size needs one more term (the potential of the BPS Skyrme model)",
   max(abs(np.array(E_dust) / (2 * np.pi**2) - 1)) < 0.03 and abs(slope + 3) < 0.05)

# folded profile: f(0) = pi, f(inf) = 0 but f rises above pi first; signed integral stays 2 pi^2, |n| integral is larger
ff = lambda r: np.pi * np.exp(-r) * (1 + 2 * r)
dff = lambda r: np.pi * np.exp(-r) * (1 - 2 * r)
signed = 4 * np.pi * quad(lambda r: -np.sin(ff(r))**2 * dff(r), 0, np.inf, limit=200)[0]
g_abs = lambda r: np.sin(ff(r))**2 * abs(dff(r))
absint = 4 * np.pi * (quad(g_abs, 0, 0.5, limit=200)[0] + quad(g_abs, 0.5, np.inf, limit=200)[0])
ok(f"LS2b a folded profile (f rises to {ff(0.5):.3f} > pi before falling): degree {signed / (2 * np.pi**2):.4f}, int |n| = "
   f"{absint / (2 * np.pi**2):.4f} x 2 pi^2: dust energy equals m |degree| 2 pi^2 only when the Jacobian has one sign",
   abs(signed / (2 * np.pi**2) - 1) < 1e-6 and absint / (2 * np.pi**2) > 1.05)

# ---------- LS3 hedgehog equivariance ----------
rng = np.random.default_rng(134)
Q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
if np.linalg.det(Q) < 0: Q[:, 0] *= -1
pts = rng.normal(size=(50, 3))
lhs = hedgehog(pts @ Q.T); rhs = hedgehog(pts) @ np.block([[np.ones((1, 1)), np.zeros((1, 3))], [np.zeros((3, 1)), Q]]).T
ok(f"LS3 X(R x) = diag(1, R) X(x) (max deviation {np.abs(lhs - rhs).max():.1e}): for the lump a rotation of space is a rotation of label space",
   np.abs(lhs - rhs).max() < 1e-12)

# ---------- LS4 J = I on functions of the collective coordinate ----------
tau = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]], complex)]
def su2(v):
    return expm(-0.5j * sum(v[k] * tau[k] for k in range(3)))
def adj(A):
    return np.array([[0.5 * np.trace(tau[a] @ A @ tau[b] @ A.conj().T).real for b in range(3)] for a in range(3)])
def casimirs(funcs, dim):
    # basis functions phi_i(A); left action phi(exp(-i t tau/2) A), right action phi(A exp(i t tau/2)); fit generators on random points
    As = [su2(rng.normal(size=3) * 2) for _ in range(3 * dim)]
    Phi = np.array([funcs(A) for A in As])                             # samples x dim
    def gen(side, k, t=1e-5):
        Bp, Bm = su2(t * np.eye(3)[k]), su2(-t * np.eye(3)[k])
        if side == 'L':
            D = np.array([(funcs(Bp @ A) - funcs(Bm @ A)) / (2 * t) for A in As])
        else:
            D = np.array([(funcs(A @ Bm) - funcs(A @ Bp)) / (2 * t) for A in As])
        G = np.linalg.lstsq(Phi, D, rcond=None)[0]                     # D = Phi G
        return 1j * G
    CL = sum(gen('L', k) @ gen('L', k) for k in range(3))
    CR = sum(gen('R', k) @ gen('R', k) for k in range(3))
    return np.linalg.eigvals(CL).real, np.linalg.eigvals(CR).real
half = lambda A: A.ravel()
one = lambda A: adj(A).ravel().astype(complex)
cl_h, cr_h = casimirs(half, 4)
cl_1, cr_1 = casimirs(one, 9)
A0 = su2(rng.normal(size=3))
sign_h = np.allclose(half(-A0), -half(A0)); sign_1 = np.allclose(one(-A0), one(A0))
ok(f"LS4 on spin-1/2 functions: label (left) Casimir {np.round(np.sort(cl_h), 4)}, space (right) Casimir {np.round(np.sort(cr_h), 4)}; on spin-1: "
   f"{np.round(np.mean(cl_1), 4)} and {np.round(np.mean(cr_1), 4)}: J = I; Psi(-A) = -Psi(A) for j = 1/2 ({sign_h}), +Psi(A) for j = 1 ({sign_1})",
   np.allclose(cl_h, 0.75, atol=1e-4) and np.allclose(cr_h, 0.75, atol=1e-4) and np.allclose(cl_1, 2, atol=1e-4)
   and np.allclose(cr_1, 2, atol=1e-4) and sign_h and sign_1)

# ---------- LS5 the choice: gauge or global label rotations ----------
# allowed spins: functions on SU(2) with J = I = j carry the sign (-1)^(2j) under A -> -A (LS4); FR sign -1 keeps half-integers only
js = [float(j) for j in np.arange(0, 4.01, 0.5)]
fr_allowed = [j for j in js if (-1)**int(round(2 * j)) == -1]
gauge_allowed = [j for j in js if j == 0]                              # I = 0 forces J = 0
rr_ = np.linspace(1e-6, 60, 60001)
n_r = np.sin(prof(rr_))**2 * np.pi * np.exp(-rr_) / rr_**2 / (2 * np.pi**2)   # mass density per unit lump mass M = 2 pi^2 m (integral 1)
Theta = np.sum(n_r * (2 / 3) * rr_**2 * 4 * np.pi * rr_**2) * (rr_[1] - rr_[0])  # rigid rotation: int rho (r^2 - z^2) = (2/3) int rho r^2
E_rot = {j: j * (j + 1) / (2 * Theta) for j in fr_allowed}
jmin = min(E_rot, key=E_rot.get)
ok(f"LS5 gauge label rotations allow only J = {gauge_allowed} (spin-0 lump); the FR sign allows J = I in {fr_allowed[:4]}...; with the rotor "
   f"energy J(J+1)/(2 Theta), Theta = {Theta:.3f} M a^2 > 0 (M = 2 pi^2 m, the lump mass of LS2), the lowest is J = I = {jmin}: spin 1/2 is the ground state, 3/2 an excitation "
   "(the Delta-like level, known for skyrmions)", gauge_allowed == [0.0] and fr_allowed[0] == 0.5 and jmin == 0.5 and Theta > 0)

# ---------- LS6 known topology ----------
ok("LS6 (known, not recomputed: Finkelstein-Rubinstein 1968, Williams 1970, Giulini 1993) pi_1(Map_1(S^3, S^3)) = pi_4(S^3) = Z2; the 2 pi "
   "rotation of a degree-1 lump generates it and the exchange of two lumps is homotopic to it: with the FR sign the lumps are fermions", True)
