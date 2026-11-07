# 145: the five testable modifications of quantum mechanics and half-integer spin in the label framework (steps 4 and 6 of paper12).
# Numbers borrowed: Donadi et al., Nat. Phys. 17, 74 (2021), arXiv:2111.13490: R0 > 0.54e-10 m (probability 0.95), signal a/R0^3 with
# a = 1.756e-29 m^3; the area coefficient 0.02351/a^2 per field of the theorem base; every step recomputed here.
# QD1 Diosi-Penrose with a length that appears in the series: the step a = sqrt(4 N eta) l_P of the lattice regulator of the entanglement
#     entropy (G hbar = a^2/(4 N eta), N = 1..1000 minimally coupled fields, theorem base) and the nucleon Compton length (set by a mass,
#     a label of the series) lie below the Gran Sasso bound, so the predicted signal exceeds the data by about 1e16 (Compton) to 1e75 (step); the
#     cosmological lengths (Lambda^-1/2, c/H0, c^2/a0) give a collapse time for a 1 kg mass split by 1 m longer than 1e50 s:
#     with these lengths the model is either excluded or gives no collapse;
# QD2 GRW/CSL: the label dynamics is deterministic and time-reversible (forward and backward evolution return the state), while a
#     localisation noise of rate lambda destroys the coherence of a split packet irreversibly: pure states go to mixed ones, which no
#     flow at level 3 of the ladder of preserved structures (symplectic form and metric kept, theorem base) does; no object of the
#     framework supplies lambda or r_C;
# QD3 Valentini non-equilibrium: a density guided by the velocity of psi (Bohm guidance) and a second label medium with the same
#     initial density and phase evolve differently; the label medium has no guidance law, its density is |psi|^2 identically, and a
#     distribution rho != |psi|^2 has no carrier among its objects; sampled from |psi|^2 the guided density stays |psi|^2 (equivariance);
# QD4 superdeterminism: local deterministic label values with setting-independent distribution reach |CHSH| <= 2 (all 16 strategies),
#     the singlet gives 2 sqrt 2: classical labels can carry Bell correlations only with setting dependence or non-locality; the
#     quantised label field gives 2 sqrt 2 without hidden variables;
# QD5 nonlinearity: the term g|psi|^2 psi of a medium with U = g n^2/2 (pressure g n^2/2) breaks superposition by an amount growing
#     with g and vanishing for the pressureless medium (g = 0); the flow with g != 0 changes the overlap |<u1|u2>| of two states, so it
#     is not an isometry and is not a quantum evolution at level 3; for the classical field of a label medium it is the pressure;
# HH1 spin: for the hedgehog U = exp(i f(r) xhat.sigma) on label space S^3 a spatial rotation by alpha equals the isorotation by
#     A = exp(-i alpha sigma_z/2); at alpha = 2 pi the rotation matrix is 1 and A = -1: the 2 pi rotation is a closed loop of
#     configurations, the Finkelstein-Rubinstein loop; the sign assigned to it is a choice, so half-integer spin follows only
#     under the sign premise.
# Run: ~/.venvs/jev/bin/python ga_qm_modifications_check.py
import itertools
import numpy as np
from scipy.special import erf
from scipy.linalg import expm

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

G, hbar, c = 6.674e-11, 1.0546e-34, 2.998e8
lP = np.sqrt(hbar * G / c**3)
R0_bound, a_sig = 0.54e-10, 1.756e-29

# ---------- QD1 Diosi-Penrose ----------
eta = 0.02351
steps = {N: np.sqrt(4 * N * eta) * lP for N in (1, 10, 100, 1000)}
compton = hbar / (1.6726e-27 * c)
micro = dict(steps); micro['Compton(p)'] = compton
excess = {k: (R0_bound / v)**3 for k, v in micro.items()}
Lam, H0, a0 = 1.1e-52, 2.2e-18, 1.2e-10
cosmo = {'Lambda^-1/2': Lam**-0.5, 'c/H0': c / H0, 'c^2/a0': c**2 / a0}
def tau_dp(mass, d, R0):
    I0 = G * mass**2 / (np.sqrt(np.pi) * R0)                     # mutual energy of two coincident Gaussians of width R0
    Id = G * mass**2 * erf(d / (2 * R0)) / d                      # at separation d
    dE = I0 - Id
    if dE <= 0 or dE / I0 < 1e-12:                                # erf cancellation: use the small-d expansion
        dE = G * mass**2 * d**2 / (12 * np.sqrt(np.pi) * R0**3)
    return hbar / dE
taus = {k: tau_dp(1.0, 1.0, v) for k, v in cosmo.items()}
age = 4.35e17
ok(f"QD1 lengths in the series (m): regulator step a = {', '.join(f'{v:.1e}' for v in steps.values())} for N = 1, 10, 100, 1000 fields, Compton "
   f"{compton:.1e}; all below R0 > {R0_bound:.2e}: predicted Gran Sasso signal exceeds the data by {min(excess.values()):.0e} (Compton) to {max(excess.values()):.0e} (step); "
   f"cosmological lengths give tau(1 kg, 1 m) = {', '.join(f'{v:.0e}' for v in taus.values())} s >> age {age:.1e} s: excluded or no collapse",
   all(v < R0_bound for v in micro.values()) and min(excess.values()) > 1e15 and min(taus.values()) > 1e50)

# ---------- QD2 reversibility vs localisation noise ----------
L, Ng = 80.0, 2048
xs = np.linspace(-L / 2, L / 2, Ng, endpoint=False); dx = xs[1] - xs[0]
kx = 2 * np.pi * np.fft.fftfreq(Ng, d=dx)
def free(ps, T):
    return np.fft.ifft(np.fft.fft(ps) * np.exp(-1j * kx**2 * T / 2))
ps0 = np.exp(-(xs + 6)**2 / 2 + 1j * xs) + np.exp(-(xs - 6)**2 / 2 - 1j * xs)
ps0 = ps0 / np.sqrt(np.sum(np.abs(ps0)**2) * dx)
back = free(free(ps0, 9.0), -9.0)
rev = np.max(np.abs(back - ps0))
# localisation noise (GRW/CSL master equation in position basis): rho(x,x') -> rho exp(-lam t (1 - exp(-(x-x')^2/(4 rC^2))))
lam, rC, Tn = 0.5, 1.0, 4.0
rho = np.outer(ps0, np.conj(ps0))
D = np.subtract.outer(xs, xs)
rho_t = rho * np.exp(-lam * Tn * (1 - np.exp(-D**2 / (4 * rC**2))))
pur0 = np.real(np.trace(rho @ rho)) * dx * dx
pur1 = np.real(np.trace(rho_t @ rho_t)) * dx * dx
ok(f"QD2 label dynamics forward 9 and back: deviation {rev:.1e} (reversible, pure); localisation noise lambda = {lam}, r_C = {rC} for t = {Tn}: "
   f"purity {pur0:.4f} -> {pur1:.4f} (irreversible, pure -> mixed: not a flow at level 3). The noise and its two parameters are not objects of the framework",
   rev < 1e-10 and abs(pur0 - 1) < 1e-6 and pur1 < 0.7)

# ---------- QD3 guidance vs a second medium ----------
psi_s = np.exp(-(xs + 2)**2 / 4 + 0.7j * xs); psi_s /= np.sqrt(np.sum(np.abs(psi_s)**2) * dx)
rho_ne = np.exp(-(xs + 0.5)**2 / 1.0); rho_ne /= np.sum(rho_ne) * dx      # rho != |psi|^2
def velocity(ps):
    dps = np.fft.ifft(1j * kx * np.fft.fft(ps))
    n = np.abs(ps)**2
    v = np.imag(np.conj(ps) * dps) / np.maximum(n, 1e-300)
    return np.where(n > 1e-20 * n.max(), v, 0.0)
def guided_moments(rho0, ps_init, T, steps_=800):
    keep = rho0 > 1e-12 * rho0.max()                              # trajectories that carry weight
    w = rho0[keep] * dx; w = w / w.sum(); X = xs[keep].copy(); dt = T / steps_; tt = 0.0
    for _ in range(steps_):
        v1 = np.interp(X, xs, velocity(free(ps_init, tt)))
        Xm = X + 0.5 * dt * v1
        v2 = np.interp(Xm, xs, velocity(free(ps_init, tt + 0.5 * dt)))
        X = X + dt * v2; tt += dt
    mean = np.sum(w * X); var = np.sum(w * (X - mean)**2)
    return mean, np.sqrt(var)
T3 = 3.0
mg, sg = guided_moments(rho_ne, psi_s, T3)
psi2 = np.sqrt(rho_ne) * np.exp(1j * np.angle(psi_s))                   # second medium: same phase, its own density
n2 = np.abs(free(psi2, T3))**2; m2 = np.sum(n2 * xs) * dx; s2 = np.sqrt(np.sum(n2 * (xs - m2)**2) * dx)
me, se = guided_moments(np.abs(psi_s)**2, psi_s, T3)
nq = np.abs(free(psi_s, T3))**2; mq = np.sum(nq * xs) * dx; sq = np.sqrt(np.sum(nq * (xs - mq)**2) * dx)
ok(f"QD3 at t = {T3}: guided non-equilibrium density mean {mg:.3f}, width {sg:.3f}; second label medium mean {m2:.3f}, width {s2:.3f} "
   f"(differ); guided |psi|^2 mean {me:.4f}, width {se:.4f} vs |psi(t)|^2 {mq:.4f}, {sq:.4f} (equivariance). Guidance is not a law "
   "of the label medium, whose density is |psi|^2 identically",
   abs(sg - s2) > 0.1 and abs(me - mq) < 2e-3 and abs(se - sq) < 2e-3)

# ---------- QD4 CHSH ----------
best = 0
for A in itertools.product([1, -1], repeat=2):
    for B in itertools.product([1, -1], repeat=2):
        Sv = A[0] * B[0] + A[0] * B[1] + A[1] * B[0] - A[1] * B[1]
        best = max(best, abs(Sv))
sx = np.array([[0, 1], [1, 0]]); sz = np.array([[1, 0], [0, -1]])
op = lambda th: np.cos(th) * sz + np.sin(th) * sx
singlet = np.array([0, 1, -1, 0]) / np.sqrt(2)
E = lambda a, b: singlet @ np.kron(op(a), op(b)) @ singlet
a_, a2, b_, b2 = 0, np.pi / 2, np.pi / 4, -np.pi / 4
Sq = E(a_, b_) + E(a_, b2) + E(a2, b_) - E(a2, b2)
ok(f"QD4 local deterministic label values, setting-independent: max |CHSH| = {best}; singlet |CHSH| = {abs(Sq):.6f} = 2 sqrt 2: "
   "Bell correlations need setting dependence (superdeterminism), non-locality, or the quantised field", best == 2 and abs(abs(Sq) - 2 * np.sqrt(2)) < 1e-12)

# ---------- QD5 nonlinearity = pressure ----------
def nls(ps, g, T, steps_=400):
    dt = T / steps_
    half = np.exp(-1j * kx**2 * dt / 4)
    for _ in range(steps_):
        ps = np.fft.ifft(np.fft.fft(ps) * half)
        ps = ps * np.exp(-1j * g * np.abs(ps)**2 * dt)
        ps = np.fft.ifft(np.fft.fft(ps) * half)
    return ps
u1 = np.exp(-(xs + 3)**2 / 2 + 0.5j * xs); u2 = np.exp(-(xs - 3)**2 / 2 - 0.5j * xs)
brk = {}; ovl = {}
w1 = u1 / np.sqrt(np.sum(np.abs(u1)**2) * dx); w2 = np.exp(-(xs + 2)**2 / 3 + 0.2j * xs); w2 = w2 / np.sqrt(np.sum(np.abs(w2)**2) * dx)
o0 = abs(np.sum(np.conj(w1) * w2) * dx)
for g in (0.0, 0.5, 2.0):
    brk[g] = np.max(np.abs(nls(u1 + u2, g, 3.0) - nls(u1, g, 3.0) - nls(u2, g, 3.0)))
    ovl[g] = abs(abs(np.sum(np.conj(nls(w1, g, 3.0)) * nls(w2, g, 3.0)) * dx) - o0)
ok(f"QD5 superposition defect after t = 3: g = 0: {brk[0.0]:.1e}, g = 0.5: {brk[0.5]:.3f}, g = 2: {brk[2.0]:.3f}; change of the overlap "
   f"|<u1|u2>| = {o0:.3f}: {ovl[0.0]:.1e}, {ovl[0.5]:.3f}, {ovl[2.0]:.3f} (not an isometry for g != 0, excluded at level 3); g is the pressure "
   "coefficient p = g n^2/2 of the classical label medium, whose linear point is dust",
   brk[0.0] < 1e-10 and brk[0.5] > 1e-2 and brk[2.0] > brk[0.5] and ovl[0.0] < 1e-12 and ovl[2.0] > 1e-3)

# ---------- HH1 hedgehog: rotation = isorotation, 2 pi ----------
sy = np.array([[0, -1j], [1j, 0]]); sxs = sx.astype(complex); szs = sz.astype(complex)
def hedgehog(p):
    r = np.linalg.norm(p); f = np.pi * np.exp(-r)
    nh = p / r
    return np.cos(f) * np.eye(2) + 1j * np.sin(f) * (nh[0] * sxs + nh[1] * sy + nh[2] * szs)
def Rz(al):
    return np.array([[np.cos(al), -np.sin(al), 0], [np.sin(al), np.cos(al), 0], [0, 0, 1]])
rng = np.random.default_rng(145)
worst = 0.0
for al in (0.4, 1.7, np.pi, 2 * np.pi):
    A = expm(-1j * al * szs / 2)
    for _ in range(20):
        p = rng.standard_normal(3)
        lhs = hedgehog(Rz(al).T @ p)            # rotated configuration U(R^-1 x)
        rhs = A.conj().T @ hedgehog(p) @ A
        worst = max(worst, np.max(np.abs(lhs - rhs)))
A2pi = expm(-1j * 2 * np.pi * szs / 2)
ok(f"HH1 hedgehog: rotation by alpha = isorotation by exp(-i alpha sigma_z/2), max deviation {worst:.1e}; at 2 pi: rotation matrix "
   f"- 1 = {np.max(np.abs(Rz(2 * np.pi) - np.eye(3))):.1e}, A = {np.real(A2pi[0, 0]):.0f}: a closed loop of configurations whose sign is "
   "chosen (Finkelstein-Rubinstein): half-integer spin under the sign premise, the sign not derived",
   worst < 1e-12 and np.max(np.abs(Rz(2 * np.pi) - np.eye(3))) < 1e-12 and np.allclose(A2pi, -np.eye(2)))
print("ALL PASS")
