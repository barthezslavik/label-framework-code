# 145: checks behind the statuses of the twelve equivalent interpretations in paper12 (step 5). Each check computes the fact on which a
# status rests; the reading itself stays a hypothesis.
# QI1 de Broglie-Bohm: in one dimension the world lines of a label medium (quantiles of n(t) = |psi(t)|^2, labels conserved) coincide
#     with the Bohm trajectories integrated from the guidance velocity Im(psi'/psi); one trajectory or all of them is the only difference;
# QI2 statistical ensemble: for the dust of the series (K = 0, pressureless) each world line moves freely, x = x0 + v0 t, independent of the
#     others; with the Fisher term the world lines bend (quantum force from the neighbours), so the medium is a literal ensemble of
#     independent particles only for K = 0;
# QI3 Copenhagen, Everett, decoherence: the outcome probabilities of a measurement and of a later measurement are the same whether the
#     state is reduced by the projection rule or entangled unitarily with a pointer and traced out (sequential statistics, qubit);
# QI4 two-state vector: the Aharonov-Bergmann-Lebowitz rule, summed over the outcomes of a later post-selection, gives the Born rule;
# QI5 consistent histories: for two-time histories of a qubit the decoherence functional is diagonal when the initial state lies in the
#     first projection basis, and its diagonal equals the sequential Born probabilities; for a superposed initial state the
#     consistency condition fails, and the formalism assigns no probabilities to that family;
# QI6 QBism / ensemble: the psi of a label medium is normalised to the label count N (here 5.08), not to 1, and its density is a matter
#     density: it is not a quantum state of one system.
# Run: ~/.venvs/jev/bin/python ga_qm_interpretations_check.py
import numpy as np
from scipy.linalg import expm

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

# ---------- QI1 Bohm trajectories = label world lines ----------
L, Ng = 60.0, 4096
xs = np.linspace(-L / 2, L / 2, Ng, endpoint=False); dx = xs[1] - xs[0]
kx = 2 * np.pi * np.fft.fftfreq(Ng, d=dx)
def free(ps, T):
    return np.fft.ifft(np.fft.fft(ps) * np.exp(-1j * kx**2 * T / 2))
psi0 = np.exp(-(xs + 3)**2 / 2 + 0.8j * xs) * (1 + 0.3 * np.exp(-(xs + 2)**2))
psi0 /= np.sqrt(np.sum(np.abs(psi0)**2) * dx)
def vel(ps):
    d = np.fft.ifft(1j * kx * np.fft.fft(ps)); n = np.abs(ps)**2
    return np.where(n > 1e-14 * n.max(), np.imag(np.conj(ps) * d) / np.maximum(n, 1e-300), 0.0)
def quantile(n, q):
    c = (np.cumsum(n) - 0.5 * n) * dx                          # cumulative at the grid points (midpoint rule)
    return np.interp(q, c, xs)
qs = np.linspace(0.1, 0.9, 9)
X = quantile(np.abs(psi0)**2, qs); T1 = 2.0; steps = 2000; dt = T1 / steps; t = 0.0
for _ in range(steps):
    k1 = np.interp(X, xs, vel(free(psi0, t)))
    k2 = np.interp(X + 0.5 * dt * k1, xs, vel(free(psi0, t + 0.5 * dt)))
    k3 = np.interp(X + 0.5 * dt * k2, xs, vel(free(psi0, t + 0.5 * dt)))
    k4 = np.interp(X + dt * k3, xs, vel(free(psi0, t + dt)))
    X = X + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4); t += dt
Xlab = quantile(np.abs(free(psi0, T1))**2, qs)
dev1 = np.max(np.abs(X - Xlab))
ok(f"QI1 Bohm trajectories vs label world lines (quantiles of |psi|^2) at t = {T1}: max deviation {dev1:.1e} over 9 lines", dev1 < 2e-3)

# ---------- QI2 free world lines only for K = 0 ----------
v0 = np.interp(quantile(np.abs(psi0)**2, qs), xs, vel(psi0))
Xdust = quantile(np.abs(psi0)**2, qs) + v0 * T1
bend = np.max(np.abs(Xlab - Xdust)); width0 = np.sqrt(np.sum(np.abs(psi0)**2 * xs**2) * dx - (np.sum(np.abs(psi0)**2 * xs) * dx)**2)
ok(f"QI2 dust (K = 0): world lines x0 + v0 t; with the Fisher term they bend by up to {bend:.3f} at t = {T1} (packet width {width0:.2f}): "
   "a literal ensemble of independent particles only for K = 0", bend > 0.1)

# ---------- QI3 projection vs unitary pointer + trace ----------
rng = np.random.default_rng(145)
def rand_state(d):
    v = rng.standard_normal(d) + 1j * rng.standard_normal(d); return v / np.linalg.norm(v)
def rand_unitary(d):
    H = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d)); H = H + H.conj().T; return expm(-1j * H)
worst3 = 0.0
for _ in range(20):
    s = rand_state(2); Ub = rand_unitary(2)                  # measure Z, evolve by Ub, measure Z again
    P = [np.diag([1, 0]), np.diag([0, 1])]
    proj = np.zeros((2, 2))
    for i in range(2):
        pi = np.vdot(s, P[i] @ s).real
        if pi < 1e-15: continue
        post = Ub @ (P[i] @ s) / np.sqrt(pi)
        for j in range(2):
            proj[i, j] = pi * np.vdot(post, P[j] @ post).real
    # unitary: two pointers record the outcomes (CNOT onto each pointer), then the joint statistics of the pointers
    psi = np.kron(s, np.kron([1, 0], [1, 0])).astype(complex)
    def cnot_to(ptr):
        M = np.zeros((8, 8))
        for b in range(8):
            q, p1, p2 = (b >> 2) & 1, (b >> 1) & 1, b & 1
            if ptr == 1: p1 ^= q
            else: p2 ^= q
            M[(q << 2) | (p1 << 1) | p2, b] = 1
        return M
    psi = cnot_to(1) @ psi
    psi = np.kron(Ub, np.eye(4)) @ psi
    psi = cnot_to(2) @ psi
    joint = np.zeros((2, 2))
    for b in range(8):
        p1, p2 = (b >> 1) & 1, b & 1
        joint[p1, p2] += abs(psi[b])**2
    worst3 = max(worst3, np.max(np.abs(joint - proj)))
ok(f"QI3 sequential statistics, projection rule vs unitary pointers traced out: max difference {worst3:.1e} over 20 random cases",
   worst3 < 1e-12)

# ---------- QI4 ABL summed over post-selection = Born ----------
worst4 = 0.0
for _ in range(20):
    s = rand_state(3); Um = rand_unitary(3); Uf = rand_unitary(3)
    A = [np.outer(Um[:, k], Um[:, k].conj()) for k in range(3)]       # intermediate measurement
    F = [Uf[:, f] for f in range(3)]                                  # final post-selection basis
    born = np.array([np.vdot(s, Ak @ s).real for Ak in A])
    abl_sum = np.zeros(3)
    for f in F:
        amps = np.array([abs(np.vdot(f, Ak @ s))**2 for Ak in A])
        pf = amps.sum()
        if pf < 1e-15: continue
        abl = amps / pf                                                # ABL probability given f
        abl_sum += pf * abl                                            # weighted by the probability of f
    worst4 = max(worst4, np.max(np.abs(abl_sum - born)))
ok(f"QI4 ABL rule summed over post-selection vs Born: max difference {worst4:.1e} over 20 random cases", worst4 < 1e-12)

# ---------- QI5 decoherence functional ----------
def dfun(s, U, P):
    D = np.zeros((4, 4), dtype=complex); hs = [(i, j) for i in range(2) for j in range(2)]
    C = {h: P[h[1]] @ U @ P[h[0]] for h in hs}
    for a, h in enumerate(hs):
        for b, g in enumerate(hs):
            D[a, b] = np.vdot(C[g] @ s, C[h] @ s)
    return D
s = rand_state(2); U = rand_unitary(2)
Pz = [np.diag([1, 0]).astype(complex), np.diag([0, 1]).astype(complex)]
D = dfun(s, U, Pz)
offd_z = np.max(np.abs(D - np.diag(np.diag(D))))
# consistent family: histories whose first projection commutes with the state's evolution record (state diagonal in the first basis)
s_diag = np.array([1, 0], dtype=complex); Dd = dfun(s_diag, U, Pz)
offd_c = np.max(np.abs(Dd - np.diag(np.diag(Dd))))
diag_born = np.array([abs(np.vdot(s_diag, Pz[i] @ s_diag))**2 * abs(np.vdot(Pz[j] @ U @ Pz[i] @ s_diag, Pz[j] @ U @ Pz[i] @ s_diag)) for i in range(2) for j in range(2)])
ok(f"QI5 decoherence functional: interfering family off-diagonal {offd_z:.3f} (inconsistent); consistent family off-diagonal {offd_c:.1e}, "
   f"diagonal = sequential Born probabilities to {np.max(np.abs(np.real(np.diag(Dd)) - diag_born)):.1e}",
   offd_z > 1e-3 and offd_c < 1e-12 and np.max(np.abs(np.real(np.diag(Dd)) - diag_born)) < 1e-12)

# ---------- QI6 label-medium psi is normalised to the label count ----------
nlab = np.exp(-(xs + 5)**2 / 4) + 0.5 * np.exp(-(xs - 6)**2 / 3)
Nlab = np.sum(nlab) * dx
ok(f"QI6 the psi of a label medium has norm = label count N = {Nlab:.2f}, not 1: a matter field, not the state of one system", abs(Nlab - 1) > 1)
print("ALL PASS")
