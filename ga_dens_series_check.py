# 149: QM part II, step 5 (link with the series) and step 6 (statistics). Two particles on a line, hbar = m = 1.
# DL1 the label medium of the series in space sees only the one-body density n(x) = rho_1(x) + rho_2(x): a product density and a
#     correlated one with the same marginals give the same n and the same medium Fisher term, but different configuration-space
#     Fisher information (the gap of DD3); the medium is the marginal level of the configuration-space picture (QM8 of paper12);
# DL2 Madelung metric: for psi = sqrt(rho) e^{iS}, ||delta psi||^2 = (1/4) FR(delta rho) + int rho (delta S)^2 (Sasaki-Fisher-Rao,
#     Khesin-Misiolek-Modin); linear evolution keeps the distance of two nearby states (unitarity = isometry), the evolution with
#     marginal terms (c = 0.3) does not;
# DL3 hbar as the coefficient between the two geometries: (1/2) int rho |grad S|^2 + (hbar^2/8) I(rho) = (hbar^2/2) ||grad psi||^2 with
#     psi = sqrt(rho) e^{iS/hbar}, for hbar = 0.7, 1, 1.6 (kinetic Wasserstein term plus Fisher term is the kinetic energy of psi);
# ST1 bosons and fermions from the same two orbitals: both densities are S_2-symmetric, and the full and marginal Fisher-Rao metrics
#     are S_2-invariant on both; the group and the metric do not distinguish the statistics;
# ST2 the fermion density vanishes quadratically on the diagonal x1 = x2, the boson density does not: statistics shows in Dens only as
#     a nodal condition on a particular state;
# ST3 Girardeau mapping in one dimension: psi_B = |psi_F| gives a hard-core boson state with exactly the fermion density, and the free
#     evolution keeps |psi_F(t)|^2 = |psi_B(t)|^2 when psi_B(t) = sgn(x1 - x2) psi_F(t): Dens cannot see the sign of S_2;
# ST4 in three dimensions the exchange loop in the relative coordinate (rotation of r by pi about an axis orthogonal to r) gives the
#     phase factor (-1)^l on a state of angular momentum l: exchange phase +-1, the two characters of pi_1 = Z_2 of the configuration
#     space of two identical particles (Leinaas-Myrheim); a global phase condition of the same class as Wallstrom's.
# Run: ~/.venvs/jev/bin/python ga_dens_series_check.py
import numpy as np

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

n, Lb = 192, 24.0
x = (np.arange(n) - n / 2) * (Lb / n); dx = x[1] - x[0]
X1, X2 = np.meshgrid(x, x, indexing='ij')
k = 2 * np.pi * np.fft.fftfreq(n, d=dx); K1, K2 = np.meshgrid(k, k, indexing='ij')
def dd(f, ax):
    K = K1 if ax == 0 else K2
    return np.fft.ifft2(1j * K * np.fft.fft2(f))
def d1(f):
    return np.real(np.fft.ifft(1j * k * np.fft.fft(f)))
def Ifis1(m):
    w = m > 1e-13 * m.max()
    return ((d1(m)**2 / np.where(w, m, 1.0)) * w).sum() * dx
def Ifis2(r):
    w = r > 1e-13 * r.max()
    g = np.real(dd(r, 0))**2 + np.real(dd(r, 1))**2
    return ((g / np.where(w, r, 1.0)) * w).sum() * dx * dx
def marg(r, ax):
    return r.sum(axis=1 - ax) * dx
def gauss2(S):
    Si = np.linalg.inv(S)
    r = np.exp(-0.5 * (Si[0, 0] * X1**2 + 2 * Si[0, 1] * X1 * X2 + Si[1, 1] * X2**2))
    return r / (r.sum() * dx * dx)

# ---------- DL1 ----------
rc = gauss2(np.array([[1.0, 0.7], [0.7, 1.0]])); rp = gauss2(np.eye(2))
n_c = marg(rc, 0) + marg(rc, 1); n_p = marg(rp, 0) + marg(rp, 1)
ok(f"DL1 same one-body density (max diff {np.abs(n_c - n_p).max():.1e}) and medium Fisher term ({Ifis1(n_c):.6f} vs {Ifis1(n_p):.6f}), "
   f"but configuration-space Fisher information {Ifis2(rc):.4f} (correlated) vs {Ifis2(rp):.4f} (product): the medium in space is the "
   f"marginal level", np.abs(n_c - n_p).max() < 1e-10 and abs(Ifis1(n_c) / Ifis1(n_p) - 1) < 1e-9 and Ifis2(rc) - Ifis2(rp) > 1.0)

# ---------- DL2 ----------
def norm(p):
    return p / np.sqrt((np.abs(p)**2).sum() * dx * dx)
rho0 = gauss2(np.array([[1.3, 0.4], [0.4, 0.8]])); S0 = 0.5 * X1 - 0.3 * X2 + 0.1 * X1 * X2
drho = rho0 * (np.sin(X1) * np.exp(-X2**2 / 4) - (rho0 * np.sin(X1) * np.exp(-X2**2 / 4)).sum() * dx * dx)
dS = 0.3 * np.cos(X1 + X2) * np.exp(-(X1**2 + X2**2) / 8)
eps = 1e-4
psiA = np.sqrt(rho0) * np.exp(1j * S0); psiB = np.sqrt(rho0 + eps * drho) * np.exp(1j * (S0 + eps * dS))
dpsi2 = (np.abs(psiB - psiA)**2).sum() * dx * dx / eps**2
w = rho0 > 1e-13 * rho0.max()
sasaki = 0.25 * ((drho**2 / np.where(w, rho0, 1.0)) * w).sum() * dx * dx + (rho0 * dS**2).sum() * dx * dx
def Qb(m):
    sm = np.sqrt(np.maximum(m, 1e-300)); ww = m > 1e-12 * m.max()
    return np.where(ww, 0.5 * np.real(np.fft.ifft(k**2 * np.fft.fft(sm))) / np.where(ww, sm, 1.0), 0.0)
def evolve(psi, c, T=1.0, steps=800):
    dt = T / steps; kin = np.exp(-1j * (K1**2 + K2**2) * dt / 4)
    for _ in range(steps):
        psi = np.fft.ifft2(kin * np.fft.fft2(psi))
        if c:
            rr = np.abs(psi)**2
            psi = psi * np.exp(-1j * c * (Qb(marg(rr, 0))[:, None] + Qb(marg(rr, 1))[None, :]) * dt)
        psi = np.fft.ifft2(kin * np.fft.fft2(psi))
    return psi
dist = lambda a, b: np.sqrt((np.abs(a - b)**2).sum() * dx * dx)
pA, pB = norm(psiA), norm(psiA + 1e-3 * norm(np.exp(-((X1 - 1)**2 + X2**2)) * np.exp(0.4j * X2)))
d0 = dist(pA, pB); dl = dist(evolve(pA, 0.0), evolve(pB, 0.0)); dn = dist(evolve(pA, 0.3), evolve(pB, 0.3))
ok(f"DL2 ||delta psi||^2 = {dpsi2:.6f}, (1/4) FR + int rho dS^2 = {sasaki:.6f}; distance of two nearby states {d0:.6e} -> linear "
   f"{dl:.6e}, with marginal terms {dn:.6e}: unitarity is the isometry of the Sasaki-Fisher-Rao metric, lost when c != 0",
   abs(dpsi2 / sasaki - 1) < 1e-3 and abs(dl / d0 - 1) < 1e-10 and abs(dn / d0 - 1) > 1e-3)

# ---------- DL3 ----------
res = []
for hb in (0.7, 1.0, 1.6):
    psi = np.sqrt(rho0) * np.exp(1j * S0 / hb)
    gS1, gS2 = 0.5 + 0.1 * X2, -0.3 + 0.1 * X1                       # analytic gradient (S0 is not periodic)
    lhs = 0.5 * (rho0 * (gS1**2 + gS2**2)).sum() * dx * dx + hb**2 / 8 * Ifis2(rho0)
    rhs = hb**2 / 2 * ((np.abs(dd(psi, 0))**2 + np.abs(dd(psi, 1))**2)).sum() * dx * dx
    res.append((hb, lhs, rhs))
ok("DL3 Wasserstein kinetic term + (hbar^2/8) Fisher information = (hbar^2/2)||grad psi||^2: " + ", ".join(
   f"hbar={h}: {a:.6f} = {b:.6f}" for h, a, b in res) + "; hbar is the coefficient between the two geometries",
   all(abs(a / b - 1) < 1e-6 for h, a, b in res))

# ---------- ST1, ST2 ----------
pa = np.exp(-(x - 1.0)**2 / 2); pb = (x + 0.5) * np.exp(-(x + 0.5)**2 / 3)
pa /= np.sqrt((pa**2).sum() * dx); pb -= (pa * pb).sum() * dx * pa; pb /= np.sqrt((pb**2).sum() * dx)
psB = norm(pa[:, None] * pb[None, :] + pb[:, None] * pa[None, :]); psF = norm(pa[:, None] * pb[None, :] - pb[:, None] * pa[None, :])
rB, rF = np.abs(psB)**2, np.abs(psF)**2
sym = max(np.abs(rB - rB.T).max(), np.abs(rF - rF.T).max())
frB = (Ifis2(rB), Ifis2(rB.T)); frF = (Ifis2(rF), Ifis2(rF.T))
ok(f"ST1 boson and fermion densities are S_2-symmetric (max asymmetry {sym:.1e}); Fisher information invariant under the swap for both "
   f"({frB[0]:.5f} = {frB[1]:.5f}, {frF[0]:.5f} = {frF[1]:.5f}): group and metric do not distinguish the statistics",
   sym < 1e-12 and abs(frB[0] - frB[1]) < 1e-9 and abs(frF[0] - frF[1]) < 1e-9)
diagF = np.abs(np.diag(rF)).max() / rF.max(); diagB = np.diag(rB).max() / rB.max()
i0 = 3 + int(np.argmax([rF[i + 1, i - 1] for i in range(3, n - 3)]))    # diagonal point where the node is steepest
quad = [rF[i0 + h, i0 - h] / (2 * h * dx)**2 for h in (1, 2, 3)]         # rho_F / (x1 - x2)^2 near the diagonal
spread = (max(quad) - min(quad)) / max(quad)
ok(f"ST2 fermion density on the diagonal {diagF:.1e} of its maximum, boson {diagB:.2f}; near the diagonal rho_F / (x1 - x2)^2 = "
   f"{', '.join(f'{q:.4f}' for q in quad)} (spread {spread:.1e}): quadratic node; statistics shows in Dens only as a nodal condition",
   diagF < 1e-12 and diagB > 0.05 and quad[0] > 1e-3 and spread < 0.1)

# ---------- ST3 Girardeau ----------
sgn = np.sign(X1 - X2)
psiBt = np.abs(psF) * np.exp(1j * 0)                                    # hard-core boson state with the fermion density
same0 = np.abs(np.abs(psiBt)**2 - rF).max()
Ft = evolve(psF, 0.0, T=0.8)
Bt = sgn * Ft                                                           # Girardeau map of the evolved state
symB = np.abs(Bt - Bt.T).max() / np.abs(Bt).max()
sameT = np.abs(np.abs(Bt)**2 - np.abs(Ft)**2).max()
ok(f"ST3 Girardeau: |psi_B|^2 = |psi_F|^2 (diff {same0:.1e}); sgn(x1-x2) psi_F(t) is symmetric (asym {symB:.1e}) with the same density "
   f"(diff {sameT:.1e}): Dens cannot see the sign of S_2 in one dimension", same0 < 1e-14 and symB < 1e-8 and sameT < 1e-14)

# ---------- ST4 exchange phase in three dimensions ----------
def Ylm_like(l, v):                                                     # (x + i y)^l, angular momentum l about z
    return (v[0] + 1j * v[1])**l
r0 = np.array([0.8, 0.3, 0.2]); r0[2] = 0.0
phases = []
for l in range(5):
    path = [np.array([np.cos(t) * r0[0] - np.sin(t) * r0[1], np.sin(t) * r0[0] + np.cos(t) * r0[1], 0.0]) for t in np.linspace(0, np.pi, 400)]
    ph = np.unwrap(np.angle([Ylm_like(l, p) for p in path]))
    phases.append(np.cos(ph[-1] - ph[0]))
ok("ST4 exchange loop (rotation of the relative coordinate by pi): phase factor " + ", ".join(f"l={l}: {p:+.0f}" for l, p in enumerate(phases))
   + " = (-1)^l; the exchange phase is +-1, the characters of pi_1 = Z_2 (Leinaas-Myrheim): a global phase condition like Wallstrom's",
   all(abs(p - (-1)**l) < 1e-9 for l, p in enumerate(phases)))

print("ALL PASS")
