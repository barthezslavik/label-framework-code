# 149: QM part II (paper12, new sections), step 3. What the extra invariant metrics of ga_dens_metric_check (DM4) do to the dynamics.
# The quantum term of the action is the metric evaluated on the translation generators, (hbar^2/8) sum_mu G(d_mu rho, d_mu rho) [von
# Renesse 2012]. For G = FR(rho) + c sum_i FR(rho_i) (marginal family, N = 2) this is (hbar^2/8) [I(rho) + c sum_i I(rho_i)], with I the
# Fisher information. Units hbar = m = 1, one coordinate per particle, two particles on a line.
# DD1 sum_mu FR_marg(d_mu rho, d_mu rho) = I(rho_1) + I(rho_2): the marginal metric gives the Fisher information of the marginals;
# DD2 dI(rho_1)/d rho at (x1, x2) = -4 Lap sqrt(rho_1) / sqrt(rho_1) at x1 (finite differences): the extra term is the Bohm potential of
#     the marginal, c Q[rho_1](x1) + c Q[rho_2](x2), in the Hamilton-Jacobi equation; in psi form a potential that depends on |psi|^2;
# DD3 superadditivity of Fisher information (Carlen 1991): I(rho) = I(rho_1) + I(rho_2) for a product density and I(rho) > I(rho_1) +
#     I(rho_2) for a correlated one (Gaussian with correlation r: tr Sigma^-1 against 1/S11 + 1/S22);
# DD4 with c != 0 the evolution is non-linear: evolve psi_a, psi_b and psi_a + psi_b; superposition fails by a relative amount that
#     grows with c and vanishes at c = 0;
# DD5 for a product state the modified evolution equals linear Schrodinger evolution with hbar_eff = hbar sqrt(1 + c) (densities agree):
#     without entanglement the extra coefficient is indistinguishable from a change of hbar, a label; it shows only for entangled states.
# Run: ~/.venvs/jev/bin/python ga_dens_dynamics_check.py
import numpy as np

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

n, Lb = 128, 24.0
x = (np.arange(n) - n / 2) * (Lb / n); dx = x[1] - x[0]
X1, X2 = np.meshgrid(x, x, indexing='ij')
k = 2 * np.pi * np.fft.fftfreq(n, d=dx); K1, K2 = np.meshgrid(k, k, indexing='ij')

def dd(f, ax):
    K = K1 if ax == 0 else K2
    return np.real(np.fft.ifft2(1j * K * np.fft.fft2(f)))
def d1(f):
    return np.real(np.fft.ifft(1j * k * np.fft.fft(f)))
def lap1(f):
    return np.real(np.fft.ifft(-k**2 * np.fft.fft(f)))
def I2(r):
    w = r > 1e-13 * r.max()                                            # drop the far tails, where spectral noise exceeds the density
    return (((dd(r, 0)**2 + dd(r, 1)**2) / np.where(w, r, 1.0)) * w).sum() * dx * dx
def I1(m):
    w = m > 1e-13 * m.max()
    return ((d1(m)**2 / np.where(w, m, 1.0)) * w).sum() * dx
def marg(r, ax):
    return r.sum(axis=1 - ax) * dx

def gauss2(S):
    Si = np.linalg.inv(S)
    r = np.exp(-0.5 * (Si[0, 0] * X1**2 + 2 * Si[0, 1] * X1 * X2 + Si[1, 1] * X2**2))
    return r / (r.sum() * dx * dx)

# ---------- DD1 ----------
r = gauss2(np.array([[1.2, 0.5], [0.5, 0.9]])) * (1 + 0.2 * np.cos(X1) * np.exp(-X2**2 / 4)); r /= r.sum() * dx * dx
lhs = 0.0
for ax in (0, 1):
    v = dd(r, ax)
    for j in (0, 1):
        mj, vj = marg(r, j), marg(v, j)
        w = mj > 1e-13 * mj.max()
        lhs += ((vj**2 / np.where(w, mj, 1.0)) * w).sum() * dx
rhs = I1(marg(r, 0)) + I1(marg(r, 1))
ok(f"DD1 sum_mu FR_marg(d_mu rho, d_mu rho) = {lhs:.8f}, I(rho_1) + I(rho_2) = {rhs:.8f}", abs(lhs / rhs - 1) < 1e-8)

# ---------- DD2 ----------
m1 = marg(r, 0)
Qm = -4 * lap1(np.sqrt(m1)) / np.sqrt(m1)                              # predicted functional derivative, a function of x1
eta = np.exp(-(X1 - 0.5)**2 - (X2 + 0.3)**2 / 2) * np.sin(X1 + X2)
eps = 1e-5
fd = (I1(marg(r + eps * eta, 0)) - I1(marg(r - eps * eta, 0))) / (2 * eps)
pred = (eta * Qm[:, None]).sum() * dx * dx
ok(f"DD2 dI(rho_1)/drho by finite differences {fd:.8f}, by -4 Lap sqrt(rho_1)/sqrt(rho_1) at x1 {pred:.8f}: the extra term is the "
   f"Bohm potential of the marginal", abs(fd / pred - 1) < 1e-6)

# ---------- DD3 superadditivity ----------
gaps = []
for rr in (0.0, 0.3, 0.6, 0.9):
    S = np.array([[1.0, rr], [rr, 1.0]])
    rg = gauss2(S)
    gaps.append((rr, I2(rg) - I1(marg(rg, 0)) - I1(marg(rg, 1)), np.trace(np.linalg.inv(S)) - 2.0))
ok("DD3 I(rho) - I(rho_1) - I(rho_2) for correlation r: " + ", ".join(f"r={a}: {g:.5f} (exact {e:.5f})" for a, g, e in gaps)
   + "; zero for a product density, positive when correlated (Carlen 1991)",
   abs(gaps[0][1]) < 1e-8 and all(g > 1e-3 and abs(g - e) < 1e-4 for a, g, e in gaps[1:]))

# ---------- DD4 / DD5 evolution ----------
def Qb(m):                                                              # Bohm potential of a one-particle density, hbar = m = 1
    sm = np.sqrt(np.maximum(m, 1e-300))
    return -0.5 * lap1(sm) / np.maximum(sm, 1e-150)
def evolve(psi, c, T=1.0, steps=800, hbar=1.0, V=None):
    dt = T / steps
    kin = np.exp(-1j * hbar * (K1**2 + K2**2) * dt / 4)
    for _ in range(steps):
        psi = np.fft.ifft2(kin * np.fft.fft2(psi))
        if c != 0.0:
            rr = np.abs(psi)**2
            Vnl = c * (Qb(marg(rr, 0))[:, None] + Qb(marg(rr, 1))[None, :])
            psi = psi * np.exp(-1j * Vnl * dt / hbar)
        if V is not None:
            psi = psi * np.exp(-1j * V * dt / hbar)
        psi = np.fft.ifft2(kin * np.fft.fft2(psi))
    return psi
def norm(p):
    return p / np.sqrt((np.abs(p)**2).sum() * dx * dx)
pa = norm(np.exp(-((X1 - 2)**2 + (X2 + 1)**2) / 2 + 0.5j * X1) * (1 + 0.5 * X1 * X2 / (1 + X1**2)))
pb = norm(np.exp(-((X1 + 2)**2 + (X2 - 1)**2) / 2 - 0.4j * X2) * (1 + 0.3 * np.tanh(X1 - X2)))
res = []
for c in (0.0, 0.1, 0.3):
    ea, eb, es = evolve(pa, c), evolve(pb, c), evolve(pa + pb, c)
    res.append((c, np.abs(es - ea - eb).max() / np.abs(es).max()))
ok("DD4 superposition defect |U(a+b) - U(a) - U(b)| / |U(a+b)|: " + ", ".join(f"c={c}: {e:.1e}" for c, e in res)
   + "; zero only at c = 0: the marginal terms make quantum evolution non-linear",
   res[0][1] < 1e-10 and res[1][1] > 1e-3 and res[2][1] > res[1][1])

c = 0.3; hb = np.sqrt(1 + c)
f1 = np.exp(-(x - 1)**2 / 1.5 + 0.7j * x); f2 = np.exp(-(x + 0.5)**2 / 0.8 - 0.3j * x)
prod = norm(f1[:, None] * f2[None, :])
# same initial density and current: psi_eff = sqrt(rho) e^{iS/hbar_eff}, S = hbar * arg(psi)
S0 = np.angle(prod)
peff = np.sqrt(np.abs(prod)**2) * np.exp(1j * S0 / hb)
rn = np.abs(evolve(prod, c, T=1.0))**2
# evolve the linear equation with hbar_eff in the time variable of the original: i hbar_eff d psi = -(hbar_eff^2/2) Lap psi
rl = np.abs(evolve(peff, 0.0, T=1.0, hbar=hb))**2
dev = np.abs(rn - rl).max() / rn.max()
ok(f"DD5 product state, c = {c}: density under the modified equation vs linear Schrodinger with hbar_eff = sqrt(1+c) = {hb:.4f}: max "
   f"relative difference {dev:.1e}: without entanglement the marginal coefficient acts as a change of hbar", dev < 2e-3)

print("ALL PASS")
