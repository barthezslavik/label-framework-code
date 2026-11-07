# 149: QM part II (paper12, new sections), step 4. Verdict on the extra invariant metrics (marginal Fisher-Rao terms, coefficient c):
# signalling. Two particles on a line, hbar = m = 1, no interaction. Initial entangled state (phi_L(x1) chi_a(x2) + phi_R(x1) chi_b(x2))
# / sqrt 2 with different chi_a, chi_b. Alice either does nothing or measures on which side particle 1 is at t = 0; Bob looks at the
# density of particle 2 at time T, averaged over Alice's outcomes with Born weights.
# DS1 with c = 0 Bob's density does not depend on Alice's choice (to numerical precision): no signalling in linear QM;
# DS2 with c != 0 it does, and the difference grows linearly with c at small c: every non-zero marginal coefficient allows signalling
#     (the specific case of Gisin's argument for this family);
# DS3 the full Fisher-Rao term alone (c = 0) is selected by Diff(M)^N x S_N together with no-signalling: verdict "forbids" for the
#     marginal terms under the no-signalling premise.
# DS4 data estimate. A bound state of two particles (masses m1, m2, M = m1 + m2) with centre-of-mass width sigma much larger than its
#     size a: the marginal terms add c (hbar^2/8) sum_i I(rho_i)/m_i, which for sigma >> a tends to c (M/mu) times the centre-of-mass
#     Fisher term (hbar^2/8M) I_cm, with M/mu = M^2/(m1 m2) (computed for sigma/a = 4, 8, 16). The centre of mass then moves with
#     hbar_eff^2 = hbar^2 (1 + c R), R = sum_i M/m_i, while internal levels keep hbar. For a 87Rb atom (37 electrons and a nucleus)
#     R = 5.9e6. Atom-recoil h/m gives alpha with delta alpha / alpha = c R / 4; agreement of alpha from 87Rb recoil and from the electron
#     g-2 at about 1e-9 (premise; Morel et al. 2020, Fan et al. 2023) bounds |c| below about 7e-16 (estimate under these premises).
# Run: ~/.venvs/jev/bin/python ga_dens_signal_check.py
import numpy as np

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

n, Lb = 128, 28.0
x = (np.arange(n) - n / 2) * (Lb / n); dx = x[1] - x[0]
X1, X2 = np.meshgrid(x, x, indexing='ij')
k = 2 * np.pi * np.fft.fftfreq(n, d=dx); K1, K2 = np.meshgrid(k, k, indexing='ij')

def lap1(f):
    return np.real(np.fft.ifft(-k**2 * np.fft.fft(f)))
def marg(r, ax):
    return r.sum(axis=1 - ax) * dx
def Qb(m):
    sm = np.sqrt(np.maximum(m, 1e-300))
    w = m > 1e-12 * m.max()
    return np.where(w, -0.5 * lap1(sm) / np.where(w, sm, 1.0), 0.0)
def evolve(psi, c, T=1.5, steps=1200):
    dt = T / steps
    kin = np.exp(-1j * (K1**2 + K2**2) * dt / 4)
    for _ in range(steps):
        psi = np.fft.ifft2(kin * np.fft.fft2(psi))
        if c != 0.0:
            rr = np.abs(psi)**2
            psi = psi * np.exp(-1j * c * (Qb(marg(rr, 0))[:, None] + Qb(marg(rr, 1))[None, :]) * dt)
        psi = np.fft.ifft2(kin * np.fft.fft2(psi))
    return psi
def norm(p):
    return p / np.sqrt((np.abs(p)**2).sum() * dx * dx)

phL = np.exp(-(x + 5)**2 / 2); phR = np.exp(-(x - 5)**2 / 2)
chA = np.exp(-x**2 / 0.6); chB = np.exp(-(x - 1)**2 / 3.0) * np.exp(0.8j * x)
phL /= np.sqrt((np.abs(phL)**2).sum() * dx); phR /= np.sqrt((np.abs(phR)**2).sum() * dx)
chA /= np.sqrt((np.abs(chA)**2).sum() * dx); chB /= np.sqrt((np.abs(chB)**2).sum() * dx)
psi0 = norm(phL[:, None] * chA[None, :] + phR[:, None] * chB[None, :])
left = (x < 0)[:, None] * np.ones((1, n))

def bob(c, measure):
    if not measure:
        return marg(np.abs(evolve(psi0, c))**2, 1)
    out = 0
    for P in (left, 1 - left):
        br = psi0 * P
        w = (np.abs(br)**2).sum() * dx * dx
        out = out + w * marg(np.abs(evolve(br / np.sqrt(w), c))**2, 1)
    return out

diffs = {}
for c in (0.0, 0.02, 0.05, 0.1):
    diffs[c] = np.abs(bob(c, True) - bob(c, False)).sum() * dx       # L1 distance of Bob's densities
ratio = diffs[0.1] / diffs[0.05]
ok(f"DS1 c = 0: L1 distance of Bob's density with and without Alice's measurement {diffs[0.0]:.1e}: no signalling in linear QM",
   diffs[0.0] < 1e-10)
ok("DS2 c != 0: " + ", ".join(f"c={c}: {v:.2e}" for c, v in diffs.items() if c) + f"; ratio for c = 0.1/0.05: {ratio:.2f} (linear in c): "
   "every non-zero marginal coefficient lets Alice signal to Bob", all(diffs[c] > 1e-5 for c in (0.02, 0.05, 0.1)) and 1.6 < ratio < 2.4)
ok("DS3 Diff(M)^N x S_N leaves c free (DM4); no-signalling sets c = 0 and selects the full Fisher-Rao term, hence linear many-body "
   "Schrodinger evolution: verdict 'forbids' for the marginal terms under the no-signalling premise", diffs[0.0] < 1e-10 and diffs[0.02] > 1e-5)

# ---------- DS4 ----------
def Rratio(m1, m2, a, sg, nn=1024):
    M = m1 + m2
    L = 10 * max(sg, a) + 10 * a
    xs = (np.arange(nn) - nn / 2) * (L / nn); h = xs[1] - xs[0]
    Y1, Y2 = np.meshgrid(xs, xs, indexing='ij')
    Xc = (m1 * Y1 + m2 * Y2) / M; rr = Y1 - Y2
    rho = np.exp(-Xc**2 / (2 * sg**2)) * np.exp(-rr**2 / (2 * a**2)); rho /= rho.sum() * h * h
    kk = 2 * np.pi * np.fft.fftfreq(nn, d=h)
    def Ifis(m):
        w = m > 1e-13 * m.max(); dm = np.real(np.fft.ifft(1j * kk * np.fft.fft(m)))
        return ((dm**2 / np.where(w, m, 1.0)) * w).sum() * h
    I1, I2 = Ifis(rho.sum(axis=1) * h), Ifis(rho.sum(axis=0) * h)
    return M * (I1 / m1 + I2 / m2) * sg**2, M**2 / (m1 * m2)     # ratio to (1/M) I_cm with I_cm = 1/sigma^2
rows = [(sg,) + Rratio(1.0, 5.0, 0.5, sg) for sg in (2.0, 4.0, 8.0)]
dev = [abs(R / Rp - 1) for sg, R, Rp in rows]
mu_u = 1 / 1822.888                                                   # electron mass in atomic mass units
M_Rb = 86.909
R_Rb = 37 * M_Rb / mu_u + M_Rb / (M_Rb - 37 * mu_u)
c_bound = 4 * 1e-9 / R_Rb
ok("DS4 ratio of the marginal terms to the centre-of-mass Fisher term, m1 = 1, m2 = 5, a = 0.5: " + ", ".join(
   f"sigma={sg}: {R:.3f}" for sg, R, Rp in rows) + f" -> M/mu = {rows[0][2]:.3f} (deviation {dev[-1]:.1e} at sigma/a = 16); "
   f"87Rb: R = {R_Rb:.2e}; with alpha agreement 1e-9 (premise) |c| < {c_bound:.0e} (estimate)",
   dev[0] > dev[1] > dev[2] and dev[2] < 0.02 and 5e-16 < c_bound < 1e-15)

print("ALL PASS")
