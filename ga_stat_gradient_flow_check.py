# 154: paper13, step 4. Gradient flows on densities (one dimension, periodic box of length 2 pi unless stated).
# GF1 heat equation d_t rho = D rho'': dS/dt = D I(rho) with S = -int rho ln rho and I = int rho'^2/rho (de Bruijn), along the flow;
# GF2 one step of the Jordan-Kinderlehrer-Otto scheme, argmin W2^2/(2 tau) + int rho ln rho, computed through quantile functions on the
#     line, matches the heat-equation solution after time tau (Wasserstein gradient flow of the entropy);
# GF3 Fokker-Planck d_t rho = (rho' + rho V')': the free energy F = int rho ln rho + int rho V decreases with dF/dt = -int rho
#     (ln rho + V)'^2, and the stationary density is the Gibbs density e^{-V}/Z;
# GF4 the same functional I(rho) is the entropy production of diffusion and, times hbar^2/8m, the quantum (Bohm) energy of paper12:
#     for D = hbar/2m (Nelson's diffusion constant) dS/dt = (4/hbar) E_Q, checked along the flow for hbar = m = 1 and hbar = 0.5.
# Run: ~/.venvs/jev/bin/python ga_stat_gradient_flow_check.py
import numpy as np
from scipy.optimize import minimize

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

n = 512; Lb = 2 * np.pi
x = Lb * np.arange(n) / n; dx = Lb / n
k = 2 * np.pi * np.fft.fftfreq(n, d=dx)
def d1(f): return np.real(np.fft.ifft(1j * k * np.fft.fft(f)))
def heat(f, D, t): return np.real(np.fft.ifft(np.fft.fft(f) * np.exp(-D * k**2 * t)))
def S(r): return -(r * np.log(r)).sum() * dx
def I(r): return (d1(r)**2 / r).sum() * dx
r0 = np.exp(np.cos(x) + 0.6 * np.sin(2 * x + 0.3)); r0 /= r0.sum() * dx

# GF1
D = 0.7; res = []
for t in (0.05, 0.2, 0.6):
    h = 1e-4
    dSdt = (S(heat(r0, D, t + h)) - S(heat(r0, D, t - h))) / (2 * h)
    res.append((t, dSdt, D * I(heat(r0, D, t))))
ok("GF1 heat equation: dS/dt vs D I(rho): " + ", ".join(f"t={t}: {a:.8f} / {b:.8f}" for t, a, b in res) + " (de Bruijn)",
   all(abs(a / b - 1) < 1e-6 for _, a, b in res))

# GF2 JKO step on the line (non-periodic: Gaussian mixture, compare with the free heat kernel)
m = 400; s = (np.arange(m) + 0.5) / m
xs = np.linspace(-12, 12, 6001); hx = xs[1] - xs[0]
p0 = 0.6 * np.exp(-(xs + 1.5)**2 / 0.5) + 0.4 * np.exp(-(xs - 1.0)**2 / 0.3); p0 /= p0.sum() * hx
def quantiles(p):
    c = np.cumsum(p) * hx; c /= c[-1]
    return np.interp(s, c, xs)
Q0 = quantiles(p0); tau = 0.05
def obj(z):
    Q = Q0[0] + z[0] + np.concatenate([[0], np.cumsum(np.exp(z[1:]))])
    dQ = np.diff(Q) * m                                 # dQ/ds
    ent = -np.log(dQ).sum() / m                        # int rho ln rho = -int ln Q'(s) ds
    w2 = ((Q - Q0)**2).sum() / m
    return w2 / (2 * tau) + ent
z0 = np.concatenate([[0.0], np.log(np.diff(Q0))])
sol = minimize(obj, z0, method='L-BFGS-B', options={'maxiter': 20000, 'maxfun': 400000})
Qj = Q0[0] + sol.x[0] + np.concatenate([[0], np.cumsum(np.exp(sol.x[1:]))])
kern = lambda sig2: np.exp(-xs**2 / (2 * sig2))
ph = np.convolve(p0, kern(2 * 1.0 * tau) / (kern(2 * tau).sum() * hx), mode='same') * hx   # heat with D = 1: variance 2 D tau
ph /= ph.sum() * hx
Qh = quantiles(ph)
w_jh = np.sqrt(((Qj - Qh)**2).mean()); w_0h = np.sqrt(((Q0 - Qh)**2).mean())
ok(f"GF2 JKO step tau = {tau}: W2 distance to the heat solution {w_jh:.4f}, against {w_0h:.4f} for no step: the entropy's Wasserstein "
   f"gradient step is the heat flow", w_jh < 0.15 * w_0h)

# GF3 Fokker-Planck with V = 2 cos x: free energy decreases, dF/dt = -relative Fisher, stationary Gibbs
V = 2 * np.cos(x); dV = d1(V)
def fp_rhs(r): return d1(d1(r) + r * dV)
def F(r): return (r * np.log(r)).sum() * dx + (r * V).sum() * dx
def step(r, dt):                                       # semi-implicit: diffusion implicit in Fourier, drift explicit
    return np.real(np.fft.ifft((np.fft.fft(r) + dt * np.fft.fft(d1(r * dV))) / (1 + dt * k**2)))
r = r0.copy(); dt = 1e-3; Fs = [F(r)]; rate_err = 0.0
for i in range(20000):
    if i in (100, 2000):
        hdt = 1e-6; rn = step(r, hdt)
        pred = -(r * d1(np.log(r) + V)**2).sum() * dx
        rate_err = max(rate_err, abs((F(rn) - F(r)) / hdt / pred - 1))
    r = step(r, dt)
    if i % 500 == 0: Fs.append(F(r))
gibbs = np.exp(-V); gibbs /= gibbs.sum() * dx
mono = all(Fs[i + 1] <= Fs[i] + 1e-12 for i in range(len(Fs) - 1))
ok(f"GF3 Fokker-Planck: free energy decreases monotonically ({Fs[0]:.5f} -> {Fs[-1]:.5f}), dF/dt = -relative Fisher to {rate_err:.1e}, "
   f"stationary density = Gibbs to {np.abs(r - gibbs).max() / gibbs.max():.1e}", mono and rate_err < 1e-3 and np.abs(r - gibbs).max() / gibbs.max() < 1e-6)

# GF4 entropy production and the quantum energy of paper12
out = []
for hb in (1.0, 0.5):
    m_ = 1.0; Dn = hb / (2 * m_); t = 0.3; h = 1e-4
    dSdt = (S(heat(r0, Dn, t + h)) - S(heat(r0, Dn, t - h))) / (2 * h)
    EQ = hb**2 / (8 * m_) * I(heat(r0, Dn, t))
    out.append((hb, dSdt, 4 / hb * EQ))
ok("GF4 dS/dt = (4/hbar) E_Q for D = hbar/2m: " + ", ".join(f"hbar={h}: {a:.8f} = {b:.8f}" for h, a, b in out)
   + "; one functional, entropy production and quantum energy", all(abs(a / b - 1) < 1e-6 for _, a, b in out))

print("ALL PASS")
