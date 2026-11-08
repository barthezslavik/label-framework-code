# 154: paper13, step 6. Horizons and the Born rule.
# HB1 the entropy of label arrangements inside a ball of radius R at fixed label density grows like N ~ R^3 (log-log slope 3), not like
#     the area R^2: the label count does not give the area law; the horizon entropy of the series stays a premise on fields
#     (rows P3, P4 of paper8);
# HB2 a label medium has n = |psi|^2 by construction, so its relative entropy to |psi|^2 is zero at all times: there is nothing to
#     relax (paper12, QD3);
# HB3 a density guided by psi in de Broglie-Bohm (two-dimensional box, eight modes) that starts away from |psi|^2 relaxes: Valentini's
#     coarse-grained H = int rho_cg ln(rho_cg/|psi|^2_cg) falls to less than half of its initial value; relaxation needs a guidance law,
#     which label media do not have.
# Run: ~/.venvs/jev/bin/python ga_stat_horizon_born_check.py
import numpy as np
from scipy.special import gammaln

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

# HB1
dens, cell = 1.0, 0.05                                     # labels per unit volume; cell volume of the coarse count
Rs = np.array([2.0, 4.0, 8.0, 16.0])
Sl = []
for R in Rs:
    Vb = 4 / 3 * np.pi * R**3; N = dens * Vb; K = Vb / cell
    Sl.append(gammaln(K + 1) - gammaln(N + 1) - gammaln(K - N + 1))  # ln C(K, N): arrangements of N labels in K cells
slope = np.polyfit(np.log(Rs), np.log(Sl), 1)[0]
ok(f"HB1 entropy of label arrangements in a ball: log-log slope {slope:.3f} in R (volume law 3, area law 2): the label count does not "
   f"give the area law", abs(slope - 3) < 0.02)

# HB2 + HB3 in the box [0, pi]^2
rng = np.random.default_rng(4)
modes = [(1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1), (2, 3), (3, 2)]
c = (rng.normal(size=len(modes)) + 1j * rng.normal(size=len(modes))); c /= np.linalg.norm(c)
Em = np.array([(m * m + n * n) / 2 for m, n in modes])
def psi_grad(x, y, t):
    ps = np.zeros_like(x, dtype=complex); px = np.zeros_like(ps); py = np.zeros_like(ps)
    for (m, n), cc, E in zip(modes, c, Em):
        ph = cc * np.exp(-1j * E * t) * (2 / np.pi)
        ps += ph * np.sin(m * x) * np.sin(n * y); px += ph * m * np.cos(m * x) * np.sin(n * y); py += ph * n * np.sin(m * x) * np.cos(n * y)
    return ps, px, py
def vel(x, y, t):
    ps, px, py = psi_grad(x, y, t)
    den = np.where(np.abs(ps)**2 > 1e-12, ps, 1e-6)
    return np.imag(px / den), np.imag(py / den)
nc = 8; edges = np.linspace(0, np.pi, nc + 1)
gx = np.linspace(0, np.pi, 241); GX, GY = np.meshgrid(gx, gx, indexing='ij')
def born_cg(t):
    ps, _, _ = psi_grad(GX, GY, t); w = np.abs(ps)**2
    H, _, _ = np.histogram2d(GX.ravel(), GY.ravel(), bins=[edges, edges], weights=w.ravel()); return H / H.sum()
def Hfun(x, y, t):
    P, _, _ = np.histogram2d(x, y, bins=[edges, edges]); P = P / P.sum(); Q = born_cg(t)
    msk = P > 0
    return (P[msk] * np.log(P[msk] / np.maximum(Q[msk], 1e-300))).sum()
# HB2: label medium, density sampled from |psi|^2 itself
ps0, _, _ = psi_grad(GX, GY, 0.0); w0 = (np.abs(ps0)**2).ravel(); w0 /= w0.sum()
Hmed = []
for t in (0.0, 1.0, 2.0):
    psT, _, _ = psi_grad(GX, GY, t); P = (np.abs(psT)**2); P = P / P.sum()
    Pc, _, _ = np.histogram2d(GX.ravel(), GY.ravel(), bins=[edges, edges], weights=P.ravel()); Q = born_cg(t)
    Hmed.append(float(np.sum(Pc * np.log(np.maximum(Pc, 1e-300) / np.maximum(Q, 1e-300)))))
ok(f"HB2 label medium (n = |psi|^2 by construction): coarse-grained H = {max(abs(h) for h in Hmed):.1e} at t = 0, 1, 2: nothing to relax",
   max(abs(h) for h in Hmed) < 1e-12)
# HB3: guided particles from a non-equilibrium start (uniform in the lower-left quarter)
Np = 6000
x = rng.uniform(0.05, np.pi / 2, Np); y = rng.uniform(0.05, np.pi / 2, Np)
H0 = Hfun(x, y, 0.0); t = 0.0; dt = 2e-3; T = 12.0
while t < T - 1e-12:
    k1 = vel(x, y, t); k2 = vel(x + dt / 2 * k1[0], y + dt / 2 * k1[1], t + dt / 2)
    k3 = vel(x + dt / 2 * k2[0], y + dt / 2 * k2[1], t + dt / 2); k4 = vel(x + dt * k3[0], y + dt * k3[1], t + dt)
    x = np.clip(x + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]), 1e-6, np.pi - 1e-6)
    y = np.clip(y + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]), 1e-6, np.pi - 1e-6)
    t += dt
H1 = Hfun(x, y, T)
ok(f"HB3 guided density from a non-equilibrium start: coarse-grained H {H0:.3f} -> {H1:.3f} at t = {T}: relaxation towards |psi|^2 "
   f"needs a guidance law", H1 < 0.5 * H0)

print("ALL PASS")
