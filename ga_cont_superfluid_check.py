# 155/156: paper14, step 4. Superfluid: the phase as a field, the quantum of circulation, and the link to the premise of paper12 (Q2).
# SF1 hydrodynamics alone does not quantise: the irrotational vortex v = w (hbar/m) e_phi / r with the Bernoulli pressure solves the
#     stationary Euler equations for every real w (residual at random points), so the circulation 2 pi w hbar/m can take any value;
# SF2 with the phase on a circle (single-valued e^{i theta}, premise Q2 of paper12, the condition of Takabayasi and Wallstrom) the
#     winding w is an integer: e^{i theta} closes around the core only for integer w, and the circulation around any loop that encloses
#     the core is w h/m (line integrals over a circle, a square and an off-centre loop; zero for a loop that does not enclose it);
# SF3 value: h/m_4 for helium-4 (CODATA h, atomic mass of 4He) is 9.97e-8 m^2/s; Rayfield and Reif (1964) measured
#     (1.00 +- 0.03)e-7 m^2/s from vortex rings: agreement within 0.1 sigma, while h/(2 m_4) is excluded by 17 sigma; the mass in h/m is
#     the mass of the condensing unit, a label of the medium: for 3He-B the observed quantum is h/(2 m_3) = 6.6e-8 m^2/s (Cooper pairs,
#     Davis et al. 1991), which the framework does not derive;
# SF4 a lattice of quantised vortices with Feynman's areal density 2 Omega/kappa reproduces, at scales larger than the spacing, the solid-body
#     velocity Omega r of a classical rotating fluid (mean azimuthal velocity from the Biot-Savart sum); for Omega = 1 rad/s in 4He the
#     density is 2.0e7 per m^2: the vorticity of the classical (SDiff) fluid is the coarse-grained density of quantised vortices.
# Run: ~/.venvs/jev/bin/python ga_cont_superfluid_check.py
import numpy as np

rng = np.random.default_rng(4)

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

# ---------- SF1 Euler residual for non-integer winding (units hbar/m = 1, rho = 1) ----------
def v_vortex(x, y, w):
    r2 = x**2 + y**2; return np.array([-w * y / r2, w * x / r2])
def euler_residual(w, n=200):
    res = 0.0
    for _ in range(n):
        x, y = rng.uniform(0.3, 2, 2) * rng.choice([-1, 1], 2); h = 1e-5
        v = v_vortex(x, y, w)
        dvdx = (v_vortex(x + h, y, w) - v_vortex(x - h, y, w)) / (2 * h)
        dvdy = (v_vortex(x, y + h, w) - v_vortex(x, y - h, w)) / (2 * h)
        adv = v[0] * dvdx + v[1] * dvdy
        p = lambda X, Y: -0.5 * (w**2) / (X**2 + Y**2)          # Bernoulli pressure, rho = 1
        gradp = np.array([(p(x + h, y) - p(x - h, y)) / (2 * h), (p(x, y + h) - p(x, y - h)) / (2 * h)])
        res = max(res, np.abs(adv + gradp).max() / (np.abs(adv).max() + 1e-30))
    return res
ws = [0.37, 1.0, 1.5, 2.0, np.sqrt(2)]
resid = max(euler_residual(w) for w in ws)
ok(f"SF1 the vortex v = w/r with Bernoulli pressure solves stationary Euler for w = {', '.join(f'{w:.3g}' for w in ws)}: max relative "
   f"residual {resid:.1e}; the hydrodynamic equations allow any circulation", resid < 1e-6)

# ---------- SF2 circle premise: integer winding and circulation ----------
def closure(w):
    th0 = 0.0; th1 = w * 2 * np.pi
    return abs(np.exp(1j * th1) - np.exp(1j * th0))
cl = {w: closure(w) for w in [1, 2, -1, 0.5, 0.37, 1.5]}
def circ(path, w):
    pts = path; seg = np.diff(pts, axis=0); mid = 0.5 * (pts[1:] + pts[:-1])
    v = np.array([v_vortex(mx, my, w) for mx, my in mid])
    return float(np.sum(v[:, 0] * seg[:, 0] + v[:, 1] * seg[:, 1]))
t = np.linspace(0, 2 * np.pi, 20001)
circle = np.c_[1.3 * np.cos(t), 1.3 * np.sin(t)]
s = np.linspace(-1, 1, 5001)
square = np.vstack([np.c_[s, -np.ones_like(s)], np.c_[np.ones_like(s), s], np.c_[-s, np.ones_like(s)], np.c_[-np.ones_like(s), -s]])
offc = np.c_[0.5 + 1.2 * np.cos(t), -0.3 + 0.9 * np.sin(t)]
outside = np.c_[3 + 0.5 * np.cos(t), 0.5 * np.sin(t)]
w = 2
vals = [circ(p, w) / (2 * np.pi) for p in (circle, square, offc)]; v_out = circ(outside, w) / (2 * np.pi)
ok("SF2 e^{i theta} closes around the core for w = 1, 2, -1, 0 (gap " + f"{max(cl[1], cl[2], cl[-1]):.0e}) and not for w = 0.5, 0.37, 1.5 "
   f"(gaps {cl[0.5]:.2f}, {cl[0.37]:.2f}, {cl[1.5]:.2f}); circulation/(h/m) for w = 2 on a circle, a square and an off-centre loop: "
   + ", ".join(f"{x:.6f}" for x in vals) + f"; on a loop not enclosing the core: {v_out:.1e}",
   max(cl[1], cl[2], cl[-1]) < 1e-12 and min(cl[0.5], cl[0.37], cl[1.5]) > 0.5 and max(abs(x - 2) for x in vals) < 1e-4 and abs(v_out) < 1e-4)

# ---------- SF3 values ----------
h = 6.62607015e-34                                      # CODATA 2018, exact
u = 1.66053906660e-27                                   # CODATA 2018
m4 = 4.002603254 * u                                    # atomic mass of 4He (AME 2020)
m3 = 3.016029322 * u                                    # atomic mass of 3He (AME 2020)
k4 = h / m4; k_meas, s_meas = 1.00e-7, 0.03e-7          # Rayfield & Reif, Phys. Rev. 136 (1964) A1194: (1.00 +- 0.03)e-3 cm^2/s
z1 = abs(k4 - k_meas) / s_meas; z2 = abs(h / (2 * m4) - k_meas) / s_meas
k3B = h / (2 * m3)
ok(f"SF3 h/m_4 = {k4:.4e} m^2/s vs measured {k_meas:.2e} +- {s_meas:.0e}: {z1:.2f} sigma; h/(2 m_4) = {h / (2 * m4):.3e} is {z2:.0f} sigma off; "
   f"3He-B quantum h/(2 m_3) = {k3B:.3e} m^2/s (observed form, Davis et al. 1991): the condensing mass is a label",
   z1 < 1 and z2 > 10 and abs(k3B / k4 - m4 / (2 * m3)) < 1e-12)

# ---------- SF4 vortex lattice -> solid-body rotation ----------
Omega = 1.0; kappa = 1.0                                # units: kappa = 1, Omega = 1 -> density 2/(kappa) per unit area
dens = 2 * Omega / kappa
a = np.sqrt(2 / (np.sqrt(3) * dens))                    # triangular lattice spacing with that density
R = 40 * a
pts = []
for i in range(-60, 61):
    for j in range(-60, 61):
        x = a * (i + 0.5 * j); y = a * (np.sqrt(3) / 2) * j
        if x * x + y * y < R * R: pts.append((x, y))
pts = np.array(pts)
def vtheta_mean(r, nphi=720):
    ph = np.linspace(0, 2 * np.pi, nphi, endpoint=False); X = r * np.cos(ph); Y = r * np.sin(ph)
    dx = X[:, None] - pts[None, :, 0]; dy = Y[:, None] - pts[None, :, 1]; r2 = dx**2 + dy**2
    vx = (-kappa / (2 * np.pi) * dy / r2).sum(1); vy = (kappa / (2 * np.pi) * dx / r2).sum(1)
    return np.mean(-vx * np.sin(ph) + vy * np.cos(ph))
rs = [10.3 * a, 20.1 * a, 30.2 * a]
ratios = [vtheta_mean(r) / (Omega * r) for r in rs]
n_He = 2 * 1.0 / k4
ok("SF4 vortex lattice with density 2 Omega/kappa: mean v_phi/(Omega r) = " + ", ".join(f"{x:.4f}" for x in ratios) +
   f" at r = 10.3, 20.1, 30.2 spacings; for 4He at Omega = 1 rad/s the density is {n_He:.2e} per m^2", max(abs(x - 1) for x in ratios) < 0.016)

print("ALL PASS")
