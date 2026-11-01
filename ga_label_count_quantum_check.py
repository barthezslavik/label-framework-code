# 125: is the number of labels conserved at the quantum level (level 3)? Doc 122 (C5-C7) used the identical conservation of the label
# current; here the two candidate charges are separated. Q = int N with N = dX1^dX2^dX3 (signed, base K6/K8) and M = int |N| (unsigned:
# the mass of label dust, action F(|N|)). (i) Q is unchanged by every compactly supported deformation of the label map (Piola identity:
# the variation is a divergence) -- so it commutes with every local operator: a superselection charge in any quantisation. (ii) M is not:
# a local fold of the map X(x) creates a label and an orientation-reversed copy (with unoriented labels, base K7, both are ordinary dust),
# raising M at fixed Q. (iii) Which configurations are quantised decides: on the space of embeddings (the SDiff configuration space of the
# framework, Arnold) the map stays injective and M = |Q| is kinematic; transport u.dX = 0 keeps injectivity, while an unconstrained scalar
# field X (sigma-model quantisation) folds. QL1-QL6.
# Run: ~/.venvs/jev/bin/python ga_label_count_quantum_check.py
import numpy as np
import sympy as sp

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

# QL1 Piola: d_i cof(dX)^i_a = 0 identically in 3D, so delta Q = int d_i(cof^i_a f^a) = boundary term
x = sp.symbols('x0:3', real=True)
X = [sp.Function(f'X{a}')(*x) for a in range(3)]
Jm = sp.Matrix(3, 3, lambda a, i: sp.diff(X[a], x[i]))
cof = Jm.adjugate().T                                              # cof[a, i] = d det / d(dX^a/dx^i)
piola = [sp.simplify(sum(sp.diff(cof[a, i], x[i]) for i in range(3))) for a in range(3)]
ok(f"QL1 Piola identity d_i cof^i_a = {piola}: the variation of Q under any smooth compactly supported deformation with fixed boundary "
   "conditions is a boundary term (classical; a quantum superselection follows only for quantisations that keep the smooth sector)", piola == [0, 0, 0])

# QL2 1D: a smooth monotone map plus a compact wiggle -> fold: signed Q fixed, unsigned M grows by twice the folded label length
xs = np.linspace(-10, 10, 200001)
X0 = xs.copy()
bump = lambda A: A * np.exp(-xs**2) * np.sin(3 * xs)              # compact (Gaussian) deformation
for A in (0.1, 1.0, 3.0):
    Xd = X0 + bump(A); dX = np.gradient(Xd, xs)
    Q, M = np.trapezoid(dX, xs), np.trapezoid(np.abs(dX), xs)
    print(f"   1D wiggle amplitude {A}: Q = {Q:.4f}, M = {M:.4f}, folded: {np.any(dX < 0)}")
Xd = X0 + bump(3.0); dX = np.gradient(Xd, xs)
ok(f"QL2 1D fold: Q stays {np.trapezoid(dX, xs):.4f} (= 20), M rises to {np.trapezoid(np.abs(dX), xs):.4f}: a label pair (one orientation-reversed) appears",
   abs(np.trapezoid(dX, xs) - 20) < 1e-3 and np.trapezoid(np.abs(dX), xs) > 20.5)

# QL3 3D: identity map plus a compact fold along x0 -> signed and unsigned volumes
n = 61; g = np.linspace(-3, 3, n); h = g[1] - g[0]
G0, G1, G2 = np.meshgrid(g, g, g, indexing='ij')
env = np.exp(-(G0**2 + G1**2 + G2**2))
Y0 = G0 + 2.5 * env * np.sin(3 * G0); Y1, Y2 = G1, G2
d0 = np.gradient(Y0, h, axis=0); d01 = np.gradient(Y0, h, axis=1); d02 = np.gradient(Y0, h, axis=2)
det = d0                                                           # Y1, Y2 identity: det = dY0/dx0
Qs, Ms = det.sum() * h**3, np.abs(det).sum() * h**3
ok(f"QL3 3D local fold: Q = {Qs:.3f} (box volume {(g[-1] - g[0] + h)**3:.3f}), M = {Ms:.3f}: unsigned count rises at fixed signed charge",
   abs(Qs - (6 + h)**3) / (6 + h)**3 < 0.01 and Ms > Qs * 1.01)

# QL4 transport keeps injectivity: X advected by a smooth velocity field (u.dX = 0 along the flow) stays monotone, M constant
# exact transport by characteristics: X(x, t) = X0(phi_-t(x)), phi the flow of dx/dt = u(x) (RK4 backwards)
uf = lambda y: 2.0 * np.exp(-y**2 / 4) * np.cos(y)
xg = np.linspace(-10, 10, 4001); y = xg.copy(); dt = -1e-3
for _ in range(1000):
    k1 = uf(y); k2 = uf(y + dt * k1 / 2); k3 = uf(y + dt * k2 / 2); k4 = uf(y + dt * k3); y = y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
Xa = y; dXa = np.gradient(Xa, xg)                                 # X0 = identity, so X(x, 1) = phi_-1(x)
Ma = np.trapezoid(np.abs(dXa), xg)
ok(f"QL4 transport by a smooth flow (t = 1): min dX/dx = {dXa.min():.3f} > 0, M = {Ma:.4f} = Q: on embeddings the count is kinematic", dXa.min() > 0 and abs(Ma - np.trapezoid(dXa, xg)) < 1e-9)

# QL5 an unconstrained scalar X (wave equation, sigma-model quantisation) with a local momentum kick folds: M changes
def wave(X, V, dt, steps):
    for _ in range(steps):
        lap = np.zeros_like(X); lap[1:-1] = (X[2:] - 2 * X[1:-1] + X[:-2]) / (xs[1] - xs[0])**2
        V = V + dt * lap; X = X + dt * V
    return X
xs_c = xs[::100]; X0c = xs_c.copy(); Vk = 6.0 * np.exp(-xs_c**2) * np.sin(3 * xs_c)
xs_bak = xs; xs = xs_c
Xw = wave(X0c.copy(), Vk, 0.005, 200); dXw = np.gradient(Xw, xs_c); xs = xs_bak
Mw = np.trapezoid(np.abs(dXw), xs_c)
ok(f"QL5 unconstrained scalar field with a local kick (t = 1): min dX/dx = {dXw.min():.2f} < 0, M = {Mw:.3f} > Q = 20: pairs form",
   dXw.min() < 0 and Mw > 20.2)
ok("QL6 recorded: Q (signed) is an invariant of the smooth sector; M (mass, label count) is conserved only when the label map is "
   "quantised on the space of embeddings (SDiff configuration space); with unoriented labels the reversed copy is ordinary dust, so in a "
   "sigma-model quantisation pair creation and annihilation of label dust are not forbidden by any charge", True)

# QL7 (paper7 review r2) the future-directed current j = sgn(J^0) J. Under label-orientation reversal X1 -> -X1, J -> -J and j -> j; on
#     embeddings J^0 never vanishes, so j = +-J is conserved and A.j is gauge invariant: the parity argument does not forbid a charge there.
#     In the sigma-model sector j^0 = |N| and its integral M changes (QL2, QL5), so A.j is not gauge invariant and only the odd A.J remains.
J4 = np.array([2.0, 0.3, -0.1, 0.2]); j_of = lambda Jx: np.sign(Jx[0]) * Jx
inv = np.allclose(j_of(-J4), j_of(J4))
ok(f"QL7 orientation reversal: J -> -J, j = sgn(J^0) J -> j ({inv}); on embeddings j is conserved (QL4: M = Q) and A.j is allowed; in a "
   "sigma-model M is not conserved (QL5), so A.j breaks gauge invariance and only the odd A.J is available", inv)
ok("QL8 recorded (classical argument, no quantum operator algebra built) dichotomy: quantised on embeddings, label matter conserves its count (no annihilation, no pair production) and may carry a gauge "
   "charge (a label); quantised as a sigma model, it carries no charge and its count is not protected. The combination forbidden in both "
   "schemes is dark matter with an abelian gauge charge that also annihilates or is pair-produced (non-abelian charges need an "
   "internal index the scalar label current lacks; not classified)", True)
