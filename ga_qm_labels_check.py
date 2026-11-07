# 145: quantum mechanics from the label medium (steps 2-3 of paper12). Equations borrowed from Madelung (1927), Bohm (1952),
# Takabayasi (1952), Wallstrom (PRA 49, 1613, 1994), Reginatto (PRA 58, 1775, 1998), Holland (1211.0983); every step recomputed here.
# QM1 relabelling invariance: n = det dX and the gradient term K(n)(grad n)^2 are unchanged by a unit-Jacobian relabelling X -> f(X);
# QM2 the variational derivative of int (hbar^2/8m)(grad n)^2/n is the Bohm potential Q = -(hbar^2/2m) lap(sqrt n)/sqrt n (3D, symbolic);
# QM3 with v = grad S/m, psi = sqrt(n) exp(iS/hbar) obeys i hbar psi_t = -(hbar^2/2m) lap psi + V psi + U'(|psi|^2) psi exactly when the
#     continuity and Bernoulli (Hamilton-Jacobi) equations of the medium hold: the residual is a combination of the two (symbolic);
# QM4 linearity: the psi equation is linear iff U is linear in n (pressureless medium, p = nU' - U = 0) and the gradient term is
#     K = hbar^2/(8 m n); for K = c n^alpha the variational derivative is proportional to Q iff the ratio of its (grad n)^2 and
#     lap n coefficients, alpha/(2n), equals that of Q, -1/(2n): alpha = -1 (symbolic, every alpha); spot checks at four alpha;
# QM5 Kelvin: every force of the medium is a gradient (Bohm, pressure, potential), so the circulation of a material loop is conserved;
#     curl-free initial data stay curl-free (numerical Lagrangian transport of a loop); zero vorticity is initial data, not a law;
# QM6 Wallstrom: psi = J_nu(kr) exp(i nu theta) solves the Helmholtz equation, hence the local medium equations, for non-integer nu;
#     the phase map exp(iS/hbar) on a loop around the axis is continuous only for integer nu: integrality of the circulation is the
#     statement that the phase label takes values in a circle (label space S^1), not a consequence of the local equations;
# QM7 n = |psi|^2 holds by construction (psi is built from n), with the label count int n = norm, conserved;
# QM8 a label medium in 3-space cannot carry entangled many-body quantum mechanics: two two-particle states whose one-body densities
#     and currents agree at t = 0 for both particles (an entangled state and a product of two superpositions) give different one-body
#     densities later under the same free Hamiltonian, so no closed dynamics of one-body label fields reproduces them.
# Run: ~/.venvs/jev/bin/python ga_qm_labels_check.py
import numpy as np
import sympy as sp
from scipy.special import jv, jvp

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

x, y, z, t = sp.symbols('x y z t', real=True)
hb, m = sp.symbols('hbar m', positive=True)
R3 = (x, y, z)

# ---------- QM1 relabelling invariance ----------
X1 = x + sp.Rational(1, 3) * y**2 + t * z
X2 = y + sp.sin(z) * sp.Rational(1, 2)
X3 = z + x * y / 5
Xs = [X1, X2, X3]
n_of = lambda L: sp.Matrix(3, 3, lambda a, i: sp.diff(L[a], R3[i])).det()
n0 = sp.simplify(n_of(Xs))
a1, a2, a3 = sp.symbols('a1 a2 a3')
# unit-Jacobian relabelling f: a shear followed by a symplectic-type twist
f = [a1 + sp.sin(a2) * a3, a2 + a3**3 / 7, a3]
Jf = sp.Matrix(3, 3, lambda i, j: sp.diff(f[i], [a1, a2, a3][j]))
detf = sp.simplify(Jf.det())
Xr = [fi.subs({a1: X1, a2: X2, a3: X3}, simultaneous=True) for fi in f]
nr = sp.simplify(n_of(Xr))
grad = lambda h: [sp.diff(h, v) for v in R3]
gt0 = sum(g**2 for g in grad(n0)) / n0
gtr = sum(g**2 for g in grad(nr)) / nr
ok(f"QM1 det df = {detf}; n after relabelling - n = {sp.simplify(nr - n0)}; Fisher term difference = "
   f"{sp.simplify(gtr - gt0)}: the gradient term is relabelling-invariant",
   detf == 1 and sp.simplify(nr - n0) == 0 and sp.simplify(gtr - gt0) == 0)

# ---------- QM2 variational derivative of the Fisher term ----------
nf = sp.Function('n', positive=True)(*R3)
dens = hb**2 / (8 * m) * sum(sp.diff(nf, v)**2 for v in R3) / nf
EL = sp.diff(dens, nf) - sum(sp.diff(sp.diff(dens, sp.diff(nf, v)), v) for v in R3)
sq = sp.sqrt(nf)
Q = -hb**2 / (2 * m) * sum(sp.diff(sq, v, 2) for v in R3) / sq
ok(f"QM2 delta/delta n of int (hbar^2/8m)(grad n)^2/n minus the Bohm potential Q = {sp.simplify(EL - Q)}", sp.simplify(EL - Q) == 0)

# ---------- QM3 Schrodinger from continuity + Bernoulli ----------
nn = sp.Function('n', positive=True)(x, y, z, t)
S = sp.Function('S', real=True)(x, y, z, t)
V = sp.Function('V', real=True)(x, y, z)
U = sp.Function('U')
psi = sp.sqrt(nn) * sp.exp(sp.I * S / hb)
lap = lambda h: sum(sp.diff(h, v, 2) for v in R3)
Up = sp.diff(U(nn), nn)
res = sp.I * hb * sp.diff(psi, t) + hb**2 / (2 * m) * lap(psi) - V * psi - Up * psi
cont = sp.diff(nn, t) + sum(sp.diff(nn * sp.diff(S, v) / m, v) for v in R3)
Qn = -hb**2 / (2 * m) * lap(sp.sqrt(nn)) / sp.sqrt(nn)
bern = sp.diff(S, t) + sum(sp.diff(S, v)**2 for v in R3) / (2 * m) + V + Up + Qn
combo = sp.simplify(sp.expand(res / psi + bern - sp.I * hb * cont / (2 * nn)))
ok(f"QM3 (Schrodinger residual)/psi = -(Bernoulli) + (i hbar/2n)(continuity), remainder {combo}: psi = sqrt(n)e^(iS/hbar) solves the "
   "Schrodinger equation exactly when the irrotational medium obeys its two equations", combo == 0)

# ---------- QM4 linearity ----------
c, al, kap = sp.symbols('c alpha kappa', real=True)
n1 = sp.Function('n', positive=True)(x)
def euler(K):
    d = K * sp.diff(n1, x)**2
    return sp.diff(d, n1) - sp.diff(sp.diff(d, sp.diff(n1, x)), x)
Q1 = -hb**2 / (2 * m) * sp.diff(sp.sqrt(n1), x, 2) / sp.sqrt(n1)
diff_fisher = sp.simplify(euler(hb**2 / (8 * m * n1)) - Q1)
# for K = c n^alpha the potential term of the psi equation must be Q (linear) up to a function of n alone: test alpha = -1/2, -2, 0
nonlin = {}
for alv in [sp.Rational(-1, 2), -2, 0, 1]:
    for cv in [hb**2 / (8 * m)]:
        dlt = sp.simplify(euler(cv * n1**alv) - Q1)
        # does dlt depend on derivatives of n? then it is not absorbable into U'(n)
        nonlin[str(alv)] = dlt.has(sp.Derivative)
# general alpha: delta/delta n of c n^alpha (n')^2 = -c[alpha n^(alpha-1) (n')^2 + 2 n^alpha n'']; Q = (hbar^2/8m)(n')^2/n^2 - (hbar^2/4m) n''/n;
# proportional to Q iff the ratio of the (n')^2 and n'' coefficients agrees: alpha/(2n) = -1/(2n), i.e. alpha = -1
cs = sp.Symbol('c', positive=True); ns, d1s, d2s = sp.symbols('n_s d1 d2', positive=True)
ELg = sp.expand(euler(cs * n1**al).subs({sp.Derivative(n1, (x, 2)): d2s}).subs({sp.Derivative(n1, x): d1s}).subs(n1, ns))
Qg = sp.expand(Q1.doit().subs({sp.Derivative(n1, (x, 2)): d2s}).subs({sp.Derivative(n1, x): d1s}).subs(n1, ns))
ratio_EL = sp.simplify(ELg.coeff(d1s, 2) / ELg.coeff(d2s, 1)); ratio_Q = sp.simplify(Qg.coeff(d1s, 2) / Qg.coeff(d2s, 1))
alpha_sol = sp.solve(sp.Eq(ratio_EL, ratio_Q), al)
pressure = sp.simplify(sp.Symbol('N') * sp.diff(kap * sp.Symbol('N'), sp.Symbol('N')) - kap * sp.Symbol('N'))
g = sp.symbols('g', positive=True)
NN = sp.Symbol('N', positive=True)
U2 = g * NN**2 / 2
p2 = sp.simplify(NN * sp.diff(U2, NN) - U2)
ok(f"QM4 Fisher term gives Q exactly (difference {diff_fisher}); for K = c n^alpha the coefficient ratio (n')^2 : n'' is {ratio_EL}, "
   f"that of Q is {ratio_Q}, equal only for alpha = {alpha_sol}; K = (hbar^2/8m) n^alpha leaves derivative terms beyond Q for "
   f"alpha = {[k for k, v in nonlin.items() if v]}; U = kappa n has pressure {pressure} (linear psi equation), U = g n^2/2 has pressure "
   f"{p2} and the term g|psi|^2 psi: linear Schrodinger = pressureless medium + Fisher gradient term",
   diff_fisher == 0 and alpha_sol == [-1] and all(nonlin.values()) and pressure == 0 and p2 == g * NN**2 / 2)

# ---------- QM5 Kelvin: transport of a material loop in a gradient flow ----------
# 2D flow v = grad phi, phi = 0.3 sin(x+0.2t) cosh(0.4y) + 0.05(x^2 - y^2), plus an optional line vortex of strength G
# (a gradient flow off its axis): the circulation of a material loop stays 0, or stays G around the vortex.
def vel(px, py, tt, G):
    vx = 0.3 * np.cos(px + 0.2 * tt) * np.cosh(0.4 * py) + 0.1 * px
    vy = 0.12 * np.sin(px + 0.2 * tt) * np.sinh(0.4 * py) - 0.1 * py
    r2 = (px - 3.0)**2 + (py - 0.5)**2
    return vx - G * (py - 0.5) / (2 * np.pi * r2), vy + G * (px - 3.0) / (2 * np.pi * r2)
def circ(P, G, tt):
    vx, vy = vel(P[:, 0], P[:, 1], tt, G)
    dP = np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)
    return 0.5 * np.sum(vx * dP[:, 0] + vy * dP[:, 1])
def transport(G):
    s = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
    P = np.c_[3.0 + 0.8 * np.cos(s), 0.5 + 0.8 * np.sin(s)]
    c0 = circ(P, G, 0.0); dt = 2e-3; tt = 0.0
    for _ in range(500):
        k1 = np.array(vel(P[:, 0], P[:, 1], tt, G)).T
        k2 = np.array(vel(*(P + 0.5 * dt * k1).T, tt + 0.5 * dt, G)).T
        k3 = np.array(vel(*(P + 0.5 * dt * k2).T, tt + 0.5 * dt, G)).T
        k4 = np.array(vel(*(P + dt * k3).T, tt + dt, G)).T
        P = P + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4); tt += dt
    return c0, circ(P, G, tt)
c0a, c1a = transport(0.0)
c0b, c1b = transport(0.7)
ok(f"QM5 material loop in a gradient flow: circulation {c0a:.2e} -> {c1a:.2e}; with a line vortex of strength 0.7: {c0b:.4f} -> "
   f"{c1b:.4f}. Forces of the medium are gradients, so zero vorticity is kept but must be given as initial data",
   abs(c0a) < 1e-6 and abs(c1a) < 1e-6 and abs(c0b - 0.7) < 1e-3 and abs(c1b - 0.7) < 1e-3)

# ---------- QM6 Wallstrom ----------
nu, kk = 0.37, 1.3
r = np.linspace(0.5, 4.0, 200)
# Helmholtz in polar coordinates for J_nu(kr) e^{i nu theta}: f'' + f'/r + (k^2 - nu^2/r^2) f = 0
fr = jv(nu, kk * r); fp = kk * jvp(nu, kk * r); fpp = kk**2 * jvp(nu, kk * r, 2)
helm = np.max(np.abs(fpp + fp / r + (kk**2 - nu**2 / r**2) * fr))
th = np.linspace(0, 2 * np.pi, 1001)
jump = {v: abs(np.exp(1j * v * th[-1]) - np.exp(1j * v * th[0])) for v in [0.37, 1.0, 2.0]}
circ_nu = nu  # circulation in units of 2 pi hbar / m
ok(f"QM6 J_nu(kr)e^(i nu theta) with nu = {nu}: Helmholtz residual {helm:.1e}, so the local medium equations hold with circulation "
   f"{circ_nu} x 2 pi hbar/m; the phase map around the axis closes for nu = 1, 2 (gap {jump[1.0]:.1e}, {jump[2.0]:.1e}) and not for nu = 0.37 "
   f"(gap {jump[0.37]:.3f}): integrality = the phase label is circle-valued, a statement about label space",
   helm < 1e-10 and jump[1.0] < 1e-12 and jump[2.0] < 1e-12 and jump[0.37] > 0.5)

# ---------- QM7 n = |psi|^2 by construction; norm = label count ----------
L, Ng = 80.0, 2048
xs = np.linspace(-L / 2, L / 2, Ng, endpoint=False); dx = xs[1] - xs[0]
kx = 2 * np.pi * np.fft.fftfreq(Ng, d=dx)
nlab = np.exp(-(xs + 5)**2 / 4) + 0.5 * np.exp(-(xs - 6)**2 / 3)
Sph = 0.8 * xs + 0.1 * np.sin(xs)
ps = np.sqrt(nlab) * np.exp(1j * Sph)
def free(ps, T, hbar=1.0, mass=1.0):
    return np.fft.ifft(np.fft.fft(ps) * np.exp(-1j * hbar * kx**2 * T / (2 * mass)))
N0 = np.sum(nlab) * dx; N1 = np.sum(np.abs(free(ps, 7.0))**2) * dx
ok(f"QM7 |psi|^2 - n = {np.max(np.abs(np.abs(ps)**2 - nlab)):.1e} (identity of the construction); label count {N0:.6f} -> {N1:.6f} "
   "after free evolution: the norm is the conserved label count",
   np.max(np.abs(np.abs(ps)**2 - nlab)) < 1e-12 and abs(N0 - N1) < 1e-9)

# ---------- QM8 entanglement: one-body fields do not close ----------
Nx = 256; Lx = 60.0
g1 = np.linspace(-Lx / 2, Lx / 2, Nx, endpoint=False); d1 = g1[1] - g1[0]
k1 = 2 * np.pi * np.fft.fftfreq(Nx, d=d1)
def packet(x0, k0, s=1.2):
    w = np.exp(-(g1 - x0)**2 / (4 * s**2) + 1j * k0 * g1)
    return w / np.sqrt(np.sum(np.abs(w)**2) * d1)
pL, pR = packet(-9.0, 1.5), packet(9.0, -1.5)
ca, cb = packet(-10.0, 0.0, 1.0), packet(10.0, 0.0, 1.0)      # orthogonal states of particle 2
prod = np.outer((pL + pR) / np.sqrt(2), (ca + cb) / np.sqrt(2))  # product of two superpositions
ent = (np.outer(pL, ca) + np.outer(pR, cb)) / np.sqrt(2)        # entangled with particle 2
def norm2(W): return W / np.sqrt(np.sum(np.abs(W)**2) * d1 * d1)
prod, ent = norm2(prod), norm2(ent)
def evolve2(W, T):
    ph = np.exp(-1j * (k1[:, None]**2 + k1[None, :]**2) * T / 2)
    return np.fft.ifft2(np.fft.fft2(W) * ph)
def onebody(W):
    n = np.sum(np.abs(W)**2, axis=1) * d1
    dW = np.fft.ifft(1j * k1[:, None] * np.fft.fft(W, axis=0), axis=0)
    j = np.sum(np.imag(np.conj(W) * dW), axis=1) * d1
    return n, j
nA, jA = onebody(prod); nB, jB = onebody(ent)
nA2, jA2 = onebody(prod.T); nB2, jB2 = onebody(ent.T)            # fields of particle 2
dn0 = max(np.max(np.abs(nA - nB)), np.max(np.abs(nA2 - nB2))); dj0 = max(np.max(np.abs(jA - jB)), np.max(np.abs(jA2 - jB2)))
nAt, _ = onebody(evolve2(prod, 6.0)); nBt, _ = onebody(evolve2(ent, 6.0))
dnt = np.max(np.abs(nAt - nBt)); scale = np.max(nBt)
ok(f"QM8 entangled vs product state: at t = 0 the one-body densities and currents of both particles differ by at most {dn0:.1e}, {dj0:.1e}; at t = 6 the one-body "
   f"densities of particle 1 differ by {dnt:.3f} (peak {scale:.3f}): fringes only in the product state. A label medium in 3-space has no closed "
   "dynamics for entangled states; many-body quantum mechanics needs labels on configuration space or the quantised label field",
   dn0 < 1e-8 and dj0 < 1e-8 and dnt > 0.3 * scale)
print("ALL PASS")
