# Does multistreaming (24) source the shear level of the frame (23)? papers/restart/25-multistream-shear.md, M1-M4.
# Run: uv run --with sympy --with numpy python ga_multistream_check.py
import numpy as np
import sympy as sp
from ga_core import ok

X = sp.symbols('t x y z')
eta = sp.diag(1, -1, -1, -1)
eps = sp.symbols('epsilon')
Phi = sp.Function('Phi')(*X[1:]); Psi = sp.Function('Psi')(*X[1:])
g = sp.diag(1 + 2 * eps * Phi, -(1 - 2 * eps * Psi), -(1 - 2 * eps * Psi), -(1 - 2 * eps * Psi))   # static weak field, two potentials

def linear_curvature(g):
    gi = g.inv()
    Gm = [[[sum(gi[r, s] * (sp.diff(g[s, m], X[k]) + sp.diff(g[s, k], X[m]) - sp.diff(g[m, k], X[s])) for s in range(4)) / 2
            for k in range(4)] for m in range(4)] for r in range(4)]
    def Riem(r, s, m, k):
        return sp.diff(Gm[r][k][s], X[m]) - sp.diff(Gm[r][m][s], X[k]) + sum(Gm[r][m][l] * Gm[l][k][s] - Gm[r][k][l] * Gm[l][m][s] for l in range(4))
    lin = lambda e: sp.simplify(sp.diff(e, eps).subs(eps, 0))
    Ric = sp.Matrix(4, 4, lambda s, k: lin(sum(Riem(r, s, r, k) for r in range(4))))
    Rs = sum(eta[i, i] * Ric[i, i] for i in range(4))
    Ein = sp.simplify(Ric - Rs * eta / 2)
    Rl = lambda a_, b_, c_, d_: lin(sum(g[a_, r] * Riem(r, b_, c_, d_) for r in range(4)))
    def Weyl(a_, b_, c_, d_):
        return sp.simplify(Rl(a_, b_, c_, d_) - (eta[a_, c_] * Ric[b_, d_] - eta[a_, d_] * Ric[b_, c_] - eta[b_, c_] * Ric[a_, d_] + eta[b_, d_] * Ric[a_, c_]) / 2
                           + Rs * (eta[a_, c_] * eta[b_, d_] - eta[a_, d_] * eta[b_, c_]) / 6)
    return Ein, Weyl
Ein, Weyl = linear_curvature(g)
lap = lambda f: sum(sp.diff(f, x, 2) for x in X[1:])
D = Phi - Psi; S = Phi + Psi
tr = sum(Ein[i, i] for i in (1, 2, 3))
trless = sp.Matrix(3, 3, lambda i, j: sp.simplify(Ein[i + 1, j + 1] - (tr / 3 if i == j else 0)))
target = sp.Matrix(3, 3, lambda i, j: sp.diff(D, X[i + 1], X[j + 1]) - (lap(D) / 3 if i == j else 0))
c_ = sp.simplify(trless[0, 1] / target[0, 1])
# M1 traceless spatial Einstein = c (d_i d_j - delta lap/3)(Phi - Psi): anisotropic stress pi_ij is the source of the slip Phi - Psi
ok(f"M1 linear G_ij (traceless) = {c_} * (d_i d_j - delta lap/3)(Phi - Psi); G_00 = {sp.simplify(Ein[0, 0])}: "
   "Einstein eq. -> (Phi - Psi) is sourced only by the anisotropic stress",
   sp.simplify(trless - c_ * target) == sp.zeros(3) and c_ != 0)
# M2 linear Weyl C_0101 = k (d_x^2 - lap/3)(Phi + Psi): the slip changes the tidal (shear) field
targW = sp.diff(S, X[1], 2) - lap(S) / 3
kW = sp.simplify(Weyl(0, 1, 0, 1) / targW)
ok(f"M2 linear Weyl C_0101 = {kW} * (d_x^2 - lap/3)(Phi + Psi): the tidal field sees Phi + Psi, so a slip Phi != Psi changes it",
   kW.free_symbols == set() and kW != 0)

# M3 multistream gives anisotropic stress, single stream none (1D toy of 24 K3: x = q - t sin q). Dispersion only along x ->
#    pi_ij = P diag(1, 0, 0) - P/3 delta = P diag(2/3, -1/3, -1/3) (traceless part), nonzero iff P > 0.
def P_xx(t, x, Nq=200001):
    q = np.linspace(-np.pi, np.pi, Nq); xs = q - t * np.sin(q) - x
    idx = np.where(np.sign(xs[:-1]) != np.sign(xs[1:]))[0]
    st = []
    for i in idx:
        qi = q[i] - xs[i] * (q[i + 1] - q[i]) / (xs[i + 1] - xs[i])
        st.append((1 / abs(1 - t * np.cos(qi)), -np.sin(qi)))
    rho = sum(r for r, _ in st); vb = sum(r * v for r, v in st) / rho
    return sum(r * (v - vb)**2 for r, v in st)
Pb, Pa = P_xx(0.5, 0.3), P_xx(2.0, 0.3)
pi_a = Pa * np.diag([1, 0, 0]) - Pa / 3 * np.eye(3)
ok(f"M3 single stream (t = 0.5): P_xx = {Pb:.1e}, anisotropic stress 0; multistream (t = 2): P_xx = {Pa:.3f}, traceless pi = {np.round(np.diag(pi_a), 3)}",
   Pb < 1e-12 and abs(np.trace(pi_a)) < 1e-12 and np.abs(pi_a).max() > 0.1)

# M4 order of magnitude (restoring c): pi ~ rho sigma^2 and lap Phi ~ 4 pi G rho give (Phi - Psi)/Phi ~ sigma^2/c^2
cl = 2.998e5
for name, sig in (("galaxy halo", 200.0), ("cluster", 1000.0)):
    print(f"   {name}: sigma = {sig:.0f} km/s -> slip (Phi - Psi)/Phi ~ sigma^2/c^2 = {(sig / cl)**2:.1e}")
ok("M4 slip from multistreaming ~ sigma^2/c^2: 4e-7 (galaxies) to 1e-5 (clusters) — a definite but tiny shear contribution",
   abs((200 / cl)**2 - 4.45e-7) < 1e-8 and abs((1000 / cl)**2 - 1.11e-5) < 1e-7)
