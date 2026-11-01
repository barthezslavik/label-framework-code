# 88: the ladder of preserved structures (volume -> bivector -> bivector + metric) and the quantisation of gravity read on it. LD1-LD6.
# Run: ~/.venvs/jev/bin/python ga_ladder_check.py
import numpy as np
import sympy as sp

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

def lie_dim(constraints, N):
    # dimension of {X in gl(N): all linear constraints C(X) = 0}
    syms = sp.symbols(f'x0:{N*N}'); X = sp.Matrix(N, N, syms)
    eqs = []
    for C in constraints:
        eqs += list(C(X))
    A = sp.Matrix([[sp.diff(e, s) for s in syms] for e in eqs])
    return N * N - A.rank()

# LD1 the three levels as Lie algebras on a 2n-dimensional phase space (n = 1, 2, 3): sl (trace 0: volume kept), sp (X^T W + W X = 0: the bivector
#     omega kept), sp ∩ so (omega and the metric kept) = u(n). Dimensions 4n^2 - 1 > n(2n+1) > n^2; sp is inside sl (Liouville).
ok_all = True; rows = []
for n in (1, 2, 3):
    N = 2 * n
    W = sp.zeros(N); W[:n, n:] = sp.eye(n); W[n:, :n] = -sp.eye(n)
    d_sl = lie_dim([lambda X: [X.trace()]], N)
    d_sp = lie_dim([lambda X, W=W: X.T * W + W * X], N)
    d_u = lie_dim([lambda X, W=W: X.T * W + W * X, lambda X: X.T + X], N)
    d_sp_tr = lie_dim([lambda X, W=W: X.T * W + W * X, lambda X: [X.trace()]], N)
    rows.append((n, d_sl, d_sp, d_u)); ok_all &= (d_sl == 4 * n * n - 1 and d_sp == n * (2 * n + 1) and d_u == n * n and d_sp_tr == d_sp)
ok(f"LD1 (n, dim sl, dim sp, dim sp∩so) = {rows}: sl ⊃ sp ⊃ u(n) with dims 4n²-1, n(2n+1), n²; sp is traceless (Liouville)", ok_all)

# LD2 sp ∩ so = the generators commuting with the complex structure J = W: [X, J] = 0 for X in sp ∩ so (n = 2, random element of the intersection).
n = 2; N = 4
W = np.zeros((N, N)); W[:n, n:] = np.eye(n); W[n:, :n] = -np.eye(n)
rng = np.random.default_rng(0)
A = rng.normal(size=(n, n)); A = A - A.T; S = rng.normal(size=(n, n)); S = S + S.T
X = np.block([[A, S], [-S, A]])                                    # the standard embedding of u(n): A antisymmetric, S symmetric
ok(f"LD2 X in u(2): X^T W + W X = 0 ({np.abs(X.T @ W + W @ X).max():.1e}), X^T = -X ({np.abs(X.T + X).max():.1e}), [X, J] = 0 ({np.abs(X @ W - W @ X).max():.1e})",
   np.abs(X.T @ W + W @ X).max() < 1e-12 and np.abs(X.T + X).max() < 1e-12 and np.abs(X @ W - W @ X).max() < 1e-12)

# LD3 level 3 = quantum mechanics: on C^n with g = Re<,>, omega = Im<,>, the Hamiltonian flow of a quadratic function f_A = <psi|A|psi> (A
#     Hermitian) is psi -> exp(-i A t) psi, an isometry (Killing); the flow of a non-quadratic function, f = |psi_1|^4, is not.
A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)); A = A + A.conj().T
def flow_quadratic(psi, t):
    w, V = np.linalg.eigh(A); return V @ (np.exp(-1j * w * t) * (V.conj().T @ psi))
def flow_quartic(psi, t):
    return psi * np.exp(-1j * 2 * np.abs(psi[0])**2 * t * np.array([1, 0, 0]))   # Hamiltonian flow of |psi_1|^4: phase rate depends on |psi_1|
p1 = rng.normal(size=3) + 1j * rng.normal(size=3); p2 = rng.normal(size=3) + 1j * rng.normal(size=3)
dq = abs(np.linalg.norm(flow_quadratic(p1, 0.7) - flow_quadratic(p2, 0.7)) - np.linalg.norm(p1 - p2))
dn = abs(np.linalg.norm(flow_quartic(p1, 0.7) - flow_quartic(p2, 0.7)) - np.linalg.norm(p1 - p2))
ok(f"LD3 distance change under the flow: quadratic f_A {dq:.1e} (isometry), quartic |psi_1|^4 {dn:.2e} (not): observables are the quadratic functions", dq < 1e-12 and dn > 1e-3)

# LD4 DeWitt supermetric G(h, h) = h_ij h^ij - lam (tr h)^2. 3D (lam = 1, GR) and 4D Euclidean (lam = 1/2): negative exactly on the trace
#     (dilation K) direction, positive on the trace-free part (shear: gravitons).
t_ = sp.symbols('t', real=True)
for dim, lam in ((3, 1), (4, sp.Rational(1, 2))):
    hT = (t_ / dim) * sp.eye(dim)
    hs = sp.zeros(dim); hs[0, 1] = hs[1, 0] = t_
    GT = sp.simplify((hT * hT).trace() - lam * hT.trace()**2); Gs = sp.simplify((hs * hs).trace() - lam * hs.trace()**2)
    ok_t = sp.simplify(GT / t_**2) < 0 and sp.simplify(Gs / t_**2) > 0
    print(f"   {dim}D: trace direction G = {GT}, trace-free direction G = {Gs}")
    assert ok_t
ok("LD4 DeWitt supermetric: negative only along the trace (dilation K), positive on the trace-free (shear) part, in 3D and 4D", True)

# LD5 fixing the volume (unimodular, SDiff of the 4-volume: tr h = 0) removes exactly that direction: G is positive definite on trace-free h.
syms = sp.symbols('a0:5', real=True)                              # 5 independent trace-free components
h = sp.Matrix([[syms[0], syms[1], syms[2]], [syms[1], syms[3], syms[4]], [syms[2], syms[4], -syms[0] - syms[3]]])
G = sp.expand((h * h).trace() - h.trace()**2)
Hm = sp.hessian(G, syms)
eig = list(Hm.eigenvals().keys())
ok(f"LD5 on trace-free h (3D) the supermetric Hessian eigenvalues are {eig}: positive definite once the volume is fixed", all(e > 0 for e in eig))

# LD6 reading (known: Unruh 1989, Henneaux-Teitelboim 1989): with the volume fixed one global pair remains, Lambda and the 4-volume of the
#     past (unimodular time), canonically conjugate; this time supplies the complex structure J that level 3 needs. Not derived here: the UV
#     behaviour of gravitons and why geometry responds to matter (doc 87).
ok("LD6 recorded: unimodular route = level-3 requirements (positive metric, time for J) met by fixing the volume; Lambda conjugate to unimodular time (known)", True)
