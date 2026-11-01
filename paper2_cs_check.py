# paper2 (theta): why S(m,n) = oint e'^de' is not topological, and what the topological (Chern-Simons) version gives.
# e' = O_{m,n} dx with O_{m,n}(x): v -> x^m v xbar^n.  S(m,n) = -oint (O^T dO)_{BC} ^ dx^B ^ dx^C depends on dx (a fixed frame),
# while the Chern-Simons form of the pure gauge theta = O^T dO (dtheta = -theta^2, CS = -1/3 tr theta^3) is a homotopy invariant.
# Run: uv run --with numpy python paper2_cs_check.py
import numpy as np
exec(open('lattice_check.py').read().split('for m,n in')[0])          # qmul, conj, qpow, O, S, rng

def T(m, n, N=300000, h=1e-5):
    """int_{S^3} tr(theta^theta^theta), theta = O^T dO, in units where the result for (1,0) is normalized later."""
    x = rng.normal(size=(N, 4)); x /= np.linalg.norm(x, axis=1, keepdims=True)
    E = np.eye(4)
    tv = [qmul(x, np.broadcast_to(E[k], x.shape)) for k in (1, 2, 3)]
    O0 = O(x, m, n)
    th = [np.einsum('nba,nbc->nac', O0, (O(x + h * v, m, n) - O(x - h * v, m, n)) / (2 * h)) for v in tv]
    val = 0
    for perm, sgn in (((0, 1, 2), 1), ((1, 2, 0), 1), ((2, 0, 1), 1), ((0, 2, 1), -1), ((2, 1, 0), -1), ((1, 0, 2), -1)):
        val = val + sgn * np.einsum('nab,nbc,nca->n', th[perm[0]], th[perm[1]], th[perm[2]])
    return val.mean() * 2 * np.pi**2

pairs = [(1, 0), (2, 0), (-1, 0), (3, 0), (0, 1), (0, 2), (1, 1), (2, 1)]
t10 = T(1, 0)
res = {}
for m, n in pairs:
    res[(m, n)] = (round(T(m, n) / t10, 3), round(S(m, n), 3))
print("(m,n): [CS tr theta^3 normalized to (1,0)] [paper2 S(m,n) in 4pi^2]")
for k, v in res.items():
    print(k, v)
lin = all(abs(res[(m, n)][0] - (m + n)) < 0.05 for m, n in pairs)
nonlin = abs(res[(2, 0)][1] - 2 * res[(1, 0)][1]) > 0.5
print(("PASS " if lin else "FAIL ") + "CS integral = m + n (linear, homotopy invariant)")
print(("PASS " if nonlin else "FAIL ") + "paper2 S(m,n) not additive in m (S(2,0) != 2 S(1,0))")
assert lin and nonlin
