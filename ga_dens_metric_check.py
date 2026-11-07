# 148/149: QM part II (paper12, new sections), step 2. Metrics on densities of the configuration space under the physical relabelling
# group Diff(M)^N x S_N instead of Diff(M^N). Numerics on the torus: M = S^1 (one coordinate per particle), N = 2 and N = 3; the
# argument does not use the dimension of M. Densities and tangent vectors (zero integral) are smooth analytic functions; a
# diffeomorphism acts by pushforward, rho'(y) = rho(x(y)) |dx/dy|.
# DM1 local metrics without derivatives, int rho^a (delta rho)^2: invariant under product diffeomorphisms only for a = -1 (Fisher-Rao);
# DM2 local terms with derivatives, int (d_1 delta rho)^2 / rho^k, k = 1, 2, 3, and int (d_1 delta rho)(d_2 delta rho) / rho: not
#     invariant under product diffeomorphisms (no metric on M is given);
# DM3 the Fisher-Rao metric of a marginal, int (delta rho_1)^2 / rho_1 with rho_1 = int rho dx_2, is invariant under product
#     diffeomorphisms but not under a volume-preserving map that mixes the particles, (x1, x2) -> (x1 + b sin x2, x2); the full
#     Fisher-Rao metric is invariant under both;
# DM4 S_N-symmetric metrics built from Fisher-Rao metrics of marginals over subsets of particles: for N = 2 and N = 3 the quadratic
#     forms G_k = sum over subsets S with |S| = k of FR(rho_S), k = 1..N, are linearly independent (rank N on random tangent vectors):
#     the invariant family has at least N free coefficients, against one for Diff(M^N) (Bauer-Bruveris-Michor 2016);
# DM5 the Wasserstein kinetic term int rho |v|^2 is not invariant under product diffeomorphisms: it needs the metric of M.
# Run: ~/.venvs/jev/bin/python ga_dens_metric_check.py
import numpy as np
from itertools import combinations

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

n = 192
g = 2 * np.pi * np.arange(n) / n; dA = (2 * np.pi / n)**2
X1, X2 = np.meshgrid(g, g, indexing='ij')
k = np.fft.fftfreq(n, 1 / n)

def rho_f(x1, x2):
    return np.exp(np.cos(x1) + 0.5 * np.sin(x1 + x2) + 0.3 * np.cos(2 * x2) + 0.2 * np.sin(x1 - 2 * x2))
Z = rho_f(X1, X2).sum() * dA
def drho_f(x1, x2):
    h = np.sin(x1) * np.cos(x2) + 0.4 * np.cos(2 * x1 + x2) + 0.3 * np.sin(x2)
    return rho_f(x1, x2) / Z * h
c0 = drho_f(X1, X2).sum() * dA
def drho0(x1, x2):                                                  # zero total integral
    return drho_f(x1, x2) - c0 * rho_f(x1, x2) / Z
def rho0(x1, x2):
    return rho_f(x1, x2) / Z

def phi_(x, a):                                                     # circle diffeomorphism with several harmonics (|a| <= 0.6)
    return x + a * np.sin(x + 0.4) + 0.3 * a * np.cos(2 * x)
def dphi_(x, a):
    return 1 + a * np.cos(x + 0.4) - 0.6 * a * np.sin(2 * x)
def inv_phi(y, a):
    x = y.copy()
    for _ in range(80):
        x -= (phi_(x, a) - y) / dphi_(x, a)
    return x

def push_product(f, a1, a2):
    x1 = inv_phi(X1, a1); x2 = inv_phi(X2, a2)
    assert np.abs(phi_(x1, a1) - X1).max() < 1e-12 and np.abs(phi_(x2, a2) - X2).max() < 1e-12
    J = 1 / (dphi_(x1, a1) * dphi_(x2, a2))
    return f(x1, x2) * J
def push_mix(f, b):                                                 # (x1, x2) -> (x1 + b sin x2, x2), unit Jacobian
    return f(X1 - b * np.sin(X2), X2)

def d(f, axis):
    K = k[:, None] if axis == 0 else k[None, :]
    return np.real(np.fft.ifft2(1j * K * np.fft.fft2(f)))

def G_pow(r, dr, a):
    return (r**a * dr**2).sum() * dA
def FR(r, dr):
    return G_pow(r, dr, -1)
def FR_marg(r, dr, axis):
    m = r.sum(axis=1 - axis) * (2 * np.pi / n); dm = dr.sum(axis=1 - axis) * (2 * np.pi / n)
    return (dm**2 / m).sum() * (2 * np.pi / n)

r, dr = rho0(X1, X2), drho0(X1, X2)
rp, drp = push_product(rho0, 0.6, -0.45), push_product(drho0, 0.6, -0.45)
rm, drm = push_mix(rho0, 0.7), push_mix(drho0, 0.7)
assert abs(rp.sum() * dA - 1) < 1e-10 and abs(drp.sum() * dA) < 1e-10

# ---------- DM1 ----------
rel = {a: abs(G_pow(rp, drp, a) / G_pow(r, dr, a) - 1) for a in (-2.0, -1.5, -1.0, -0.5, 0.0)}
ok("DM1 int rho^a (delta rho)^2 under a product diffeomorphism, relative change: "
   + ", ".join(f"a={a:+.1f}: {v:.1e}" for a, v in rel.items()) + "; invariant only at a = -1 (Fisher-Rao)",
   rel[-1.0] < 1e-9 and all(v > 1e-3 for a, v in rel.items() if a != -1.0))

# ---------- DM2 ----------
def T(rr, ddr, kind):
    if kind == 'mix':
        return (d(ddr, 0) * d(ddr, 1) / rr).sum() * dA
    return (d(ddr, 0)**2 / rr**kind).sum() * dA
rel2 = {kk: abs(T(rp, drp, kk) / T(r, dr, kk) - 1) for kk in (1, 2, 3, 'mix')}
ok("DM2 derivative terms under a product diffeomorphism, relative change: "
   + ", ".join(f"{kk}: {v:.1e}" for kk, v in rel2.items()) + "; none invariant", all(v > 1e-3 for v in rel2.values()))

# ---------- DM3 ----------
fr_full = [FR(r, dr), FR(rp, drp), FR(rm, drm)]
fr_m1 = [FR_marg(r, dr, 0), FR_marg(rp, drp, 0), FR_marg(rm, drm, 0)]
ok(f"DM3 full Fisher-Rao: {fr_full[0]:.6f} / product {fr_full[1]:.6f} / mixing {fr_full[2]:.6f}; marginal Fisher-Rao of particle 1: "
   f"{fr_m1[0]:.6f} / product {fr_m1[1]:.6f} / mixing {fr_m1[2]:.6f}: the marginal metric is invariant under the subgroup only",
   abs(fr_full[1] / fr_full[0] - 1) < 1e-9 and abs(fr_full[2] / fr_full[0] - 1) < 1e-9
   and abs(fr_m1[1] / fr_m1[0] - 1) < 1e-9 and abs(fr_m1[2] / fr_m1[0] - 1) > 1e-3)

# ---------- DM4 linear independence of the S_N-symmetric marginal family ----------
rng = np.random.default_rng(5)
def family_rank(N, m=16):
    shp = (m,) * N; h = 2 * np.pi / m
    grids = np.meshgrid(*([2 * np.pi * np.arange(m) / m] * N), indexing='ij')
    s = sum(np.cos(gr + 0.3 * i) for i, gr in enumerate(grids)) + 0.4 * np.sin(sum(grids))
    rr = np.exp(0.6 * s); rr /= rr.sum() * h**N
    rows = []
    for _ in range(3 * N + 4):
        c = rng.normal(size=(5, N))
        hh = sum(c[j, i] * np.cos((j + 1) * grids[i] + c[j, (i + 1) % N]) for j in range(5) for i in range(N))
        hh += 0.5 * rng.normal() * np.sin(sum(grids))
        dd = rr * (hh - (rr * hh).sum() * h**N)
        vals = []
        for kk in range(1, N + 1):
            tot = 0.0
            for S in combinations(range(N), kk):
                ax = tuple(i for i in range(N) if i not in S)
                mS = rr.sum(axis=ax) * h**len(ax) if ax else rr
                dS = dd.sum(axis=ax) * h**len(ax) if ax else dd
                tot += (dS**2 / mS).sum() * h**kk
            vals.append(tot)
        rows.append(vals)
    A = np.array(rows)
    sv = np.linalg.svd(A / np.abs(A).max(axis=0), compute_uv=False)
    return int((sv > 1e-8 * sv[0]).sum()), sv
r2, sv2 = family_rank(2, 32); r3, sv3 = family_rank(3, 16)
ok(f"DM4 S_N-symmetric marginal Fisher-Rao forms G_1..G_N are linearly independent: rank {r2} for N = 2, {r3} for N = 3 "
   f"(smallest singular values {sv2[-1]:.1e}, {sv3[-1]:.1e}): at least N free coefficients under Diff(M)^N x S_N, one under Diff(M^N)",
   r2 == 2 and r3 == 3)

# ---------- DM5 Wasserstein needs the metric of M ----------
# tangent vector delta rho = -div(rho v) with v = grad phi; kinetic term int rho |v|^2 depends on the metric: compare with the pushforward
phi = lambda x1, x2: np.sin(x1) + 0.5 * np.cos(x2 + 0.3)
def W(rr, ph):
    return (rr * (d(ph, 0)**2 + d(ph, 1)**2)).sum() * dA
# push the flow: the pushed velocity potential is phi(x(y)); with the flat metric on the image the kinetic term changes
x1 = inv_phi(X1, 0.6); x2 = inv_phi(X2, -0.45)
w0 = W(r, phi(X1, X2)); w1 = W(rp, phi(x1, x2))
ok(f"DM5 Wasserstein kinetic term int rho |grad phi|^2 with the flat metric: {w0:.5f} before, {w1:.5f} after a product diffeomorphism: "
   f"not invariant, the kinetic term needs the metric of M", abs(w1 / w0 - 1) > 1e-3)

print("ALL PASS")
