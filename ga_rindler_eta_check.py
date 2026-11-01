# 137: entanglement per area of a plane (the Rindler horizon) versus the sphere of doc 130, for the stiff label step (free massless scalar).
# Regulator of doc 130 (Srednicki 1993): a lattice of step a across the surface, the directions along the surface exact. For a plane the
# transverse momentum k is continuous and every k gives a 1D chain with mass k: S/A = (1/2pi) int k S_1(k a) dk / a^2.
# RE1 the half-line entanglement of a massive chain: S_1(k) -> (1/6) ln(1/k) for k -> 0 (c = 1, one end point); finite-size converged;
# RE2 the low entanglement spectrum of the half-line is equally spaced (as for the corner-transfer-matrix / lattice-boost form, Peschel);
#     only equidistance is tested: identifying the modular Hamiltonian with the boost (Bisognano-Wichmann) is known, not computed here;
# RE3 eta_plane = (1/2pi) int k S_1(k) dk in units 1/a^2;
# RE4 eta_sphere from the radial lattice (doc 130 JC8, recomputed with a larger lattice): S = s (R/a)^2 + c, eta_sphere = s/(4 pi);
#     the leading coefficients agree at the 1e-3 level (spread of the sphere fits over two lattices and three radius ranges, and the plane
#     integration grid, are the error estimate); the ln R coefficient is not determined by these fits;
# RE5 with a transverse lattice of the same step (cubic lattice) the plane coefficient differs: eta depends on the regulator, not on the shape;
# RE7 a harmonic medium H = (1/2) pi^2 + (c_s^2/2) phi K phi (kinetic term unchanged, potential matrix K -> c_s^2 K, same lattice):
#     X ~ K^-1/2, P ~ K^1/2, so X P and the entropy do not change; other quadratic Hamiltonians (anisotropic, other UV) are not covered; for dust
#     (c_s = 0) the mode has no ground state and no contribution is defined;
# RE8 a third surface with another mean curvature, the cylinder (radial lattice, angular momentum m and axial momentum k exact): on one
#     Gauss-Legendre grid in ln k the cylinder and its flat control (uniform chain, transverse momentum m/R) differ by a relative amount
#     that falls with R (convergence in cyl_convergence_eta.py): curvature enters only subleading terms; Jacobson's local horizons are planar
#     patches, whose coefficient is eta_plane (RE3);
# RE6 G: eta = 1/(4 G hbar) gives G = a^2/(4 eta_hat N): the lattice step a and the field count N replace G; G is not fixed without them.
# Run: ~/.venvs/jev/bin/python ga_rindler_eta_check.py
import numpy as np
from scipy.integrate import quad

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

def chain_half(kap, N=None, spectrum=False, c2=1.0):
    # Dirichlet chain of N sites, K = (2 + kap^2) on the diagonal, -1 off; region = the first N/2 sites (one entangling point in the bulk:
    # the far wall is at distance N/2 >> 1/kap, so the second end of the region does not contribute)
    if N is None:
        N = int(min(4000, max(200, 60 / kap)))
    K = c2 * (np.diag(np.full(N, 2 + kap**2)) - np.diag(np.ones(N - 1), 1) - np.diag(np.ones(N - 1), -1))
    w, V = np.linalg.eigh(K)
    X = 0.5 * (V / np.sqrt(w)) @ V.T; P = 0.5 * (V * np.sqrt(w)) @ V.T
    n = N // 2
    nu = np.sqrt(np.clip(np.linalg.eigvals(X[:n, :n] @ P[:n, :n]).real, 0.25, None))
    nu = np.sort(nu[nu > 0.5 + 1e-13])[::-1]
    S = np.sum((nu + 0.5) * np.log(nu + 0.5) - (nu - 0.5) * np.log(nu - 0.5))
    if spectrum:
        return S, 2 * np.arctanh(1 / (2 * nu))                         # nu = coth(eps/2)/2 -> eps
    return S

# ---------- RE1 ----------
ks = np.array([0.01, 0.02, 0.04])
S1 = np.array([chain_half(k) for k in ks])
slope = -np.polyfit(np.log(ks), S1, 1)[0]
conv = abs(chain_half(0.1, 600) - chain_half(0.1, 1200))
print("   S_1(k) at k = 0.01, 0.02, 0.04: " + ", ".join(f"{s:.4f}" for s in S1) + f"; size change at k = 0.1: {conv:.1e}")
ok(f"RE1 half-line entanglement of the massive chain: S_1 = -{slope:.4f} ln k + const for small k (1/6 = {1 / 6:.4f}: c = 1, one entangling "
   "point); converged in the chain length", abs(slope - 1 / 6) < 0.01 and conv < 1e-8)

# ---------- RE2 ----------
_, eps = chain_half(0.3, spectrum=True)
d = np.diff(np.sort(eps)[:6])
ok(f"RE2 entanglement spectrum of the half-line (k = 0.3): lowest levels {np.round(np.sort(eps)[:5], 3)}, spacings {np.round(d, 3)} equal to "
   f"{np.abs(d - d.mean()).max() / d.mean():.1e}: an equidistant spectrum, as expected for the lattice-boost form; the identification with "
   "the boost (Bisognano-Wichmann) is known and not tested here", np.abs(d - d.mean()).max() / d.mean() < 1e-3)

# ---------- RE3 plane ----------
kgrid = np.concatenate([np.geomspace(0.01, 0.5, 18), np.linspace(0.55, 8, 40), np.geomspace(8.5, 400, 40)])
Sk = np.array([chain_half(k) for k in kgrid])
# below k = 0.01 use the asymptote -(1/6) ln k + c0 fitted at the smallest points
c0 = Sk[0] + np.log(kgrid[0]) / 6
small = quad(lambda k: k * (-np.log(k) / 6 + c0), 0, kgrid[0])[0]
main = np.trapezoid(kgrid * Sk, kgrid)
# large k: S_1 decays as a power (k^-p ln k-like); fit p on the last decade and add the tail beyond k = 400
big = kgrid > 40
p_fit, lnA = np.polyfit(np.log(kgrid[big]), np.log(Sk[big]), 1)
tail = np.exp(lnA) * kgrid[-1]**(p_fit + 2) / -(p_fit + 2)                  # int_{kmax}^inf A k^(p+1) dk
eta_plane = (small + main + tail) / (2 * np.pi)
print(f"   S_1(8) = {Sk[np.searchsorted(kgrid, 8)]:.1e}, S_1(400) = {Sk[-1]:.1e}, decay S_1 ~ k^{p_fit:.2f}; pieces: k < 0.01 {small:.2e}, "
      f"main {main:.5f}, tail {tail:.1e}")
ok(f"RE3 plane (Rindler horizon) in the radial-lattice regulator: eta_plane = (1/2pi) int k S_1(k) dk = {eta_plane:.5f} / a^2 per field "
   f"(integrand decays as k^{p_fit + 1:.2f}, tail beyond k = 400 is {tail / main:.1e} of the total)",
   0.01 < eta_plane < 0.04 and p_fit + 2 < -0.5 and tail / main < 1e-2)

# ---------- RE4 sphere ----------
def ent_sphere(n, Nsites, lmax=6000):
    j = np.arange(1, Nsites + 1, dtype=float)
    S = 0.0
    for l in range(lmax + 1):
        K = np.zeros((Nsites, Nsites))
        K[np.arange(Nsites), np.arange(Nsites)] = (l * (l + 1) / j**2 + ((j + 0.5)**2 + (j - 0.5)**2) / j**2)
        K[0, 0] = l * (l + 1) + 2.25
        off = -(j[:-1] + 0.5)**2 / (j[:-1] * j[1:])
        K[np.arange(Nsites - 1), np.arange(1, Nsites)] = off
        K[np.arange(1, Nsites), np.arange(Nsites - 1)] = off
        w, V = np.linalg.eigh(K)
        Xm = 0.5 * (V / np.sqrt(w)) @ V.T; Pm = 0.5 * (V * np.sqrt(w)) @ V.T
        nu = np.sqrt(np.clip(np.linalg.eigvals(Xm[:n, :n] @ Pm[:n, :n]).real, 0.25, None))
        nu = nu[nu > 0.5 + 1e-12]
        Sl = np.sum((nu + 0.5) * np.log(nu + 0.5) - (nu - 0.5) * np.log(nu - 0.5))
        S += (2 * l + 1) * Sl
        if l > 50 and (2 * l + 1) * Sl < 1e-10 * S:
            break
    return S
ns = np.arange(10, 41, 5)
Rr = ns + 0.5
Sv = {Ns: np.array([ent_sphere(n, Ns) for n in ns]) for Ns in (100, 150)}
fits = []
for Ns in (100, 150):
    for lo, hi in ((10, 40), (20, 40), (10, 30)):
        m = (ns >= lo) & (ns <= hi)
        A_ = np.vstack([Rr[m]**2, np.log(Rr[m]), np.ones(m.sum())]).T
        c = np.linalg.lstsq(A_, Sv[Ns][m], rcond=None)[0]
        fits.append((Ns, lo, hi, 'ln', c[0] / (4 * np.pi), c[1]))
        fits.append((Ns, lo, hi, 'no ln', np.polyfit(Rr[m]**2, Sv[Ns][m], 1)[0] / (4 * np.pi), np.nan))
for n_, s_ in zip(ns, Sv[100]):
    print(f"   sphere n = {n_:2d}: S = {s_:9.3f}, S/(4 pi R^2) = {s_ / (4 * np.pi * (n_ + 0.5)**2):.5f}")
etas = np.array([f[4] for f in fits]); blog = np.array([f[5] for f in fits if f[3] == 'ln'])
eta_sph = etas.mean(); spread = (etas.max() - etas.min()) / eta_sph
# plane on a coarser grid as the integration error
kgrid2 = np.concatenate([np.geomspace(0.01, 0.5, 9), np.linspace(0.55, 8, 20), np.geomspace(8.5, 400, 20)])
Sk2 = np.array([chain_half(k) for k in kgrid2])
c02 = Sk2[0] + np.log(kgrid2[0]) / 6
big2 = kgrid2 > 40; p2, lA2 = np.polyfit(np.log(kgrid2[big2]), np.log(Sk2[big2]), 1)
eta_plane2 = (quad(lambda k: k * (-np.log(k) / 6 + c02), 0, kgrid2[0])[0] + np.trapezoid(kgrid2 * Sk2, kgrid2)
              + np.exp(lA2) * kgrid2[-1]**(p2 + 2) / -(p2 + 2)) / (2 * np.pi)
pl_err = abs(eta_plane2 - eta_plane) / eta_plane
rel = abs(eta_sph - eta_plane) / eta_plane
print(f"   sphere fits (lattice 100/150 sites, R ranges, with/without ln R): eta = {etas.min():.6f} ... {etas.max():.6f} (spread {spread:.1e}); "
      f"ln R coefficient {np.nanmin(blog):.3f} ... {np.nanmax(blog):.3f} (not determined); plane {eta_plane:.6f}, coarse grid {eta_plane2:.6f}")
ok(f"RE4 sphere in the same regulator: eta_sphere = {eta_sph:.5f} (spread of fits {spread:.1e}), plane {eta_plane:.5f} (grid change {pl_err:.1e}): "
   f"difference {rel:.1e}, within {rel / spread:.1f} of the extrapolation spread. In one regulator the area coefficient of the plane equals that of "
   "the sphere at the 1e-3 level; the ln R term is not determined by these fits", rel < 2e-3 and spread < 2e-3 and pl_err < 1e-3)

# ---------- RE5 another regulator ----------
# integral over the Brillouin zone: (1/(2pi)^2) int_{[-pi,pi]^2} S dq = average of S over the zone (zone area (2pi)^2)
def eta_cubic(M=40):
    q = (np.arange(M) + 0.5) * np.pi / M
    vals = [chain_half(np.sqrt(4 * np.sin(qx / 2)**2 + 4 * np.sin(qy / 2)**2)) for qx in q for qy in q]
    return np.mean(vals)                                                   # the quarter zone with midpoints = the full-zone average
ec1, ec2 = eta_cubic(24), eta_cubic(40)
print(f"   cubic lattice: eta = {ec1:.5f} (24^2 points), {ec2:.5f} (40^2 points)")
ok(f"RE5 with a transverse lattice of the same step (cubic lattice) the plane gives eta = {ec2:.5f}, not {eta_plane:.5f} ({(ec2 / eta_plane - 1) * 100:.1f} %, grid change {abs(ec2 - ec1) / ec2:.1e}): the coefficient "
   "depends on the regulator (known: area coefficients are not universal), while within one regulator it does not depend on the shape (RE4)",
   abs(ec2 - ec1) / ec2 < 0.002 and abs(ec2 - eta_plane) / eta_plane > 5 * abs(ec2 - ec1) / ec2 + 0.01)

# ---------- RE7 sound speed ----------
sv = [chain_half(0.2, c2=c) for c in (1.0, 0.1**2, 1e-3**2)]
ok(f"RE7 the half-line entropy at k = 0.2 for c_s = 1, 0.1, 0.001: {sv[0]:.10f}, {sv[1]:.10f}, {sv[2]:.10f}: independent of the stiffness, so "
   "a medium H = pi^2/2 + c_s^2 phi K phi/2 contributes the same eta per mode; other quadratic Hamiltonians are not covered; "
   "dust (c_s = 0) has no ground state of the mode and no defined contribution (sharpens doc 130 item (d))", max(sv) - min(sv) < 1e-9)


# ---------- RE8 cylinder against its flat control ----------
def S_rad(K, n):
    w, V = np.linalg.eigh(K)
    X = 0.5 * (V / np.sqrt(w)) @ V.T; P = 0.5 * (V * np.sqrt(w)) @ V.T
    nu = np.sqrt(np.clip(np.linalg.eigvals(X[:n, :n] @ P[:n, :n]).real, 0.25, None)); nu = nu[nu > 0.5 + 1e-12]
    return np.sum((nu + 0.5) * np.log(nu + 0.5) - (nu - 0.5) * np.log(nu - 0.5))
ktab = np.geomspace(1e-3, 80, 200)
Stab = np.array([chain_half(k) for k in ktab])
p_tab = np.polyfit(np.log(ktab[-20:]), np.log(Stab[-20:]), 1)[0]         # power-law tail beyond the table (no clamping)
def S1i(k):
    k = np.asarray(k, float)
    inside = np.exp(np.interp(np.log(np.minimum(k, ktab[-1])), np.log(ktab), np.log(Stab)))
    return np.where(k > ktab[-1], Stab[-1] * (k / ktab[-1])**p_tab, inside)
from numpy.polynomial.legendre import leggauss
xg, wg = leggauss(24)                                                      # Gauss-Legendre in ln k on [1e-3, 60]
la, lb = np.log(1e-3), np.log(60.0)
kc = np.exp(0.5 * (lb - la) * xg + 0.5 * (lb + la)); wc = 0.5 * (lb - la) * wg * kc
Nc = 120; jc = np.arange(1, Nc + 1, dtype=float)
bdc = ((jc + 0.5) + (jc - 0.5)) / jc; offc = -(jc[:-1] + 0.5) / np.sqrt(jc[:-1] * jc[1:])
def per_area(f, R):
    tot = 0.0
    for m in range(0, 6000):
        sm = np.sum(wc * np.array([f(m, k) for k in kc])) / np.pi
        tot += sm * (1 if m == 0 else 2)
        if m > 20 and sm < 1e-9 * tot:
            break
    return tot / (2 * np.pi * R)
dev8 = []
for n in (10, 20):
    R = n + 0.5
    fl = per_area(lambda m, k: S1i(np.hypot(m / R, k)), R)
    cy = per_area(lambda m, k: S_rad(np.diag(bdc + m**2 / jc**2 + k**2) + np.diag(offc, 1) + np.diag(offc, -1), n), R)
    dev8.append(cy / fl - 1)
    print(f"   cylinder R = {R}: per area {cy:.6f}, flat control {fl:.6f} (same k grid), relative difference {cy / fl - 1:.2e}")
ok(f"RE8 cylinder (mean curvature 1/2R, sphere 1/R, plane 0) against its flat control on one Gauss-Legendre grid in ln k: relative difference "
   f"{dev8[0]:.2e} at R = 10.5 and {dev8[1]:.2e} at R = 20.5, falling by a factor {dev8[0] / dev8[1]:.1f}; the convergence study "
   "(cyl_convergence_eta.py: grid 30 -> 60 points changes R = 20.5 by 1.5e-6, lattice 150 -> 250 sites by 1e-5; at R = 40.5 the difference "
   "is zero within 1.5e-5) shows that curvature enters only subleading terms on the shapes checked",
   abs(dev8[1]) < abs(dev8[0]) / 3 and abs(dev8[1]) < 1e-3)

# ---------- RE6 G ----------
a_over_lP = np.sqrt(4 * eta_plane)                                          # N eta_hat/a^2 = 1/(4 l_P^2): a = sqrt(4 N eta_hat) l_P
print(f"   one field: eta = 1/(4 l_P^2) needs a = {a_over_lP:.3f} l_P (doc 130: 0.306); N fields: a = {a_over_lP:.3f} sqrt(N) l_P")
ok(f"RE6 eta = N eta_hat/a^2 = 1/(4 G hbar) gives G hbar = a^2/(4 N eta_hat), eta_hat = {eta_plane:.4f} (radial lattice) or {ec2:.4f} "
   "(cubic): G is traded for the step a and the field count N, both outside the label axioms; with a = the step of axiom D the value is "
   "excluded by BBN (doc 112). G is not fixed by eta alone", abs(a_over_lP - 0.306) < 0.01)
