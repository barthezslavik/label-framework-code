# 135: can a compact label space (S^3, doc 134) replace axiom D (discrete labels, doc 91) and the self-interaction axiom (docs 107, 120)?
# CL1 the label number inside a sub-region is not an integer and varies continuously with the region: topology fixes only the total;
# CL2 the total degree is fixed under every smooth deformation inside a sector (LS1) and differs between sectors (degree 1, 2): within
#     a sector it cannot fluctuate, while axiom D needs Poisson counts
#     delta N ~ sqrt(N) in every region of the vacuum 4-volume (DL3), and D concerns the four vacuum labels, not the dust labels;
# CL3 size of a lump from the framework's media: E = m Q + kappa int n^2 has no minimum (the lump spreads), kappa < 0 collapses;
# CL4 with self-gravity E = kappa c3/R^3 - G M^2 c1/R has a minimum, but R* = sqrt(3 kappa c3/(G M^2 c1)) is set by the new constant kappa;
#     with the coupling beta of the dust-scalar sector the attraction is G(1 + 2 beta^2): the same form, weaker than needed by ~1e38;
# CL5 the size required by s/m = 0.1 cm^2/g IF the lumps scatter as identical hard spheres (sigma = pi (2R)^2 = 4 pi R^2; the scattering of
#     lumps is not computed, the model is assumed): R = sqrt((s/m) m/(4 pi)) for m = 1 MeV - 1 TeV;
# CL6 the size term of the BPS Skyrme model is a potential V(X) of the labels; a volume-preserving relabelling changes its energy:
#     V breaks SDiff of label space, i.e. it is not an object of the framework (labels are names).
# Run: ~/.venvs/jev/bin/python ga_compact_labels_check.py
import numpy as np
from scipy.integrate import quad

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

prof = lambda r: np.pi * np.exp(-r)
dens_r = lambda r: np.sin(prof(r))**2 * np.pi * np.exp(-r) / (2 * np.pi**2) * 4 * np.pi   # radial density of the degree, total 1

# ---------- CL1 sub-region counts ----------
Qs = [quad(dens_r, 0, R)[0] for R in (0.5, 1.0, 2.0, 4.0, 50.0)]
print("   degree inside radius 0.5, 1, 2, 4, 50: " + ", ".join(f"{q:.4f}" for q in Qs))
ok("CL1 the label number inside a ball is not an integer and grows continuously with the radius; only the total is 1",
   all(0 < q < 1 for q in Qs[:4]) and abs(Qs[-1] - 1) < 1e-9 and np.all(np.diff(Qs) > 0))

# ---------- CL2 no fluctuation of the total ----------
deg = lambda f0: 4 * np.pi * quad(lambda r: np.sin(f0 * np.exp(-r))**2 * f0 * np.exp(-r), 0, np.inf, limit=200)[0] / (2 * np.pi**2)
d1, d1b, d2 = deg(np.pi), 4 * np.pi * quad(lambda r: np.sin(np.pi * np.exp(-r / 0.4))**2 * np.pi * np.exp(-r / 0.4) / 0.4, 0, np.inf)[0] / (2 * np.pi**2), deg(2 * np.pi)
ok(f"CL2 degree {d1:.4f} (size 1), {d1b:.4f} (size 0.4) in the sector f(0) = pi and {d2:.4f} in the sector f(0) = 2 pi: inside a sector the "
   "total is a superselected integer with delta Q = 0 (the non-spherical deformation of LS1 keeps it too). Axiom D (DL3) needs Poisson counts "
   "delta N ~ sqrt(N) in every region of the vacuum 4-volume, and it concerns the four vacuum labels, not the three dust labels: compactness "
   "of the dust label space does not replace D", abs(d1 - 1) < 1e-9 and abs(d1b - 1) < 1e-9 and abs(d2 - 2) < 1e-9)

# ---------- CL3 dust + stiff step: no size ----------
# hedgehog of size a: Q = 1 for every a, int n^2 d^3x = c3 / a^3 (c3 computed)
c3 = quad(lambda r: (np.sin(prof(r))**2 * np.pi * np.exp(-r) / r**2)**2 * 4 * np.pi * r**2, 1e-9, 60)[0]
a = np.linspace(0.2, 5, 200)
E = lambda kap: 1.0 + kap * c3 / a**3                                   # m = 1
ok(f"CL3 E(a) = m Q + kappa c3/a^3 (c3 = {c3:.3f}): monotone in a for kappa > 0 (decreasing: the lump spreads) and for kappa < 0 "
   "(increasing toward a -> 0: collapse); no size from dust and the stiff step", np.all(np.diff(E(1.0)) < 0) and np.all(np.diff(E(-1.0)) > 0))

# ---------- CL4 self-gravity fixes a size only through the new constant kappa ----------
# Newtonian self-energy of the density profile, c1 from the enclosed mass (M = m Q = 1, unit size)
rr = np.linspace(1e-6, 60, 60001)
rho = np.sin(prof(rr))**2 * np.pi * np.exp(-rr) / rr**2 / (2 * np.pi**2)  # mass density per unit M at a = 1
Menc = np.cumsum(rho * 4 * np.pi * rr**2) * (rr[1] - rr[0])
c1 = np.sum(Menc * rho * 4 * np.pi * rr) * (rr[1] - rr[0])             # W = -G M^2 c1 / a
def Rstar(kap, G=1.0, M=1.0):
    return np.sqrt(3 * kap * c3 / (G * M**2 * c1))
Rk = [Rstar(k) for k in (1e-2, 1.0, 1e2)]
ok(f"CL4 with self-gravity E = kappa c3/a^3 - G M^2 c1/a (c1 = {c1:.3f}) has a minimum at a* = sqrt(3 kappa c3/(G M^2 c1)): "
   f"{Rk[0]:.3f}, {Rk[1]:.3f}, {Rk[2]:.3f} for kappa = 0.01, 1, 100: the size is set by the new constant kappa (a label), not by the framework",
   abs(Rk[2] / Rk[0] - 100) < 1e-9)
# a nuclear-size lump of 1 GeV held by gravity: required kappa from a* = 1 fm, in units hbar = c = 1
GeV_fm = 5.068                                                          # 1 GeV^-1 = 0.1973 fm -> 1 fm = 5.068 GeV^-1
G_N = 6.709e-39                                                         # GeV^-2
kap_req = (1.0 * GeV_fm)**2 * G_N * 1.0**2 * c1 / (3 * c3)              # GeV^-5 ... dimensionful; compare with the stiff scale 1/(m a^3)
kap_nat = (1.0 * GeV_fm)**3 / 1.0                                       # kappa giving pressure ~ rho at a = 1 fm (sound speed ~ 1)
print(f"   1 GeV lump of 1 fm held by gravity needs kappa = {kap_req:.1e}, i.e. {kap_req / kap_nat:.1e} of the c_s ~ 1 stiffness")
ok(f"CL4b gravity (and the dust-scalar attraction G(1 + 2 beta^2), beta < 0.06) can hold a 1 GeV lump at 1 fm only with a stiffness "
   f"{kap_req / kap_nat:.0e} of the natural one: gravity is not what sizes a particle-scale lump", kap_req / kap_nat < 1e-35)

# ---------- CL5 the size the self-interaction axiom needs ----------
s_over_m = 0.1                                                          # cm^2/g
GeV_g = 1.783e-24
for m_GeV in (1e-3, 1.0, 1e3):
    R_cm = np.sqrt(s_over_m * m_GeV * GeV_g / (4 * np.pi))
    print(f"   m = {m_GeV:g} GeV: hard-sphere radius R = {R_cm * 1e13:.3g} fm")
R1 = np.sqrt(s_over_m * GeV_g / (4 * np.pi)) * 1e13
ok(f"CL5 s/m = 0.1 cm^2/g for identical hard spheres (sigma = 4 pi R^2, an assumed scattering model) needs R = {R1:.2f} fm at 1 GeV, "
   "R ~ m^(1/2): a nuclear-size lump at GeV mass. The scale pair (m, R) is one more label; compact labels give an extended lump, not the "
   "number; the contact form and velocity independence belong to the assumed model", 0.5 < R1 < 5)

# ---------- CL6 the BPS size term breaks relabelling ----------
# V = 1 - X^0 vanishes at the vacuum label (1,0,0,0). Relabelling: rotation in the (X^0, X^1) plane by angle t g(rho), rho = X2^2 + X3^2,
# g = 0 near rho = 0 (identity near the vacuum label, the gauge kind of doc 134); g depends only on a quantity the rotation preserves,
# so the generating field is divergence-free on S^3 and the map preserves the label volume.
def V_energy(t, M=(200, 60, 120)):
    s_ = (np.arange(M[0]) + 0.5) / M[0]; th = (np.arange(M[1]) + 0.5) * np.pi / M[1]; ph = (np.arange(M[2]) + 0.5) * 2 * np.pi / M[2]
    S, T, P = np.meshgrid(s_, th, ph, indexing='ij')
    R = S / (1 - S)
    w = R**2 * np.sin(T) / (1 - S)**2 / M[0] * (np.pi / M[1]) * (2 * np.pi / M[2])
    f = prof(R)
    X = np.stack([np.cos(f), np.sin(f) * np.sin(T) * np.cos(P), np.sin(f) * np.sin(T) * np.sin(P), np.sin(f) * np.cos(T)])
    rho = X[2]**2 + X[3]**2
    g = np.clip((rho - 0.2) / 0.2, 0, 1) * np.clip((0.8 - rho) / 0.2, 0, 1)
    ang = t * g
    X0r = np.cos(ang) * X[0] - np.sin(ang) * X[1]
    return np.sum((1 - X0r) * w)
E0, Et = V_energy(0.0), V_energy(1.0)
ok(f"CL6 the size term V = 1 - X^0 of the BPS model: energy {E0:.3f} before and {Et:.3f} after a volume-preserving relabelling that is the "
   "identity near the vacuum label: V depends on the names of the labels and breaks SDiff; it is not an object of the framework",
   abs(Et - E0) / E0 > 0.02)
