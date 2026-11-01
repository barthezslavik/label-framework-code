# 66: Solar System test of the self-dual 'simple' transition (nu_1, mu = x/(1+x); 33, 37, base E3). QUMOND with the Galactic external field:
# the anomalous potential has a quadrupole Phi = -GM/r - (Q2/2) x^i x^j (e_i e_j - delta_ij/3) near the Sun (Blanchet-Novak 2011).
# Literature (verified 2026-10-20): Hees, Folkner, Jacobson, Park, PRD 89, 102002 (2014), arXiv:1402.6950: Cassini Q2 = (3 +- 3)e-27 s^-2;
# Hees, Famaey, Angus, Gentile, MNRAS 455, 449 (2016), arXiv:1510.01369: q = -2 Q2 sqrt(GM)/(3 a0^1.5), eta = g_e/a0, Table 2 (nu_2, nu_3, nu_5);
# Park, Hees, Famaey, Desmond, Durakovic, PRD 114, 024066 (2026), arXiv:2602.17884: Q2 = (1.6 +- 1.8)e-27 s^-2.
# Method: Q2 = 3 G int rho_ph P2(mu)/r^3 d^3x, rho_ph = div[(nu - 1) grad Phi_N]/(4 pi G), integrated by parts; units GM = a0 = 1.
# Run: uv run --with numpy --with scipy python mond_solar_q2_check.py
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

nu_alpha = lambda a: (lambda y: ((1 + np.sqrt(1 + 4 * y**(-a))) / 2)**(1 / a))

def q_of(nu, eta):
    gNe = brentq(lambda g: g * nu(g) - eta, 1e-6, eta)              # Newtonian external field from the MOND one
    def inner(mu, r):
        gN = np.sqrt(1 / r**4 - 2 * gNe * mu / r**2 + gNe**2)
        P2 = 0.5 * (3 * mu**2 - 1)
        return (nu(gN) - 1) / r**2 * (-3 * P2 * (1 / r**2 - gNe * mu) - 3 * gNe * mu * (1 - mu**2))
    def radial(lr):
        r = np.exp(lr)
        return quad(lambda mu: inner(mu, r), -1, 1, limit=200, epsabs=1e-13)[0] * r
    Q2 = -1.5 * quad(radial, np.log(1e-3), np.log(1e3), limit=400, points=[0.0])[0]
    return 2 / 3 * Q2                                                # -q = +2 Q2/3 in units GM = a0 = 1 (positive in the paper table)

# K1 validation against Hees et al. 2016, Table 2: -q x 100 = 10 (nu_2, eta 1.20), 8.29 (nu_3, 1.20), 5.51 (nu_5, 1.30)
ref = [(2, 1.20, 0.100), (3, 1.20, 0.0829), (5, 1.30, 0.0551)]
got = [(a, e, q_of(nu_alpha(a), e), qr) for a, e, qr in ref]
for a, e, qc, qr in got:
    print(f"   nu_{a}, eta = {e}: -q = {qc:.4f} (paper {qr})")
ok("K1 the pipeline reproduces the published -q of nu_2, nu_3, nu_5 to 5%", all(abs(qc / qr - 1) < 0.05 for a, e, qc, qr in got))

# K2 the simple function nu_1 (= self-dual x/(1+x)). Q2 = (3/2)(-q) a0^1.5 / sqrt(GM_sun). Range: g_e = 1.9-2.4e-10 m/s^2, a0 = 1.0-1.3e-10.
GM = 1.32712e20
rows = []
for ge in (1.9e-10, 2.4e-10):
    for a0 in (1.0e-10, 1.2e-10, 1.3e-10):
        mq = q_of(nu_alpha(1), ge / a0)
        Q2 = 1.5 * mq * a0**1.5 / np.sqrt(GM)
        rows.append((ge, a0, ge / a0, mq, Q2))
        print(f"   simple nu_1: g_e = {ge:.1e}, a0 = {a0:.1e}, eta = {ge / a0:.2f}: -q = {mq:.4f}, Q2 = {Q2:.2e} s^-2")
Qmin = min(r[4] for r in rows)
ok(f"K2 simple nu_1 predicts Q2 >= {Qmin:.1e} s^-2 over the whole range; Cassini 2014 allows <= 6e-27 (1 sigma), DE440 2026 allows "
   f"<= 3.4e-27: excluded by a factor >= {Qmin / 6e-27:.0f}", Qmin > 6e-27)

# K3 the whole self-dual class, not only simple: self-duality h(1/y) = 1/h(y) with the deep limit h ~ 1/y (y -> 0) forces h ~ 1/y also at
#     large y, i.e. g_D -> a0: a constant tail at high acceleration (algebra). The SM4 alternative (51) is also tested for Q2.
def nu_from_h(h):
    def nu(y):
        g = brentq(lambda gg: gg - y * (1 + h(gg)), y * (1 + 1e-14), y + 2 + 10 * np.sqrt(y))
        return g / y
    return nu
h_alt = lambda y: np.exp(-np.log(y) + 0.5 * np.log(y) * np.exp(-np.log(y)**2))
yy = 1e6
ok(f"K3a self-dual h with deep limit has h(y) y -> {h_alt(yy) * yy:.4f} at y = 1e6 (and simple h = 1/y exactly): tail g_D -> a0", abs(h_alt(yy) * yy - 1) < 1e-3)
mq_alt = q_of(nu_from_h(h_alt), 1.58)
Q2_alt = 1.5 * mq_alt * (1.2e-10)**1.5 / np.sqrt(GM)
print(f"   self-dual alternative (SM4), eta = 1.58: -q = {mq_alt:.4f}, Q2 = {Q2_alt:.2e}")
ok(f"K3b the other self-dual law is excluded too: Q2 = {Q2_alt:.1e} > 6e-27", Q2_alt > 6e-27)

# K4 the standard RAR form nu = 1/(1 - exp(-sqrt(y))) (McGaugh; nu-hat_1 of Hees et al.) -- 37 claimed it passes the Solar System because its
#     tail is exponential. But Q2 is set by the transition near g ~ g_e, not by the tail.
nu_rar = lambda y: 1 / (1 - np.exp(-np.sqrt(y)))
mq_rar = q_of(nu_rar, 1.58)
Q2_rar = 1.5 * mq_rar * (1.2e-10)**1.5 / np.sqrt(GM)
print(f"   McGaugh RAR nu-hat_1, eta = 1.58: -q = {mq_rar:.4f}, Q2 = {Q2_rar:.2e}")
ok(f"K4 REFUTED (37: 'the RAR form passes the Solar System'): Q2 = {Q2_rar:.1e} s^-2 > 6e-27 -- excluded too (Hees et al.: nu-hat with alpha < 6)", Q2_rar > 6e-27)

# K5 sanity: a surviving family member, nu-bar_2 with a0 = 0.815e-10 at eta = 2.30 (paper: Q2 = 5.9e-27, inside the bound)
nu_bar = lambda a: (lambda y: (-np.expm1(-y**a))**(-1 / (2 * a)) + (1 - 1 / (2 * a)) * np.exp(-y**a))
mq_b = q_of(nu_bar(2), 2.30)
Q2_b = 1.5 * mq_b * (0.815e-10)**1.5 / np.sqrt(GM)
ok(f"K5 nu-bar_2 at eta = 2.30: Q2 = {Q2_b:.1e} (paper 5.9e-27): survivors exist, but they have a sharp transition and no self-duality",
   abs(Q2_b / 5.9e-27 - 1) < 0.15)

# K6 is the survivor self-dual? With g = nu(y) y, h(x) = g_D/g_N as a function of x = g/a0; self-duality needs h(1/x) h(x) = 1.
def h_of_nu(nu):
    def h(x):
        y = brentq(lambda yy: nu(yy) * yy - x, 1e-8, x)
        return nu(y) - 1
    return h
hb = h_of_nu(nu_bar(2))
prods = [hb(x) * hb(1 / x) for x in (0.3, 3.0, 10.0)]
ok(f"K6 nu-bar_2 h(x) h(1/x) = {[round(p, 3) for p in prods]} (1 for a self-dual law): the surviving law is not self-dual",
   all(abs(p - 1) > 0.1 for p in prods))

# K7 paper3 uses nu = sqrt(1 + 1/y) (not the simple function). Its high-g tail: nu - 1 ~ 1/(2y), g_D -> a0/2. Q2 over the same range.
nu_p3 = lambda y: np.sqrt(1 + 1 / y)
Q2_p3 = [1.5 * q_of(nu_p3, ge / a0) * a0**1.5 / np.sqrt(GM) for ge in (1.9e-10, 2.4e-10) for a0 in (1.0e-10, 1.2e-10, 1.5e-10)]
ok(f"K7 paper3's nu = sqrt(1 + 1/y): Q2 = {min(Q2_p3):.1e} .. {max(Q2_p3):.1e} s^-2, {min(Q2_p3) / 6e-27:.1f}-{max(Q2_p3) / 6e-27:.1f}x above Cassini: excluded",
   min(Q2_p3) > 6e-27)
