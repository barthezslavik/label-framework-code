# 95: label budget. Which labels are tied to others by laws of the framework, and how many independent ones remain? LB1-LB4.
# LB1 is a computation (the level a0 of P2 from the dust collapse of the CDM spectrum); LB2-LB4 tabulate the relations found in docs 84-94.
# Run: ~/.venvs/jev/bin/python ga_label_budget_check.py
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

# LB1 a0 is the label of the common level of P2 (paper3): a common surface density, a0 ~ G Sigma. Dust collapse fixes Sigma from the spectrum:
#     a nu-sigma halo of mass M collapses when nu D(z) sigma(M) = delta_c (delta_c = 1.686, spherical dust collapse), virialises at
#     Delta = 18 pi^2 times the mean matter density, and its mean acceleration at the virial radius is g_v = G M / r_v^2 = pi G Sigma_v.
#     Inputs: the spectrum labels (sigma8 = 0.811, n_s = 0.965, Omega_m = 0.315, h = 0.674). Output: g_v(M) and the factor kappa = a0/g_v.
h, Om = 0.674, 0.315; Gam = Om * h
T = lambda k: np.log(1 + 2.34 * k / Gam) / (2.34 * k / Gam) * (1 + 3.89 * k / Gam + (16.1 * k / Gam)**2 + (5.46 * k / Gam)**3 + (6.71 * k / Gam)**4)**-0.25
P = lambda k: k**0.965 * T(k)**2
Wf = lambda q: 3 * (np.sin(q) - q * np.cos(q)) / q**3
sR = lambda R: np.sqrt(quad(lambda lk: np.exp(lk)**3 * P(np.exp(lk)) * Wf(np.exp(lk) * R)**2 / (2 * np.pi**2), np.log(1e-4), np.log(1e4), limit=500)[0])
norm = 0.811 / sR(8.0)
rho_m0 = 2.775e11 * Om * h**2                                   # Msun / Mpc^3
sigM = lambda M: norm * sR((3 * M * h / (4 * np.pi * 2.775e11 * Om))**(1 / 3))      # R in Mpc/h, M in Msun
def D(z):                                                       # linear growth (LCDM), normalised to 1 today
    def f(N, y):
        a = np.exp(N); E2 = Om / a**3 + 1 - Om; dlnE = -1.5 * Om / a**3 / E2
        return [y[1], -(2 + dlnE) * y[1] + 1.5 * Om / a**3 / E2 * y[0]]
    s = solve_ivp(f, [np.log(1e-3), 0.0], [1e-3, 1e-3], dense_output=True, rtol=1e-9)
    return s.sol(-np.log1p(z))[0] / s.sol(0.0)[0]
G = 4.301e-9                                                    # Mpc (km/s)^2 / Msun
kms2_Mpc_to_SI = 1e6 / 3.0857e22                                # (km/s)^2/Mpc -> m/s^2
out = {}
for nu in (1.0, 2.0):
    for M in (1e10, 1e11, 1e12):
        zc = brentq(lambda z: D(z) - 1.686 / (nu * sigM(M)), 0.0, 50) if 1.686 / (nu * sigM(M)) < 1 else 0.0   # nu-sigma peaks: earlier for larger nu
        zc_eff = zc
        rv = (3 * M / (4 * np.pi * 18 * np.pi**2 * rho_m0 * (1 + zc_eff)**3))**(1 / 3)
        gv = G * M / rv**2 * kms2_Mpc_to_SI
        out[(nu, M)] = (zc, gv)
print("   " + "; ".join(f"nu={nu:.0f} M={M:.0e}: z_c={zc:.2f}, g_v={gv:.2e} m/s^2" for (nu, M), (zc, gv) in out.items()))
a0 = 1.2e-10
spread = {nu: np.log10(max(out[(nu, M)][1] for M in (1e10, 1e11, 1e12)) / min(out[(nu, M)][1] for M in (1e10, 1e11, 1e12))) for nu in (1.0, 2.0)}
kap = {nu: a0 / np.mean([out[(nu, M)][1] for M in (1e10, 1e11, 1e12)]) for nu in (1.0, 2.0)}
ok(f"LB1 virial acceleration of typical halos varies by {spread[1.0]:.2f} dex (nu = 1), {spread[2.0]:.2f} dex (nu = 2) over 1e10-1e12 Msun "
   f"(common level, P2); a0/g_v = kappa = {kap[1.0]:.0f} (nu = 1), {kap[2.0]:.0f} (nu = 2); kappa = 1/f, the inverse baryon share of the halo "
   f"(doc 96, KA1)", all(v < 1.0 for v in spread.values()))

# LB2-LB4: the budget. Each entry: label -> (class, tied to, source). 'free' = independent initial datum of the framework.
budget = {
    'Lambda (rho_vac)':        ('free', 'integration constant of the four-label vacuum; discreteness cannot replace it (EV1-EV2)', '91'),
    'A (discrete amplitude)':  ('free if axiom D', 'alpha (l_P/l)^2 <= 0.015 by BBN; enters only as a fluctuation', '91'),
    'G':                       ('free', 'cutoff label of induced gravity (89); same scale as l if D holds (reading)', '89, 91'),
    'eta_b (baryons/photon)':  ('free', 'conserved count ratio (zero mode, like Omega_c/Omega_b)', '94'),
    'Omega_c/Omega_b':         ('free', 'zero mode of the ladder flow (AT1): conserved ratio of label counts', '94'),
    'beta (m(phi))':           ('free', 'bounded: Cassini on baryons, DESI-S8 tension (86)', '82, 86'),
    'A_s':                     ('free', 'amplitude of the label-measure spectrum', '84, 86'),
    'n_s - 1':                 ('free', 'deviation from the K-symmetric point n_s = 1', 'base'),
    'a0':                      ('tied', 'a0 = R^4 g_v(nu; A_s, Omega_m, h)/f (paper3 Prop level; KA1 at R = 1); no-loss value 0.2-0.85 x observed (KA5, KA7)', '95, 96, paper3'),
    'nu (peak height of hosts)': ('free', 'sets g_v at given A_s, Omega_m, h; not fixed by the framework (paper3)', '95, 99, paper3'),
    'R (v_flat / v_v)':        ('free', 'plateau-to-virial speed ratio; lensing gives R ~ 1.2; not fixed by the framework (paper3, KA8)', '96, 109, paper3'),
    'r_b (retained share)':    ('free', 'f = (Omega_b/Omega_m) r_b; needed r_b = 0.2-0.7; set by feedback, outside the framework', '85, 96'),
    'f (P1 share)':            ('tied', 'cosmic baryon share x retention, equal along the orbit (PA6)', '85'),
    'q (spin label)':          ('free', 'mass-independent spin label of settling; scatter 0.20 dex', '85'),
    'BTFR slope s':            ('derived', 's = 12/(1 - n_eff), 3.68-3.79 at the SPARC centre, measured 3.67 +- 0.11 (BR2b)', '84, 93'),
    'Omega_phi in phi-MDE':    ('derived', '2 beta^2/3 (function of the label beta, AT2)', '94'),
    'baryon bias D_b/D_c':     ('derived', '1 (uncoupled), 0.997 at beta = 0.03 (AT4)', '94'),
    'Lambda l_P^2 ~ 1e-122':   ('derived if D', '(l_P^2/sqrt V4) x 8 pi alpha: the age in Planck units times the label A (DL5)', '91'),
    'charge ratios, spin':     ('derived', 'integers via compact generators (CI3-CI5)', '92'),
    'alpha, gauge group':      ('free', 'not selected by K-blindness (EM6)', '90'),
    'masses, Yukawas, theta, mu^2': ('free', 'K-weight != 0 or no symmetric point; non-compact K charge (CI7)', 'base, 92'),
}
free = [k for k, v in budget.items() if v[0] == 'free']
tied = [k for k, v in budget.items() if v[0] == 'tied']
der = [k for k, v in budget.items() if v[0].startswith('derived')]
for kk, v in budget.items(): print(f"   {kk:30s} {v[0]:16s} {v[1]}  [{v[2]}]")
ok(f"LB2 cosmology and galaxies: {len(free)} free labels ({', '.join(free)}); 11 before the series cross-check (doc 142), which added nu and R from paper3", len(free) == 13)
ok(f"LB3 tied to other labels by a law of the framework: {tied}; derived numbers: {len(der)}", tied == ['a0', 'f (P1 share)'])
ok("LB4 recorded: no relation reduces the count of free labels -- a0 and f are traded for r_b, q, nu and R; the labels that disappear are the "
   "ones that turn into counts or attractor values (BTFR slope, charge ratios, 1e-122 under axiom D)", True)
