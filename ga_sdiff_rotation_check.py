# Solid-body cores and flat envelopes from SDiff symmetries, without a0 (papers/restart/34-sdiff-rotation.md). R1-R5.
# R1 Kelvin: at fixed angular momentum bivector L the kinetic energy is minimal exactly for rigid rotation v = x . Omega (Omega a bivector).
# R2 Lynden-Bell: phase-space SDiff (Liouville) conserves the fine-grained phase density, so coarse-grained cells obey exclusion; maximal entropy
#    at fixed mass and energy is the Fermi-Dirac form f = eta0/(1 + e^{beta (eps - mu)}).
# R3-R5 the self-gravitating Lynden-Bell sphere: degenerate core (uniform density -> v ~ r, solid body) and isothermal envelope (rho ~ r^-2 ->
#    v ~ const); the knee is set by the degeneracy and by (eta0, sigma) of each system, not by a universal acceleration.
# Run: uv run --with sympy --with numpy --with scipy python ga_sdiff_rotation_check.py
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp, quad
from ga_core import ok

# R1 rigid rotation minimises kinetic energy at fixed angular momentum (random particles; the inertia map acts on bivectors).
rng = np.random.default_rng(3)
m = rng.uniform(0.5, 2, 40); x = rng.normal(size=(40, 3)); x -= (m[:, None] * x).sum(0) / m.sum()
v = rng.normal(size=(40, 3))
L = (m[:, None] * np.cross(x, v)).sum(0)                                   # dual of the bivector sum m x ^ v
I = sum(mi * (xi @ xi * np.eye(3) - np.outer(xi, xi)) for mi, xi in zip(m, x))
w = np.linalg.solve(I, L)                                                  # dual of Omega
vr = np.cross(w, x)                                                        # x . Omega in vector (dual) form
KE = lambda u: 0.5 * (m * (u**2).sum(1)).sum()
Lr = (m[:, None] * np.cross(x, vr)).sum(0)
du = v - vr
cross = (m[:, None] * (vr * du).sum(1)[:, None]).sum()
ok(f"R1 rigid flow with the same L: |L_rigid - L| = {np.abs(Lr - L).max():.1e}; KE(v) = {KE(v):.3f} >= KE(rigid) = {KE(vr):.3f}; "
   f"cross term {cross:.1e} = 0, so KE(v) = KE(rigid) + KE(v - rigid)",
   np.abs(Lr - L).max() < 1e-10 and abs(cross) < 1e-10 and abs(KE(v) - KE(vr) - KE(du)) < 1e-10 and KE(v) > KE(vr))

# R2 Lynden-Bell entropy s(f) = -[f ln f + (eta0 - f) ln(eta0 - f)]: stationarity with multipliers alpha (mass), beta (energy) gives Fermi-Dirac.
f, eta0, al, be, eps = sp.symbols('f eta0 alpha beta epsilon', positive=True)
s = -(f * sp.log(f) + (eta0 - f) * sp.log(eta0 - f))
sol = sp.solve(sp.Eq(sp.diff(s, f), al + be * eps), f)
ok(f"R2 ds/df = alpha + beta eps -> f = {sp.simplify(sol[0])} (Fermi-Dirac in eps, occupancy <= eta0: exclusion from Liouville)",
   len(sol) == 1 and sp.simplify(sol[0] - eta0 / (1 + sp.exp(al + be * eps))) == 0)

# R3 the self-gravitating Lynden-Bell sphere, dimensionless: psi = beta (mu - phi); rho(psi) = F(psi) = int_0^inf sqrt(t) dt/(1+e^{t-psi})
#    (Fermi-Dirac integral, up to constants); Poisson psi'' + 2 psi'/r = -F(psi). v_c^2 = M(r)/r with M = -r^2 psi' (units G = beta = 1).
def F(p):
    return quad(lambda t: np.sqrt(t) / (1 + np.exp(np.clip(t - p, -700, 700))), 0, max(p, 0) + 60, limit=200)[0]
pg = np.linspace(-25, 120, 3001); Fg = np.array([F(p) for p in pg])
Fi = lambda p: np.where(p < -25, np.sqrt(np.pi) / 2 * np.exp(np.maximum(p, -700)), np.where(p > 120, 2 / 3 * np.abs(p)**1.5, np.interp(p, pg, Fg)))  # tails: Boltzmann / degenerate
def sphere(psi0, rmax=4e3):
    rhs = lambda r, y: [y[1], -Fi(y[0]) - 2 * y[1] / r]
    r0 = 1e-6
    sol = solve_ivp(rhs, [r0, rmax], [psi0, -Fi(psi0) * r0 / 3], rtol=1e-9, atol=1e-12, dense_output=True)
    r = np.logspace(-4, np.log10(rmax), 4000)
    psi, dpsi = sol.sol(r)
    vc = np.sqrt(np.maximum(-r * dpsi, 1e-300))
    return r, vc, np.array([Fi(p) for p in psi])
res = {}
for psi0 in (-3.0, 10.0, 25.0, 40.0):   # (psi0 >~ 50: the envelope appears only at r > 1e14 -- a detached degenerate ball)
    r, vc, rho = sphere(psi0, rmax=1e14)
    slope = np.gradient(np.log(vc), np.log(r))
    peaks = np.where((vc[1:-1] > vc[:-2]) & (vc[1:-1] >= vc[2:]))[0] + 1
    k = peaks[0]                                                           # first maximum: end of the solid-body core
    inner = slope[(r > 1e-3 * r[k]) & (r < 0.05 * r[k])].mean()
    outer = vc[(r > 1e12) & (r < 1e14)]                                    # last two decades: isothermal envelope
    res[psi0] = dict(inner=inner, r_k=r[k], v_k=vc[k], a_k=vc[k]**2 / r[k], flat=outer.std() / outer.mean(), vout=outer.mean())
    print(f"   psi0 = {psi0:6.1f}: inner slope dlnv/dlnr = {inner:.3f}, core knee r = {r[k]:.3g}, v = {vc[k]:.3f}, "
          f"a_knee = v^2/r = {vc[k]**2 / r[k]:.3g}, envelope v (r = 1e12..1e14) = {res[psi0]['vout']:.3f} (rel. spread {res[psi0]['flat']:.3f})")
ok("R3 every sphere is solid-body at the centre (dlnv/dlnr = 1: uniform core) and tends to the flat isothermal envelope v -> sqrt2 sigma "
   "(spread < 5% over r = 1e12..1e14)",
   all(abs(q['inner'] - 1) < 0.02 for q in res.values()) and all(q['flat'] < 0.05 and abs(q['vout'] - np.sqrt(2)) < 0.05 for q in res.values()))

# R4 no universal acceleration: in physical units the family has two scales (eta0, sigma) plus the degeneracy psi0. The knee acceleration in
#    the dimensionless units already varies with psi0, and rescaling (eta0, sigma) moves it further (homology: a ~ sigma^2 / r_unit).
ak = np.array([res[p]['a_k'] for p in (-3.0, 10.0, 25.0, 40.0)])
ok(f"R4 knee acceleration a_k = {np.round(ak, 3)} (dimensionless) varies by x{ak.max() / ak.min():.1f} with the degeneracy alone: "
   "the solid-body -> flat transition is not at a fixed acceleration", ak.max() / ak.min() > 2)

# R5 homology: at fixed psi0 the physical knee acceleration scales as sqrt(G eta0) sigma^{5/2} (dimensional analysis):
#    [eta0] = mass/(length^3 velocity^3), so G eta0 sigma^3 has units 1/time^2 and the only length is sigma/sqrt(G eta0 sigma^3).
G_, e_, s_ = sp.symbols('G eta0 sigma', positive=True)
r_unit = s_ / sp.sqrt(G_ * e_ * s_**3)
a_unit = sp.simplify(s_**2 / r_unit)
ok(f"R5 length unit {sp.simplify(r_unit)}, acceleration unit {a_unit}: an a0-like constant would need eta0 sigma^5 fixed across galaxies",
   sp.simplify(a_unit - sp.sqrt(G_ * e_) * s_**sp.Rational(5, 2)) == 0)

# R6 against the data-side fact that the knee sits at a nearly fixed acceleration (RAR; constant core surface density rho0 r0, Donato+ 2009):
#    with a universal eta0 (fixed phase density) the knee acceleration ~ G rho0 r0 scales as sigma^{5/2}, so a fixed knee would need
#    eta0 ~ sigma^-5 across galaxies -- an extra relation, i.e. the a0 problem reappears as a regression between eta0 and sigma.
rho0 = e_ * s_**3; r0 = s_ / sp.sqrt(G_ * rho0)
ok(f"R6 G rho0 r0 = {sp.simplify(G_ * rho0 * r0)} at fixed degeneracy: constant only if eta0 ~ sigma^-5 "
   f"(then G rho0 r0 = {sp.simplify((G_ * rho0 * r0).subs(e_, 1 / s_**5))})",
   sp.simplify(G_ * rho0 * r0 - sp.sqrt(G_) * sp.sqrt(e_) * s_**sp.Rational(5, 2)) == 0 and sp.simplify((G_ * rho0 * r0).subs(e_, 1 / s_**5) - sp.sqrt(G_)) == 0)
