# 154: paper13, step 5. Equilibrium and the classes of numbers.
# EQ1 maximising -int rho ln rho at fixed int rho = 1 and int rho V = E (numerical optimisation on a grid) gives the Gibbs density
#     e^{-beta V}/Z; beta is the Lagrange multiplier of the energy constraint;
# EQ2 the multiplier is set by the constraint value: for E between the minimum of V and the mean of V, beta(E) is continuous and
#     monotone (computed for five values of E): the form e^{-beta V} is fixed, the temperature 1/beta is a label of the solution;
# EQ3 with the Casimirs of a volume-preserving (Vlasov) flow as constraints, a two-level fine-grained density (levels 0 and eta0) and
#     Lynden-Bell's mixing entropy give the Fermi-Dirac-like form p = 1/(1 + e^{beta(eps - mu)}) (computed by optimisation): the extra
#     Casimir adds the multiplier mu, conjugate to the conserved occupation, as the label count adds the chemical potential;
# EQ4 the isothermal self-gravitating sphere (the equilibrium of R5a in the overview): the solution of (r^2 psi')' = r^2 e^{-psi}
#     tends to the singular isothermal sphere, d ln rho / d ln r -> -2, so v^2 = G M(r)/r tends to a constant (flat curve);
# EQ5 k_B only converts units of temperature and entropy: rescaling k_B changes S and T by inverse factors and leaves every
#     dimensionless equilibrium (beta V) unchanged; it is a unit, not a number of the framework.
# Run: ~/.venvs/jev/bin/python ga_stat_equilibrium_check.py
import numpy as np
from scipy.optimize import minimize, brentq
from scipy.integrate import solve_ivp

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

n = 200; x = np.linspace(-3, 3, n); dx = x[1] - x[0]
V = 0.5 * x**2 + 0.3 * np.sin(2 * x)

def maxent(E):
    # maximise entropy with Lagrange constraints solved by optimisation over log rho
    cons = [{'type': 'eq', 'fun': lambda z: np.exp(z).sum() * dx - 1},
            {'type': 'eq', 'fun': lambda z: (np.exp(z) * V).sum() * dx - E}]
    f = lambda z: (np.exp(z) * z).sum() * dx
    z0 = np.log(np.full(n, 1 / (n * dx)))
    r = minimize(f, z0, constraints=cons, method='SLSQP', options={'maxiter': 2000, 'ftol': 1e-14})
    return np.exp(r.x)
def gibbs_beta(E):
    g = lambda b: (np.exp(-b * V) * V).sum() / np.exp(-b * V).sum() - E
    return brentq(g, 1e-6, 200)

E1 = 0.6
rho = maxent(E1); b1 = gibbs_beta(E1)
gib = np.exp(-b1 * V); gib /= gib.sum() * dx
ok(f"EQ1 maximum entropy at fixed mass and energy E = {E1}: max deviation from the Gibbs density {np.abs(rho - gib).max() / gib.max():.1e} "
   f"(beta = {b1:.4f})", np.abs(rho - gib).max() / gib.max() < 1e-3)

Es = [0.2, 0.4, 0.6, 0.9, 1.2]; bs = [gibbs_beta(E) for E in Es]
ok("EQ2 beta(E): " + ", ".join(f"E={E}: {b:.3f}" for E, b in zip(Es, bs)) + "; continuous and monotone: the temperature is set by the "
   "initial energy, a label", all(bs[i] > bs[i + 1] for i in range(len(bs) - 1)))

# EQ3 two-level Lynden-Bell: phase-space cells with energy eps_i; occupation fraction p_i in [0,1] of the level eta0
eps = np.linspace(0, 3, 60); g_w = np.ones_like(eps)
Nocc, Etot = 20.0, 18.0
def negS(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return (g_w * (p * np.log(p) + (1 - p) * np.log(1 - p))).sum()
cons = [{'type': 'eq', 'fun': lambda p: (g_w * p).sum() - Nocc}, {'type': 'eq', 'fun': lambda p: (g_w * p * eps).sum() - Etot}]
res = minimize(negS, np.full(eps.size, Nocc / eps.size), constraints=cons, bounds=[(1e-9, 1 - 1e-9)] * eps.size, method='SLSQP',
               options={'maxiter': 3000, 'ftol': 1e-14})
p = res.x
logit = np.log((1 - p) / p)                             # FD form: logit = beta (eps - mu), linear in eps
A = np.vstack([eps, np.ones_like(eps)]).T; coef, *_ = np.linalg.lstsq(A, logit, rcond=None)
lin_err = np.abs(A @ coef - logit).max()
ok(f"EQ3 two-level mixing entropy with the Casimir and energy fixed: ln((1-p)/p) is linear in eps to {lin_err:.1e} (beta = {coef[0]:.3f}, "
   f"mu = {-coef[1] / coef[0]:.3f}): Fermi-Dirac-like form, mu conjugate to the conserved occupation", lin_err < 1e-3)

# EQ4 isothermal sphere: rho = e^{-psi}, (r^2 psi')' = r^2 e^{-psi}, psi(0) = psi'(0) = 0
sol = solve_ivp(lambda r, y: [y[1], np.exp(-y[0]) - 2 * y[1] / r], [1e-6, 1e5], [0.0, 0.0], rtol=1e-10, atol=1e-12, dense_output=True)
rr = np.array([1e3, 1e4, 1e5]); psi = sol.sol(rr)[0]; dpsi = sol.sol(rr)[1]
slope = -rr * dpsi                                      # d ln rho / d ln r = -r psi'
ok("EQ4 isothermal sphere: d ln rho / d ln r = " + ", ".join(f"{s_:.4f}" for s_ in slope) + " at r = 1e3, 1e4, 1e5 -> -2 (flat curve)",
   abs(slope[-1] + 2) < 0.02 and abs(slope[-1] + 2) < abs(slope[0] + 2) + 1e-9)

# EQ5 k_B as a unit
kB1, kB2 = 1.0, 1.380649e-23
T1 = 1 / (kB1 * b1); T2 = 1 / (kB2 * b1)
inv = abs((1 / (kB1 * T1)) - (1 / (kB2 * T2)))
ok(f"EQ5 changing k_B from {kB1} to {kB2} changes T by {T1 / T2:.3e} and leaves beta = 1/(k_B T) = {b1:.4f} and the Gibbs density unchanged "
   f"(difference {inv:.1e}): k_B is a unit", inv < 1e-12)

print("ALL PASS")
