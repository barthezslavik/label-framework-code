# 84: program item 1 (paper3 premise P2). Does the framework's own data -- the fluctuation spectrum of the label measure -- put galaxies
# on a common k = 2 level (fixed surface density = fixed acceleration)? Hierarchical formation: a halo of mass M collapses when its
# fluctuation sigma(M) ~ M^{-(n+3)/6} reaches a threshold, so its characteristic density is rho_c ~ (1+z_f)^3 ~ M^{-(n+3)/2}. HS1-HS4.
# Run: ~/.venvs/jev/bin/python ga_p2_spectrum_check.py
import numpy as np
import sympy as sp
from scipy.integrate import quad

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

n, M = sp.symbols('n M', positive=False)
Mp = sp.symbols('M', positive=True)
rho = Mp**(-(n + 3) / 2); r = (Mp / rho)**sp.Rational(1, 3)
Sig_exp = sp.simplify(sp.diff(sp.log(rho * r), Mp) * Mp)                      # d ln(rho r)/d ln M
v2_exp = sp.simplify(sp.diff(sp.log(Mp / r), Mp) * Mp)                        # d ln v^2/d ln M
s_exp = sp.simplify(2 / v2_exp)                                               # M ~ v^s
k_exp = sp.solve(sp.Eq(1 - sp.Symbol('k') * sp.diff(sp.log(r), Mp) * Mp, 0), sp.Symbol('k'))[0]

# HS1 surface density rho_c r ~ M^{-(n+2)/3}: constant iff n = -2 (the k = 2 level set); volume density constant iff n = -3 (k = 3).
ok(f"HS1 d ln Sigma/d ln M = {Sig_exp}: common surface density (k = 2 level) iff n = {sp.solve(Sig_exp, n)}",
   sp.solve(Sig_exp, n) == [-2])

# HS2 population slope M ~ v^s with s = 12/(1 - n), and the level k = 6/(n + 5) (s = 2k/(k-1) of paper3): n = -2 -> k = 2, s = 4; n = -3 -> k = 3, s = 3.
ok(f"HS2 s = {s_exp}, level k = {sp.simplify(k_exp)}: n=-2 -> s={s_exp.subs(n, -2)}, k={k_exp.subs(n, -2)}; n=-3 -> s={s_exp.subs(n, -3)}, k={k_exp.subs(n, -3)}",
   sp.simplify(s_exp - 12 / (1 - n)) == 0 and s_exp.subs(n, -2) == 4 and k_exp.subs(n, -2) == 2 and s_exp.subs(n, -3) == 3)

# HS3 consistency with the phase-space form of P2 (doc 34, R6): constant knee acceleration needs eta0 ~ sigma^-5. With rho_c and v^2 ~ M/r:
#     rho/sigma^3 ~ sigma^p; at n = -2 the exponent is -5.
sig = sp.sqrt(Mp / r)
p_exp = sp.simplify(sp.diff(sp.log(rho / sig**3), Mp) / sp.diff(sp.log(sig), Mp))
ok(f"HS3 phase density rho/sigma^3 ~ sigma^({p_exp}); at n = -2: {p_exp.subs(n, -2)} (doc 34 R6 needs -5)", p_exp.subs(n, -2) == -5)

# HS4 the effective index of the CDM spectrum (BBKS transfer function, Planck h = 0.674, Omega_m = 0.315, n_s = 0.965), n_eff = -3 - 6 dln sigma/dln M:
#     galaxy halo masses 1e10-1e12 Msun/h give n_eff = -2.36 .. -2.10, i.e. s = 3.57 .. 3.87 (if P1 holds); measured baryonic TF 3.85 +- 0.09,
#     systematic range 3.5-4.0 (Lelli 2019). The slope runs with mass: steeper at higher mass.
h, Om = 0.674, 0.315; Gam = Om * h
T = lambda k: np.log(1 + 2.34 * k / Gam) / (2.34 * k / Gam) * (1 + 3.89 * k / Gam + (16.1 * k / Gam)**2 + (5.46 * k / Gam)**3 + (6.71 * k / Gam)**4)**-0.25
P = lambda k: k**0.965 * T(k)**2
rho_m = 2.775e11 * Om
W = lambda x: 3 * (np.sin(x) - x * np.cos(x)) / x**3
def sig2(Mm):
    R = (3 * Mm / (4 * np.pi * rho_m))**(1 / 3)
    return quad(lambda lk: np.exp(lk)**3 * P(np.exp(lk)) * W(np.exp(lk) * R)**2, np.log(1e-4), np.log(1e4), limit=500)[0]
def neff(Mm, e=1.05):
    return -3 - 6 * (np.log(sig2(Mm * e)) - np.log(sig2(Mm / e))) / 2 / (2 * np.log(e))
nn = {Mm: neff(Mm) for Mm in (1e10, 1e11, 1e12)}
ss = {Mm: 12 / (1 - v) for Mm, v in nn.items()}
print("   " + ", ".join(f"M={Mm:.0e}: n_eff={nn[Mm]:.2f}, s={ss[Mm]:.2f}" for Mm in nn))
ok(f"HS4 CDM n_eff over galaxy masses {min(nn.values()):.2f}..{max(nn.values()):.2f} -> s = {min(ss.values()):.2f}..{max(ss.values()):.2f}, inside the BTFR range 3.5-4.0",
   3.5 < min(ss.values()) and max(ss.values()) < 4.0)
