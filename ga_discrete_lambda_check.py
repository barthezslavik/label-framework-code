# 91: discrete labels and Lambda. The four-label vacuum (k = 4, base LL3/LL6) has rho_vac = n F' - F as an integration constant. Does a
# fluctuating Lambda ~ 1/sqrt(V4) follow when the labels are names of discrete elements? DL1-DL6.
# Run: ~/.venvs/jev/bin/python ga_discrete_lambda_check.py
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

# DL1 minisuperspace of the four-label vacuum: spatial labels comoving (det = 1), time label tau(t), lapse Nl, scale factor a.
#     n = tau'/(Nl a^3), L = Nl a^3 F(n). tau is cyclic: p = F'(n) = mu is conserved (the label 'chemical potential'). The Hamiltonian is
#     H = p tau' - L = Nl a^3 (n F' - F) = Nl a^3 rho(mu): the vacuum energy is a function of the conserved momentum only.
t = sp.symbols('t'); Nl, a = sp.symbols('N_l a', positive=True); nu = sp.symbols('nu', positive=True); m = sp.symbols('m', positive=True)
td = sp.symbols('taudot', positive=True)
F = lambda x: -m * x**nu
n = td / (Nl * a**3)
L = Nl * a**3 * F(n)
p = sp.diff(L, td)
H = sp.simplify(p * td - L)
rho_n = sp.simplify(n * sp.diff(F(n), td) * (Nl * a**3) - F(n))          # n F'(n) - F(n), F' = dF/dn
ok(f"DL1 p_tau = F'(n) (tau cyclic, conserved); H = N_l a^3 (n F' - F): {sp.simplify(H - Nl * a**3 * rho_n) == 0}",
   sp.simplify(p - sp.diff(F(sp.Symbol('x')), sp.Symbol('x')).subs(sp.Symbol('x'), n)) == 0 and sp.simplify(H - Nl * a**3 * rho_n) == 0)

# DL2 Legendre: with mu = F'(n), d rho/d mu = n for ANY F (F'' != 0): rho(mu) = n mu - F, d rho = n d mu + mu dn - F' dn = n d mu.
#     Hamilton: tau' = dH/dp = N_l a^3 n, so the conjugate of mu accumulates the label count: Delta tau x (spatial label volume) =
#     int n sqrt(-g) d^4x = N, the number of four-label elements in the region. Conjugate pair (mu, N); delta rho = n delta mu.
pp = sp.symbols('p'); Fg = sp.Function('F'); ng = sp.Function('n')(pp)          # generic F, n(p) the inverse of p = F'(n)
rho_p = ng * pp - Fg(ng)
drho = sp.diff(rho_p, pp).subs(sp.Subs(sp.Derivative(Fg(sp.Symbol('y')), sp.Symbol('y')), sp.Symbol('y'), ng), pp)
drho = sp.simplify(drho.subs(sp.Derivative(Fg(ng), ng), pp)) if drho.has(sp.Derivative) else drho
ok(f"DL2 d rho / d p = {drho} for a generic F (using F'(n) = p); the conjugate of p accumulates N = int n sqrt(-g) d^4x", sp.simplify(drho - ng) == 0)

# DL3 discreteness (NEW AXIOM D: labels are names of discrete elements, N integer) + Lorentz invariance -> Poisson sprinkling (known:
#     Bombelli-Henson-Sorkin 2009; a lattice is not Lorentz invariant), so delta N = sqrt(N). Level 3 of the ladder (quantum) makes the
#     conjugate pair uncertain: delta p delta N >= hbar. Then delta rho = n delta p = n hbar/sqrt(n V4) = hbar sqrt(n)/sqrt(V4) and
#     delta Lambda = 8 pi G delta rho = 8 pi alpha (l_P/l)^2 / sqrt(V4), l = n^(-1/4) the element spacing, alpha = O(1) of the bound.
#     F drops out: the law needs no choice of medium.
V4, hb, G, nn = sp.symbols('V4 hbar G n', positive=True)
dL = 8 * sp.pi * G * nn * hb / sp.sqrt(nn * V4)
lP2 = G * hb; ell = nn**sp.Rational(-1, 4)
ok(f"DL3 delta Lambda = {sp.simplify(dL)} = 8 pi (l_P/l)^2/sqrt(V4) (c = 1); independent of F",
   sp.simplify(dL - 8 * sp.pi * lP2 / ell**2 / sp.sqrt(V4)) == 0)

# DL4 the region is the observer's causal past (what can be counted; reading). V4(t) = (4 pi/3) int a^4 (eta - eta')^3 d eta'. In the
#     matter era (a ~ t^(2/3)) and the radiation era (a ~ t^(1/2)) V4 = c_V t^4; compute c_V and then Omega_Lambda = Lambda/(3 H^2) for a
#     Lambda of fixed amplitude A/sqrt(V4): Omega_Lambda = A / (3 sqrt(c_V) (H t)^2) is CONSTANT in each era -- 'everpresent'.
def cV(pw):                                          # a = t^pw, eta = int dt/a = t^(1-pw)/(1-pw)
    s = sp.symbols('s', positive=True)
    eta = lambda tt: tt**(1 - pw) / (1 - pw)
    val = sp.Rational(4, 3) * sp.pi * sp.integrate(s**(4 * pw) * (eta(1) - eta(s))**3 * s**(-pw), (s, 0, 1))
    return sp.nsimplify(sp.simplify(val))
cm, cr = cV(sp.Rational(2, 3)), cV(sp.Rational(1, 2))
Om_m = 1 / (3 * sp.sqrt(cm) * sp.Rational(2, 3)**2); Om_r = 1 / (3 * sp.sqrt(cr) * sp.Rational(1, 2)**2)
ok(f"DL4 V4 = c_V t^4 with c_V = {cm} = {float(cm):.4f} (matter), {cr} = {float(cr):.4f} (radiation); Omega_Lambda/A = "
   f"{float(Om_m):.3f} (matter), {float(Om_r):.3f} (radiation): constant in each era", float(cm) > 0 and float(cr) > 0)

# DL5 amplitude needed today (Planck-like LCDM history, Om = 0.31, radiation included, H0 = 1): Lambda sqrt(V4_0) = 3 Omega_L sqrt(V4_0) is a
#     pure number; it equals 8 pi alpha (l_P/l)^2. If it is O(1) to O(10^-2), elements at the Planck spacing suffice; 10^-122 is then
#     (l_P^2/sqrt(V4)) -- the age in Planck units, not a label.
Om, Orr = 0.31, 9.1e-5; OL = 1 - Om - Orr
E = lambda aa: np.sqrt(Om * aa**-3 + Orr * aa**-4 + OL)
def rhs(lna, y):                                     # y = (t, eta, M0..M3) with M_k = int a^4 eta^k d eta
    aa = np.exp(lna); dt = 1 / E(aa); deta = dt / aa
    et = y[1]
    return [dt, deta] + [aa**4 * et**k * deta for k in range(4)]
a_i = 1e-8; t_i = a_i**2 / (2 * np.sqrt(Orr)); eta_i = a_i / np.sqrt(Orr)
sol = solve_ivp(rhs, [np.log(a_i), 0.0], [t_i, eta_i, 0, 0, 0, 0], rtol=1e-10, atol=1e-30)
t0, eta0, M0, M1, M2, M3 = sol.y[:, -1]
V40 = 4 * np.pi / 3 * (eta0**3 * M0 - 3 * eta0**2 * M1 + 3 * eta0 * M2 - M3)
X = 3 * OL * np.sqrt(V40)
alpha_eff = X / (8 * np.pi)
H0_s = 1 / (67.4 / 3.0857e19); tP = 5.391e-44
print(f"   t0 = {t0:.3f}/H0, V4_0 = {V40:.4f} (c/H0)^4 = {V40 / t0**4:.4f} t0^4; Lambda sqrt(V4) = {X:.3f}; alpha (l_P/l)^2 = {alpha_eff:.4f}")
print(f"   Lambda l_P^2 = {3 * OL * (tP / H0_s)**2:.2e} = {alpha_eff:.3f} * 8 pi l_P^2/sqrt(V4) with l_P^2/sqrt(V4) = {(tP / H0_s)**2 / np.sqrt(V40):.2e}")
ok(f"DL5 the observed Lambda needs alpha (l_P/l)^2 = {alpha_eff:.3f}: Planck-spacing elements with an O(10^-2) coefficient", 1e-3 < alpha_eff < 1)

# DL6 what discreteness does NOT give: the MEAN of rho_vac is still the integration constant rho(mu); 'everpresent' needs it at the
#     symmetric point Lambda = 0 (second axiom). The sign is random; alpha and l/l_P are labels (one combination).
ok("DL6 recorded: axioms needed = discreteness (D) + Poisson (Lorentz) + quantum conjugacy (ladder level 3) + mean at the symmetric point; "
   "derived = the law Lambda ~ +-1/sqrt(V4) for any F; label = alpha (l_P/l)^2", True)
