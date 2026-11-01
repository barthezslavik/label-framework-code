# 130: geometry responds to matter from horizon thermodynamics (Jacobson 1995 route; equations borrowed, every step recomputed here).
# Ingredients inside the framework: Unruh temperature T = kappa/2pi of the boost (KMS, doc 49), entropy proportional to area (doc 58 EN4,
# Bekenstein-Hawking as input), energy flux of the labels T_ab (div T = 0, doc 87), Lambda as a label paired with the 4-volume (doc 88).
# JC1 Raychaudhuri focusing of a null congruence, checked symbolically in FLRW;  JC2 Clausius algebra dQ = T dS  =>  T_kk = (eta/2pi) R_kk;
# JC3 lemma: S_kk = 0 for every null k  =>  S = f g (rank 9 of 10);  JC4 Bianchi + conservation  =>  G + Lambda g = (2pi/eta) T with
# Lambda an integration constant, eta = 1/(4G);  JC5 FLRW: only the null-null equation is imposed, H^2 - 8piG rho/3 stays constant
# (Lambda/3 set by initial data);  JC6 the null-null content is exactly the trace-free (unimodular) equation = SDiff of the 4-volume (doc 88);
# JC7 the doc 87 counterexample (dust on a flat background) violates the Clausius relation;  JC8 area law from the entanglement of the
# stiff label step (free massless scalar, doc 89 IG5) on a radial lattice (Srednicki 1993 method): S = s (R/a)^2, s depends on the cutoff a;
# JC9 eta is a cutoff label: N fields give N s/(4 pi a^2), same N/a^2 scaling as Sakharov induced 1/G (doc 112); coefficients differ by
# regularisation.  Run: ~/.venvs/jev/bin/python ga_jacobson_check.py
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

# ---------- JC1 Raychaudhuri in FLRW: plane null congruence k = (1/a, 1/a^2, 0, 0), affine; cross-section area ~ a^2 ----------
t, x, y, z = sp.symbols('t x y z')
a = sp.Function('a')(t)
X = [t, x, y, z]
g = sp.diag(-1, a**2, a**2, a**2)
gi = g.inv()
Gam = [[[sum(gi[i, l] * (sp.diff(g[l, j], X[k]) + sp.diff(g[l, k], X[j]) - sp.diff(g[j, k], X[l])) for l in range(4)) / 2
         for k in range(4)] for j in range(4)] for i in range(4)]
def ricci(i, j):
    return sp.simplify(sum(sp.diff(Gam[k][i][j], X[k]) - sp.diff(Gam[k][i][k], X[j])
                           + sum(Gam[k][k][l] * Gam[l][i][j] - Gam[k][j][l] * Gam[l][i][k] for l in range(4)) for k in range(4)))
Ric = sp.Matrix(4, 4, lambda i, j: ricci(i, j))
kv = sp.Matrix([1 / a, 1 / a**2, 0, 0])
null = sp.simplify((kv.T * g * kv)[0])
geo = [sp.simplify(sp.diff(kv[i], t) * kv[0] + sum(Gam[i][j][k] * kv[j] * kv[k] for j in range(4) for k in range(4))) for i in range(4)]
Rkk = sp.simplify((kv.T * Ric * kv)[0])
H = sp.diff(a, t) / a
theta = sp.simplify(2 * H * kv[0])                       # d ln(area)/d lambda, d/d lambda = k^t d/dt
dtheta = sp.simplify(kv[0] * sp.diff(theta, t))
ok(f"JC1 FLRW plane null congruence: k null ({null}), affine geodesic ({geo}); R_kk = -2 Hdot/a^2 ({sp.simplify(Rkk + 2 * sp.diff(H, t) / a**2) == 0}) and Raychaudhuri "
   f"dtheta/dlambda = -theta^2/2 - R_kk holds (shear 0)",
   null == 0 and all(e == 0 for e in geo) and sp.simplify(Rkk + 2 * sp.diff(H, t) / a**2) == 0 and sp.simplify(dtheta + theta**2 / 2 + Rkk) == 0)
# local form used by the Clausius step: with theta = sigma = 0 at p, theta = -lambda R_kk and dA/A = -(lambda^2/2) R_kk to second order;
# check by integrating Raychaudhuri with theta(0) = 0 for a slowly varying R_kk(lambda) = R0 (1 + c lambda)
R0, c0 = 0.3, 0.2
sol = solve_ivp(lambda l, s: [-s[0]**2 / 2 - R0 * (1 + c0 * l), s[0]], [0, 0.05], [0.0, 0.0], rtol=1e-12, atol=1e-14, dense_output=True)
lam = 0.05
lnA = sol.sol(lam)[1]
ok(f"JC1b theta(0) = 0: ln(A/A0) at lambda = {lam} is {lnA:.6e}, -(lambda^2/2) R_kk = {-lam**2 / 2 * R0:.6e} (relative difference "
   f"{abs(lnA / (-lam**2 / 2 * R0) - 1):.1e}, O(lambda))", abs(lnA / (-lam**2 / 2 * R0) - 1) < 0.02)

# ---------- JC2 Clausius algebra on a local Rindler horizon ----------
lam_, kap, eta, Tkk, Rkk_ = sp.symbols('lambda kappa eta T_kk R_kk', real=True)
# boost Killing field near the bifurcation surface chi = -kappa lambda k; heat flux dQ = int T_ab chi^a k^b dlambda dA = -kappa lambda T_kk
# entropy dS = eta dA = eta int theta dlambda dA, theta = -lambda R_kk;  temperature T = kappa/2pi (KMS of the boost, doc 49)
dQ = sp.integrate(-kap * lam_ * Tkk, (lam_, -1, 0))      # flux through the horizon generators before p (lambda < 0, theta(p) = 0)
dS = eta * sp.integrate(-lam_ * Rkk_, (lam_, -1, 0))
rel = sp.solve(sp.Eq(dQ, kap / (2 * sp.pi) * dS), Tkk)[0]
ok(f"JC2 dQ = T dS with T = kappa/2pi gives T_kk = {rel}: kappa (the Unruh scale) cancels; holds for every null k at every point",
   sp.simplify(rel - eta / (2 * sp.pi) * Rkk_) == 0)

# ---------- JC3 lemma: symmetric S with S_kk = 0 for all null k is proportional to g ----------
eta4 = np.diag([-1.0, 1, 1, 1])
rng = np.random.default_rng(130)
rows = []
for _ in range(40):
    n = rng.normal(size=3); n /= np.linalg.norm(n)
    k = np.r_[1.0, n] * rng.uniform(0.5, 2)
    rows.append([k[i] * k[j] * (1 if i == j else 2) for i in range(4) for j in range(i, 4)])
M = np.array(rows)
sv = np.linalg.svd(M, compute_uv=False)
rank = int((sv > 1e-10 * sv[0]).sum())
kernel = np.linalg.svd(M)[2][-1]
Sk = np.zeros((4, 4)); idx = 0
for i in range(4):
    for j in range(i, 4):
        Sk[i, j] = Sk[j, i] = kernel[idx]; idx += 1
ok(f"JC3 null-null conditions on a symmetric 4x4 tensor have rank {rank} of 10; the kernel is g (max |S/S_00 - g/g_00| = "
   f"{np.abs(Sk / Sk[0, 0] - eta4 / eta4[0, 0]).max():.1e})", rank == 9 and np.abs(Sk / Sk[0, 0] - eta4 / eta4[0, 0]).max() < 1e-9)

# ---------- JC4 Bianchi + conservation fix f and leave Lambda as an integration constant ----------
# R_ab - (2pi/eta) T_ab = f g_ab.  Divergence: grad^a R_ab = (1/2) grad_b R (contracted Bianchi), grad^a T_ab = 0  =>  grad_b(R/2 - f) = 0
Rs, f, Lam, Gn = sp.symbols('R f Lambda G', real=True)
fsol = Rs / 2 - Lam                                         # R/2 - f = Lambda, a constant
# trace: R - (2pi/eta) T = 4 f  =>  consistent with f = R/2 - Lambda  iff  R = -(2pi/eta) T + 4 Lambda (the Einstein trace)
Ttr = sp.symbols('T', real=True)
trace_ok = sp.simplify((Rs - 2 * sp.pi / eta * Ttr - 4 * fsol).subs(Rs, -2 * sp.pi / eta * Ttr + 4 * Lam)) == 0
eta_G = sp.solve(sp.Eq(2 * sp.pi / eta, 8 * sp.pi * Gn), eta)[0]
ok(f"JC4 f = R/2 - Lambda: R_ab - R g_ab/2 + Lambda g_ab = (2pi/eta) T_ab; Lambda a constant of integration; trace consistent; "
   f"Newton's normalisation needs eta = {eta_G}", trace_ok and sp.simplify(eta_G - 1 / (4 * Gn)) == 0)

# ---------- JC5 FLRW: impose only the null-null equation, Lambda appears as initial data ----------
# null-null: R_kk = 8piG T_kk  <=>  -2 Hdot = 8piG (rho + p) (from JC1, T_kk = (rho + p)/a^2); conservation rho_dot = -3H(rho + p)
Gnum = 1.0 / (8 * np.pi)
def rhs(tt, s):
    lna, Hh, rm, rr = s
    return [Hh, -4 * np.pi * Gnum * (rm + 4 * rr / 3), -3 * Hh * rm, -4 * Hh * rr]
consts = []
for Lam_in in (0.0, 0.7, -0.2):
    rm0, rr0 = 0.3, 1e-4
    H0 = np.sqrt(8 * np.pi * Gnum * (rm0 + rr0) / 3 + Lam_in / 3)
    s = solve_ivp(rhs, [0, 3], [0, H0, rm0, rr0], rtol=1e-11, atol=1e-13, dense_output=True)
    tt = np.linspace(0, 3, 50)
    lna, Hh, rm, rr = s.sol(tt)
    C = Hh**2 - 8 * np.pi * Gnum * (rm + rr) / 3
    consts.append((Lam_in, C.min() * 3, C.max() * 3))
    print(f"   initial Lambda {Lam_in:+.2f}: 3(H^2 - 8piG rho/3) over t in [0,3] stays in [{C.min() * 3:.10f}, {C.max() * 3:.10f}]")
ok("JC5 spatially flat FLRW with the null-null equation only: H^2 - 8piG rho/3 = Lambda/3 is conserved, its value is whatever the initial data put in",
   all(abs(lo - L) < 1e-8 and abs(hi - L) < 1e-8 for L, lo, hi in consts))
Hs, rho, pr = sp.symbols('H rho p', real=True)
ddt = 2 * Hs * (-4 * sp.pi * Gn * (rho + pr)) - 8 * sp.pi * Gn / 3 * (-3 * Hs * (rho + pr))
ok(f"JC5b symbolic: d/dt (H^2 - 8piG rho/3) = {sp.simplify(ddt)} given the null-null equation and conservation", sp.simplify(ddt) == 0)
# with spatial curvature K the null-null equation reads -2 (Hdot - K/a^2) = 8piG (rho + p); the conserved quantity is then
# 3 (H^2 + K/a^2) - 8piG rho (not 3 H^2 - 8piG rho)
Kc, av = sp.symbols('K a', real=True)
ddtK = 2 * Hs * (-4 * sp.pi * Gn * (rho + pr) + Kc / av**2) - 2 * Kc / av**2 * Hs - 8 * sp.pi * Gn / 3 * (-3 * Hs * (rho + pr))
ok(f"JC5c curved FLRW: d/dt (H^2 + K/a^2 - 8piG rho/3) = {sp.simplify(ddtK)}, so Lambda/3 = H^2 + K/a^2 - 8piG rho/3", sp.simplify(ddtK) == 0)

# ---------- JC6 null-null content = trace-free (unimodular) Einstein equation ----------
# S_kk = 0 for all null k  <=>  S - (tr S/4) g = 0 (JC3). With S = R - 8piG T: R_ab - R g/4 = 8piG (T_ab - T g/4), the trace-free equation
# of unimodular gravity, i.e. the field equation for SDiff of the 4-volume; Lambda (the 4-volume conjugate, doc 88 LD6) enters only via JC4.
Ssym = sp.Matrix(4, 4, lambda i, j: sp.Symbol(f's{min(i, j)}{max(i, j)}'))
etaM = sp.diag(-1, 1, 1, 1)
trS = sum((etaM.inv() * Ssym)[i, i] for i in range(4))
TF = Ssym - trS / 4 * etaM
ksym = sp.symbols('n1 n2 n3', real=True)
kk = sp.Matrix([1, *ksym])
on_cone = sp.expand((kk.T * TF * kk)[0] - (kk.T * Ssym * kk)[0])
null_cond = sp.simplify(on_cone.subs(ksym[2], sp.sqrt(1 - ksym[0]**2 - ksym[1]**2)))
ok(f"JC6 S_kk = (S_tracefree)_kk on the null cone (difference {null_cond}); with JC3 the Clausius condition is exactly the trace-free "
   "equation, the field equation for SDiff of the 4-volume (doc 88)", null_cond == 0)

# ---------- JC7 the doc 87 counterexample violates Clausius ----------
# dust on a fixed flat background: R_kk = 0, T_kk = rho (u.k)^2 > 0 for every null k
u = np.array([1.0, 0, 0, 0]); kn = np.array([1.0, 0.6, 0.0, 0.8])
Tkk_dust = 1.0 * (u @ eta4 @ kn)**2
ok(f"JC7 dust (rho = 1) on flat space: T_kk = {Tkk_dust:.2f}, R_kk = 0, so dQ = T dS fails on every local Rindler horizon: the MG1 "
   "counterexample is excluded by the horizon axiom, which is therefore independent of the label axioms", Tkk_dust > 0)

# ---------- JC8 area law from the entanglement of the stiff label step (free massless scalar), radial lattice ----------
def ent_sphere(n, Nsites, lmax):
    j = np.arange(1, Nsites + 1, dtype=float)
    S = 0.0
    for l in range(lmax + 1):
        K = np.zeros((Nsites, Nsites))
        K[np.arange(Nsites), np.arange(Nsites)] = (l * (l + 1) / j**2 + ((j + 0.5)**2 + (j - 0.5)**2) / j**2)
        K[0, 0] = l * (l + 1) + 2.25                           # j = 1: (j - 1/2)^2 term with phi_0 absent is (1/2)^2/1 -> Srednicki: 9/4
        off = -(j[:-1] + 0.5)**2 / (j[:-1] * j[1:])
        K[np.arange(Nsites - 1), np.arange(1, Nsites)] = off
        K[np.arange(1, Nsites), np.arange(Nsites - 1)] = off
        w, V = np.linalg.eigh(K)
        Om = (V * np.sqrt(w)) @ V.T
        Xm = 0.5 * (V / np.sqrt(w)) @ V.T
        Pm = 0.5 * Om
        nu = np.sqrt(np.clip(np.linalg.eigvals(Xm[:n, :n] @ Pm[:n, :n]).real, 0.25, None))
        nu = nu[nu > 0.5 + 1e-12]
        Sl = np.sum((nu + 0.5) * np.log(nu + 0.5) - (nu - 0.5) * np.log(nu - 0.5))
        S += (2 * l + 1) * Sl
        if l > 50 and (2 * l + 1) * Sl < 1e-9 * S:
            break
    return S
Nsites = 60
ns = np.arange(8, 31, 4)
Sv = np.array([ent_sphere(n, Nsites, 4000) for n in ns])
Rr = ns + 0.5
cA = np.polyfit(Rr**2, Sv, 1)
resid = np.abs(Sv - np.polyval(cA, Rr**2)).max() / Sv.max()
cV = np.polyfit(Rr**3, Sv, 1)
residV = np.abs(Sv - np.polyval(cV, Rr**3)).max() / Sv.max()
for n_, s_ in zip(ns, Sv):
    print(f"   n = {n_:2d} (R = {n_ + 0.5:4.1f} a): S = {s_:8.2f}, S/R^2 = {s_ / (n_ + 0.5)**2:.4f}")
s_coef = cA[0]
ok(f"JC8 ground-state entanglement of a sphere of the c = 1 label step: S = {s_coef:.3f} (R/a)^2 + {cA[1]:.2f} (area fit residual "
   f"{resid:.1e}, volume fit residual {residV:.1e}); area law, coefficient in units of the cutoff a", resid < 0.01 and residV > 3 * resid
   and 0.25 < s_coef < 0.35)

# ---------- JC9 eta is a cutoff label ----------
eta_ent = s_coef / (4 * np.pi)                                # entropy per area, per field, in units 1/a^2
a_over_lP = np.sqrt(s_coef / np.pi)                           # eta = 1/(4 l_P^2) for one field
eta_sak = 1 / (24 * np.pi)                                    # Sakharov (doc 112): 1/(4G) = 4 pi N/(96 pi^2 l^2) = N/(24 pi l^2)
print(f"   sphere entanglement: eta_sphere = N x {eta_ent:.4f}/a^2;  Sakharov induced: 1/4G = N x {eta_sak:.4f}/l^2;  ratio {eta_ent / eta_sak:.2f} "
      f"(regularisation dependent); one field gives eta_sphere = 1/4 l_P^2 only if a is chosen as {a_over_lP:.2f} l_P")
ok(f"JC9 eta_sphere = N s/(4 pi a^2) is fixed by the cutoff and the field count, not by the label axioms; it scales as N/a^2 like the induced "
   f"1/G (doc 112), so the species count enters both, and their coefficients differ ({eta_ent / eta_sak:.2f}) by regularisation: "
   "G stays a label", eta_ent > 0 and abs(eta_ent / eta_sak - 1) > 0.05)
