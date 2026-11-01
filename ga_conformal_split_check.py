# 132: the 15 conformal generators as phase-space functions written by the geometric product, and their action on the volume I.
# Generators (Minkowski, eta = diag(-1,1,1,1), phase space (x, p)): P = p, M = <x p>_2 (Lorentz), D = <x p>_0 = x.p (dilation),
# K = x p x = 2 (x.p) x - x^2 p (transversions). CS1 the GA forms: <xp>_0 = x.p, <xp>_2 = x^p, xpx = 2(x.p)x - x^2 p (Dirac matrices);
# CS2 every generator is conformal Killing, L_xi g = 2 sigma g, and div xi = 4 sigma; CS3 sigma = 0 (volume I kept) exactly for P and M
# (Poincare, 10), sigma != 0 for D and K (5); CS4 the 15 functions close under the Poisson bracket (dimension 15), the Killing form has
# signature (8 positive, 7 negative) as so(2,4); CS5 dilation grading {D, P} = -P... weights -1, 0, +1 for P, (M, D), K;
# CS6 the scalar part of xp generates x -> lambda x, p -> p/lambda (I scaled by lambda^4), the bivector part preserves x^2 and I.
# Run: ~/.venvs/jev/bin/python ga_conformal_split_check.py
import itertools
import numpy as np
import sympy as sp

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

eta = sp.diag(-1, 1, 1, 1)
X = sp.symbols('x0:4', real=True)
Pm = sp.symbols('p0:4', real=True)
xl = [sum(eta[m, n] * X[n] for n in range(4)) for m in range(4)]          # x_mu
dot = lambda u, v: sum(eta[m, m] * u[m] * v[m] for m in range(4))
xdotp = sum(X[m] * Pm[m] for m in range(4))                             # x^mu p_mu (p with lower index)
x2 = dot(X, X)

# ---------- CS1 GA forms with Dirac matrices (Cl(1,3) with gamma_0^2 = -1, gamma_i^2 = +1) ----------
s = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]
I2, Z2 = np.eye(2), np.zeros((2, 2))
g0 = 1j * np.block([[I2, Z2], [Z2, -I2]])                               # (i gamma^0_Dirac)^2 = -1
gi = [np.block([[Z2, si], [-si, Z2]]) * 1j for si in s]                 # (i gamma^i)^2 = +1
gam = [g0] + gi
ok_cl = all(np.allclose(gam[a] @ gam[b] + gam[b] @ gam[a], 2 * float(eta[a, b]) * np.eye(4)) for a in range(4) for b in range(4))
rng = np.random.default_rng(132)
errs = []
for _ in range(20):
    xv, pv = rng.normal(size=4), rng.normal(size=4)
    xm = sum(xv[m] * gam[m] for m in range(4)); pm = sum(pv[m] * gam[m] for m in range(4))
    xp = xm @ pm
    sc = np.trace(xp).real / 4
    biv = (xm @ pm - pm @ xm) / 2
    e1 = abs(sc - (-xv[0] * pv[0] + xv[1:] @ pv[1:]))
    e2 = np.abs(xp - sc * np.eye(4) - biv).max()                        # xp = scalar + bivector, nothing else
    xdp = -xv[0] * pv[0] + xv[1:] @ pv[1:]; xx = -xv[0]**2 + xv[1:] @ xv[1:]
    e3 = np.abs(xm @ pm @ xm - (2 * xdp * xm - xx * pm)).max()
    errs.append(max(e1, e2, e3))
ok(f"CS1 Cl(1,3) by Dirac matrices ({ok_cl}): xp = x.p + x^p exactly, x p x = 2(x.p)x - x^2 p (max error {max(errs):.1e})", ok_cl and max(errs) < 1e-12)

# ---------- the 15 generators as phase-space functions (p_mu lower) ----------
gens = {}
for m in range(4):
    gens[f'P{m}'] = Pm[m]
for m, n in itertools.combinations(range(4), 2):
    gens[f'M{m}{n}'] = xl[m] * Pm[n] - xl[n] * Pm[m]                     # x_m p_n - x_n p_m: components of x^p
gens['D'] = xdotp
for m in range(4):
    gens[f'K{m}'] = 2 * xl[m] * xdotp - x2 * Pm[m]                       # components of x p x
def pb(f, g):
    return sp.expand(sum(sp.diff(f, X[m]) * sp.diff(g, Pm[m]) - sp.diff(f, Pm[m]) * sp.diff(g, X[m]) for m in range(4)))

# ---------- CS2-CS3 vector fields xi^mu = dG/dp_mu, conformal Killing equation, divergence ----------
rows = []
allck = True
for name, G in gens.items():
    xi = [sp.diff(G, Pm[m]) for m in range(4)]                          # xi^mu
    xi_l = [sum(eta[m, n] * xi[n] for n in range(4)) for m in range(4)]
    div = sp.simplify(sum(sp.diff(xi[m], X[m]) for m in range(4)))
    sigma = sp.simplify(div / 4)
    ck = all(sp.simplify(sp.diff(xi_l[m], X[n]) + sp.diff(xi_l[n], X[m]) - 2 * sigma * eta[m, n]) == 0 for m in range(4) for n in range(4))
    allck &= ck
    rows.append((name, sigma))
keep = [n_ for n_, sg in rows if sg == 0]
scale = [n_ for n_, sg in rows if sg != 0]
for n_, sg in rows:
    print(f"   {n_:4s}: sigma = {sg}")
ok("CS2 all 15 are conformal Killing fields: L_xi g = 2 sigma g with div xi = 4 sigma", allck and len(rows) == 15)
ok(f"CS3 volume kept (sigma = 0): {keep} = P and M, {len(keep)} generators (Poincare); volume rescaled: {scale} = D and K, {len(scale)}",
   sorted(keep) == sorted([k for k in gens if k[0] in 'PM']) and sorted(scale) == sorted([k for k in gens if k[0] in 'DK']))

# ---------- CS4 closure and Killing form ----------
names = list(gens)
monos = sorted({mo for G in gens.values() for mo in sp.Poly(G, *X, *Pm).monoms()})
def coeffs(f):
    if f == 0: return [0] * len(allmonos)
    P_ = sp.Poly(f, *X, *Pm)
    d = dict(zip(P_.monoms(), P_.coeffs()))
    return [d.get(mo, 0) for mo in allmonos]
brackets = {(a_, b_): pb(gens[a_], gens[b_]) for a_ in names for b_ in names}
allmonos = sorted({mo for f in list(gens.values()) + [f for f in brackets.values() if f != 0] for mo in sp.Poly(f, *X, *Pm).monoms()})
B = sp.Matrix([coeffs(gens[n_]) for n_ in names]).T                     # columns = generators
struct = {}
closed = True
for (a_, b_), f in brackets.items():
    v = sp.Matrix(coeffs(f))
    sol = sp.Matrix(B.solve_least_squares(v)) if f != 0 else sp.zeros(15, 1)
    if (B * sol - v).norm() != 0: closed = False
    struct[a_, b_] = sol
ad = {a_: np.array([[float(struct[a_, b_][c_]) for b_ in names] for c_ in range(15)]) for a_ in names}
Kf = np.array([[np.trace(ad[a_] @ ad[b_]) for b_ in names] for a_ in names])
ev = np.linalg.eigvalsh(Kf)
npos, nneg = int((ev > 1e-9).sum()), int((ev < -1e-9).sum())
ok(f"CS4 the 15 functions close under the Poisson bracket ({closed}, rank {B.rank()}); Killing form signature ({npos} +, {nneg} -) = so(2,4)",
   closed and B.rank() == 15 and (npos, nneg) == (8, 7))

# ---------- CS5 dilation grading ----------
wP = all(sp.simplify(pb(gens['D'], gens[f'P{m}']) - gens[f'P{m}']) == 0 for m in range(4))
wK = all(sp.simplify(pb(gens['D'], gens[f'K{m}']) + gens[f'K{m}']) == 0 for m in range(4))
wM = all(pb(gens['D'], gens[k]) == 0 for k in gens if k[0] == 'M')
ok(f"CS5 grading by D: {{D, P}} = +P ({wP}), {{D, M}} = 0 ({wM}), {{D, K}} = -K ({wK}): P, (M, D), K carry opposite and zero dilation weight", wP and wK and wM)

# ---------- CS6 flows: scalar part scales x, p and I; bivector part keeps x^2 and I ----------
dx = [pb(X[m], gens['D']) for m in range(4)]; dp = [pb(Pm[m], gens['D']) for m in range(4)]
scal = all(sp.simplify(dx[m] + X[m]) == 0 or sp.simplify(dx[m] - X[m]) == 0 for m in range(4))
sign = sp.simplify(dx[0] / X[0])
pres = all(sp.simplify(pb(x2, gens[k])) == 0 for k in gens if k[0] == 'M')
ok(f"CS6 <xp>_0 = D moves x -> x + {sign} x eps, p -> p - {sign} p eps (I scaled by (1 + 4 eps)), the six components of <xp>_2 keep x^2 "
   f"({pres}) and have sigma = 0; the split of the single product xp into scalar + bivector is the split rescale / keep", scal and pres)
