# 155/156: paper14, step 5. Link with the series: what the symmetry fixes (forms and counts) and what stays a label (values).
# CX1 for 300 random stable values of the moduli of each medium (linear elasticity at an unstressed background, the quadratic forms of EL2)
#     the mode counts at a generic direction (fluid (1,2), smectic (2,1), columnar (3,0), isotropic solid (3,0)), the bound
#     c_T/c_L < sqrt(3)/2 of the isotropic solid and the sin^2 cos^2 form of the smectic second sound do not change, while the speeds
#     spread over the draws: forms and counts are fixed by G, values are labels;
# CX2 the solid with shear modulus mu -> 0 tends to the fluid of paper1: c_T -> 0 linearly in sqrt(mu) and c_L^2 -> K/rho; restoring SDiff
#     restores the two omega = 0 transverse modes;
# CX3 the superfluid of Son (2002), P(X) with X = (d theta)^2, is the medium of one timelike label of paper1, F(n) = P(n^2): Son's
#     rho = 2 X P_X - P and p = P equal paper1's rho = n F' - F and p_perp = F for three test functions (symbolic).
# Run: ~/.venvs/jev/bin/python ga_cont_series_check.py
import numpy as np
import sympy as sp

rng = np.random.default_rng(5)

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

def energy(med, p, e):
    tr = np.trace(e); dev = e - tr / 3 * np.eye(3)
    if med == 'fluid': return 0.5 * p[0] * tr**2
    if med == 'smectic': return 0.5 * p[0] * tr**2 + 0.5 * p[1] * e[2, 2]**2 + p[2] * tr * e[2, 2]
    if med == 'columnar':
        ep = e[0, 0] + e[1, 1]
        return 0.5 * p[0] * tr**2 + 0.5 * p[1] * ep**2 + p[2] * tr * ep + 0.5 * p[3] * ((e[0, 0] - e[1, 1])**2 + 4 * e[0, 1]**2)
    if med == 'iso': return 0.5 * p[0] * tr**2 + p[1] * np.sum(dev * dev)
def draw(med):
    while True:
        if med == 'fluid': p = rng.uniform(0.2, 5, 1)
        elif med == 'iso': p = rng.uniform(0.2, 5, 2)
        elif med == 'smectic':
            p = np.r_[rng.uniform(0.2, 5, 2), rng.uniform(-2, 2)]
            if p[0] * p[1] - p[2]**2 <= 0.05: continue
        else:
            p = np.r_[rng.uniform(0.2, 5, 2), rng.uniform(-2, 2), rng.uniform(0.2, 5)]
            if p[0] * p[1] - p[2]**2 <= 0.05: continue
        return p
def dyn(med, p, k):
    # M_ab = d^2 E / du_a du_b with eps = sym(u k^T)
    M = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            ua, ub = np.eye(3)[a], np.eye(3)[b]
            s = lambda u: 0.5 * (np.outer(u, k) + np.outer(k, u))
            M[a, b] = 0.25 * (energy(med, p, s(ua + ub)) * 2 - energy(med, p, s(ua - ub)) * 2)
    return np.sort(np.linalg.eigvalsh(M))
def kv(th, ph=0.7): return np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
def cnt(w2): return int((w2 > 1e-10).sum()), int((np.abs(w2) <= 1e-10).sum())

want = {'fluid': (1, 2), 'smectic': (2, 1), 'columnar': (3, 0), 'iso': (3, 0)}
counts_ok, cL_spread, rmax, form_dev, form_const = True, {}, 0.0, 0.0, []
for med in want:
    cls = []
    for _ in range(300):
        p = draw(med); w2 = dyn(med, p, kv(0.6))
        counts_ok &= cnt(w2) == want[med]; cls.append(np.sqrt(w2.max()))
        if med == 'iso': rmax = max(rmax, np.sqrt(w2[0] / w2[-1]))
        if med == 'smectic':
            pr = [np.prod(dyn(med, p, kv(t))[1:]) / (np.sin(t)**2 * np.cos(t)**2) for t in (0.3, 0.7, 1.2)]
            form_dev = max(form_dev, (max(pr) - min(pr)) / np.mean(pr)); form_const.append(np.mean(pr))
    cL_spread[med] = max(cls) / min(cls)
ok("CX1 300 random stable moduli per medium: mode counts unchanged (" + ", ".join(f"{m} {want[m]}" for m in want) + f"); isotropic "
   f"c_T/c_L <= {rmax:.4f} < 0.8660; smectic omega_1^2 omega_2^2 / sin^2 cos^2 constant over angles to {form_dev:.0e}, while that constant "
   f"varies by a factor {max(form_const) / min(form_const):.0f} over draws; fastest speed varies by a factor " +
   ", ".join(f"{m} {v:.1f}" for m, v in cL_spread.items()), counts_ok and rmax < np.sqrt(0.75) and form_dev < 1e-9 and
   min(cL_spread.values()) > 2)

K = 2.0; rows = []
for mu in [1e-1, 1e-2, 1e-4, 0.0]:
    w2 = dyn('iso', [K, mu], kv(0.6)); rows.append((mu, np.sqrt(max(w2[0], 0)), w2[-1]))
lin = [ct / np.sqrt(mu) for mu, ct, _ in rows if mu > 0]
ok("CX2 isotropic solid as mu -> 0 (K = 2, rho = 1): " + "; ".join(f"mu = {mu:g}: c_T = {ct:.4g}, c_L^2 = {cl2:.4g}" for mu, ct, cl2 in rows) +
   f"; c_T/sqrt(mu) = {', '.join(f'{x:.4f}' for x in lin)}; at mu = 0 the fluid counts {cnt(dyn('iso', [K, 0.0], kv(0.6)))}",
   max(abs(x - 1) for x in lin) < 1e-9 and abs(rows[-1][2] - K) < 1e-12 and cnt(dyn('iso', [K, 0.0], kv(0.6))) == (1, 2))

X, n = sp.symbols('X n', positive=True)
tests = [X**2, X + sp.Rational(1, 3) * X**3, sp.sqrt(X) * (1 + X)]
good = True
for P in tests:
    rho_son = 2 * X * sp.diff(P, X) - P; p_son = P
    F = P.subs(X, n**2)
    rho_l1 = n * sp.diff(F, n) - F; p_l1 = F
    good &= sp.simplify((rho_son - rho_l1.subs(n, sp.sqrt(X)))) == 0 and sp.simplify(p_son - p_l1.subs(n, sp.sqrt(X))) == 0
ok("CX3 Son's superfluid P(X) is paper1's single timelike label with F(n) = P(n^2): rho and p agree for X^2, X + X^3/3, sqrt(X)(1 + X)", good)

print("ALL PASS")
