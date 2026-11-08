# 152/154: paper13, step 2. Entropy as a count of labels. Densities on the torus T^2; a flow map acts by pushforward (relabellings are
# gauge and leave the physical density unchanged, paper1).
# SE1 under volume-preserving flow maps (compositions of shears, unit Jacobian) every functional int f(rho) is invariant: f = rho ln rho,
#     rho^2, rho^3 (the Casimirs of SDiff); the fine-grained entropy is one of them;
# SE2 under a flow map that changes volume only the label count int rho is invariant; int rho ln rho, rho^2, rho^3 change: general flows
#     keep no entropy functional among the int f(rho);
# SE3 int rho ln(rho/mu) is invariant under every diffeomorphism when the reference density mu is pushed forward too: the entropy of
#     the series is the relative entropy to the label volume;
# SE4 Boltzmann count: N labels placed in K cells of volumes v_k with occupation fractions p_k; (1/N) ln W -> -sum p ln(p/v) (Stirling)
#     as N grows (N = 1e2 .. 1e6): the relative entropy to the label volume is the logarithm of the number of label arrangements.
# Run: ~/.venvs/jev/bin/python ga_stat_entropy_check.py
import numpy as np
from scipy.special import gammaln

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

n = 256
g = 2 * np.pi * np.arange(n) / n; dA = (2 * np.pi / n)**2
X, Y = np.meshgrid(g, g, indexing='ij')
def rho_f(x, y):
    return np.exp(1.2 * np.cos(x) + 0.8 * np.sin(y + 0.3) + 0.5 * np.cos(x + 2 * y))
Z = rho_f(X, Y).sum() * dA
rho = lambda x, y: rho_f(x, y) / Z
mu0 = lambda x, y: np.ones_like(x) / (2 * np.pi)**2

# volume-preserving map: phi = S2 o S1, S1(x,y) = (x + a sin y, y), S2(x,y) = (x, y + b sin(x + 0.4)); inverse is explicit
a, b = 0.9, 0.7
def push_sdiff(f):
    y1 = Y - b * np.sin(X + 0.4)                       # invert S2
    x1 = X - a * np.sin(y1)                            # invert S1
    return f(x1, y1)                                   # unit Jacobian
# volume-changing product map: x -> x + c sin(x), y -> y + d sin(y + 0.2) (circle diffeomorphisms, |c|,|d| < 1)
c, d = 0.6, -0.5
def inv(yv, cc, sh):
    x = yv.copy()
    for _ in range(80):
        x -= (x + cc * np.sin(x + sh) - yv) / (1 + cc * np.cos(x + sh))
    return x
def push_diff(f):
    x1 = inv(X, c, 0.0); y1 = inv(Y, d, 0.2)
    J = (1 + c * np.cos(x1)) * (1 + d * np.cos(y1 + 0.2))
    return f(x1, y1) / J

r = rho(X, Y); rs = push_sdiff(rho); rd = push_diff(rho)
F = {'mass': lambda q: q, 'rho ln rho': lambda q: q * np.log(q), 'rho^2': lambda q: q**2, 'rho^3': lambda q: q**3}
I = lambda q, f: (f(q)).sum() * dA
dev_s = {k: abs(I(rs, f) / I(r, f) - 1) for k, f in F.items()}
dev_d = {k: abs(I(rd, f) / I(r, f) - 1) for k, f in F.items()}
ok("SE1 volume-preserving flow map: relative change of " + ", ".join(f"{k} {v:.1e}" for k, v in dev_s.items())
   + "; every int f(rho) is invariant (Casimirs of volume-preserving flows)", max(dev_s.values()) < 1e-9)
ok("SE2 volume-changing flow map: " + ", ".join(f"{k} {v:.1e}" for k, v in dev_d.items())
   + "; only the label count is invariant", dev_d['mass'] < 1e-9 and min(v for k, v in dev_d.items() if k != 'mass') > 1e-3)

mu_d = push_diff(mu0)
rel0 = (r * np.log(r / mu0(X, Y))).sum() * dA
rel1 = (rd * np.log(rd / mu_d)).sum() * dA
ok(f"SE3 relative entropy to the label volume: {rel0:.8f} before, {rel1:.8f} after a volume-changing map with the reference pushed too; "
   f"the entropy of the series is relative to the label volume", abs(rel1 / rel0 - 1) < 1e-9)

# SE4 Boltzmann count
p = np.array([0.4, 0.3, 0.2, 0.1]); v = np.array([0.25, 0.25, 0.3, 0.2])
target = -(p * np.log(p / v)).sum()
rows = []
for N in (10**2, 10**3, 10**4, 10**5, 10**6):
    k = np.round(p * N).astype(int); k[-1] = N - k[:-1].sum()
    lnW = gammaln(N + 1) - gammaln(k + 1).sum() + (k * np.log(v)).sum()   # multinomial count weighted by cell volumes
    rows.append((N, lnW / N))
errs = [abs(x - target) for _, x in rows]
ok("SE4 (1/N) ln W for N labels in cells: " + ", ".join(f"N={N:.0e}: {x:.6f}" for N, x in rows) + f" -> -sum p ln(p/v) = {target:.6f}",
   errs[-1] < 1e-4 and all(errs[i + 1] < errs[i] for i in range(len(errs) - 1)))

print("ALL PASS")
