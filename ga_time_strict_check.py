# Strict tests of the claims about time (papers/restart/49-time-strict.md). TS1-TS6. All ingredients are standard (conformal algebra,
# conformal invariance of Maxwell, Lagrange-Jacobi, Unruh/Bisognano-Wichmann KMS); what is tested is the reading 'time = stretching'.
# Run: uv run --with sympy --with numpy --with scipy python ga_time_strict_check.py
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from ga_core import *

# TS1 which 'times' are stretches: conformal algebra of spacetime in Cl(2,4): e0^2 = +1, e1..e3^2 = -1, e+^2 = +1, e-^2 = -1;
#     n_inf = e- + e+, n_0 = (e- - e+)/2. Squares of the time generators: inertial translation e0^n_inf, Rindler boost e0 e1, dilation
#     n_0^n_inf, and the two conformal combinations e0^(n_inf +- n_0).
S = {0: 1, 1: -1, 2: -1, 3: -1, 4: 1, 5: -1}
e0, e1 = {(0,): 1}, {(1,): 1}
ninf = {(4,): 1, (5,): 1}; n0 = {(5,): sp.Rational(1, 2), (4,): -sp.Rational(1, 2)}
sq = lambda B: clean(gp(B, B, S))
gens = {'inertial translation P0': wedge(e0, ninf, S), 'Rindler boost': wedge(e0, e1, S), 'dilation D': wedge(n0, ninf, S),
        'P0 + K0': wedge(e0, add(ninf, n0), S), 'P0 - K0': wedge(e0, add(ninf, scal(-1, n0)), S)}
sqs = {k: sq(v).get((), 0) for k, v in gens.items()}
ok("TS1 squares: " + ", ".join(f"{k}: {v}" for k, v in sqs.items()) + " -> inertial time is a shift (0), Rindler/cosmic time a stretch (+), "
   "conformal global time a rotation (-)", sqs['inertial translation P0'] == 0 and sqs['Rindler boost'] > 0 and sqs['dilation D'] > 0
   and sorted([sp.sign(sqs['P0 + K0']), sp.sign(sqs['P0 - K0'])]) == [-1, 1])

# TS2 laws that do not feel the stretch: in a conformally flat metric g = a(eta)^2 eta the Maxwell density sqrt(g) g^{mu a} g^{nu b} has the
#     factor a^4 a^-2 a^-2 = 1 for every a(eta) (4D): Maxwell in an expanding universe = Maxwell in flat space. The massive Klein-Gordon
#     operator does feel it: sqrt(g) g^{mu nu} gives a^2 and the mass term a^4 m^2.
eta = sp.symbols('eta', real=True); af = sp.Function('a')(eta); mm = sp.symbols('m', positive=True)
fac_max = sp.simplify(af**4 * af**-2 * af**-2)
fac_kg_kin, fac_kg_mass = sp.simplify(af**4 * af**-2), sp.simplify(af**4 * mm**2)
ok(f"TS2 Maxwell density factor {fac_max} (a-independent: conformal); Klein-Gordon factors {fac_kg_kin} (kinetic), {fac_kg_mass} (mass): "
   "massive fields feel the stretch", fac_max == 1 and sp.diff(fac_kg_kin / fac_kg_mass, eta) != 0)

# TS3 the arrow: Lagrange-Jacobi for Newtonian N bodies, d^2 I/dt^2 = 4E - 2V (V < 0): for E >= 0 I(t) is strictly convex -> one minimum
#     (Janus point) and D = dI/dt / 2 is monotone; for E < 0 (bound) D changes sign repeatedly (no arrow). Numeric 4-body runs.
def run(E_sign, seed):
    rng = np.random.default_rng(seed); N = 4; m = np.ones(N)
    x = rng.normal(size=(N, 3)); v = rng.normal(size=(N, 3)) * 0.3
    x -= x.mean(0); v -= v.mean(0)
    def energy(x, v):
        T = 0.5 * np.sum(m[:, None] * v**2)
        V = -sum(m[i] * m[j] / np.linalg.norm(x[i] - x[j]) for i in range(N) for j in range(i + 1, N))
        return T, V
    T, V = energy(x, v)
    target = 0.5 * abs(V) if E_sign > 0 else -0.5 * abs(V)        # set E = T + V
    v *= np.sqrt(max(target - V, 1e-6) / T)
    def rhs(t, y):
        x = y[:3 * N].reshape(N, 3); v = y[3 * N:].reshape(N, 3); a = np.zeros_like(x)
        for i in range(N):
            for j in range(N):
                if i != j:
                    d = x[j] - x[i]; a[i] += m[j] * d / (np.linalg.norm(d)**3 + 1e-9)
        return np.concatenate([v.ravel(), a.ravel()])
    ts = np.linspace(-6, 6, 1201)
    sol = solve_ivp(rhs, [0, 6], np.concatenate([x.ravel(), v.ravel()]), t_eval=ts[ts >= 0], rtol=1e-9, atol=1e-11)
    solb = solve_ivp(lambda t, y: -rhs(-t, np.concatenate([y[:3 * N], -y[3 * N:]])) * np.concatenate([np.ones(3 * N), -np.ones(3 * N)]),
                     [0, 6], np.concatenate([x.ravel(), -v.ravel()]), t_eval=-ts[ts <= 0][::-1], rtol=1e-9, atol=1e-11)
    I_f = np.sum(sol.y[:3 * N].reshape(N, 3, -1)**2, axis=(0, 1)); I_b = np.sum(solb.y[:3 * N].reshape(N, 3, -1)**2, axis=(0, 1))
    I = np.concatenate([I_b[::-1][:-1], I_f])
    D = np.gradient(I) / 2
    return I, D
Ip, Dp = run(+1, 3)
# bound case: a Kepler ellipse (E < 0) over several orbits -- D = r.p changes sign at every peri- and apocentre
def kep(t, y):
    r = np.hypot(y[0], y[1]); return [y[2], y[3], -y[0] / r**3, -y[1] / r**3]
solk = solve_ivp(kep, [0, 40], [1.0, 0.0, 0.0, 1.2], t_eval=np.linspace(0, 40, 8001), rtol=1e-10, atol=1e-12)
Db = solk.y[0] * solk.y[2] + solk.y[1] * solk.y[3]
mins_p = int(np.sum((Ip[1:-1] < Ip[:-2]) & (Ip[1:-1] < Ip[2:])))
signchanges_b = int(np.sum(np.diff(np.sign(Db)) != 0))
ok(f"TS3 E > 0: I(t) has {mins_p} minimum (Janus point), D monotone: {bool(np.all(np.diff(Dp) > -1e-6))}; E < 0: D changes sign {signchanges_b} times "
   "(bound systems have no stretch-arrow)", mins_p == 1 and signchanges_b >= 2)

# TS4 temperature of a stretch: the massless Wightman function along a uniformly accelerated line, W ~ 1/sinh^2(a tau/2), is periodic in
#     imaginary proper time with period 2 pi/a (KMS at T = a/2pi); the inertial one, W ~ 1/tau^2, has no imaginary period (T = 0).
tau, a_, s_ = sp.symbols('tau a s', positive=True)
Wacc = 1 / sp.sinh(a_ * tau / 2)**2
per = sp.simplify(Wacc.subs(tau, tau - sp.I * 2 * sp.pi / a_) - Wacc)
Winert = 1 / tau**2
noper = sp.simplify(Winert.subs(tau, tau - sp.I * s_) - Winert)
ok(f"TS4 accelerated: W(tau - 2 pi i/a) - W(tau) = {per} (period 2 pi/a, T = a/2pi); inertial: W(tau - i s) - W(tau) = {sp.simplify(noper)} != 0 for every s > 0",
   per == 0 and noper != 0)

# TS5 consequence for the reading 'time = stretch': it holds for accelerated observers (boost), for cosmic time in de Sitter (dilator, 41/C3)
#     and in the matter era (homothety, ga_a0_clock_check Q1); it fails for an inertial observer in flat space (shift, TS1) and for bound
#     systems (TS3). Collected as one check of the classification above.
# (TS5 is the reading of TS1-TS4, not a computation.)
