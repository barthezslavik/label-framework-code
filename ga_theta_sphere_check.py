# theta = sqrt2/(2 pi) as a circle on the Euclidean de Sitter sphere (papers/restart/31-theta-sphere.md). Q1-Q5.
# Euclidean dS_4 = S^4 of radius l = 1/H in R^5. A uniformly accelerated observer is the orbit of a rotor R = exp(-B tau/2),
# B a unit bivector; the orbit radius is |X.B|, its offset from the plane is |X^B|. T = 1/(2 pi * orbit radius).
# Run: uv run --with sympy python ga_theta_sphere_check.py
import sympy as sp
from ga_core import *

E5 = {k: 1 for k in range(5)}
l, rho, tau = sp.symbols('l rho tau', positive=True)
d = sp.sqrt(l**2 - rho**2)
B = {(0, 4): 1}                                                    # unit bivector of the orbit plane, B^2 = -1
R = add({(): sp.cos(tau / 2)}, scal(-sp.sin(tau / 2), B))
X0 = vec([rho, d, 0, 0, 0])
X = clean({k: sp.simplify(v) for k, v in gp(gp(R, X0, E5), {k: (-v if len(k) == 2 else v) for k, v in R.items()}, E5).items()})
norm2 = lambda V: sp.simplify(gp(V, V, E5).get((), 0))
XdotB = grade(gp(X0, B, E5), 1); XwB = grade(gp(X0, B, E5), 3)
ok(f"Q1 rotor orbit X(tau) = R X0 R~ stays on S^4 (|X|^2 = {norm2(X)}), is a circle of radius |X.B| = {sp.sqrt(norm2(XdotB))} "
   f"at offset |X^B| = {sp.sqrt(sp.simplify(-gp(XwB, XwB, E5).get((), 0)))}; period 2 pi -> T = 1/(2 pi rho)",
   sp.simplify(norm2(X) - l**2) == 0 and sp.simplify(norm2(XdotB) - rho**2) == 0
   and sp.simplify(-gp(XwB, XwB, E5).get((), 0) - d**2) == 0 and sp.simplify(X.get((0,), 0).subs(tau, 2 * sp.pi) - rho) == 0)

# Lorentzian continuation (the same worldline on the hyperboloid, cf. inv_misc_check M4): intrinsic a^2 = 1/rho^2 - 1/l^2.
t = sp.symbols('t', positive=True)
Xl = sp.Matrix([rho * sp.sinh(t / rho), d, 0, 0, rho * sp.cosh(t / rho)]); G = sp.diag(-1, 1, 1, 1, 1)
ip = lambda u, v: (u.T * G * v)[0]
A = Xl.diff(t, 2); At = A - ip(A, Xl) / l**2 * Xl
a2 = sp.simplify(ip(At, At))
H = 1 / l
ok(f"Q2 a^2 = {a2} = 1/rho^2 - H^2, so a = H tan(chi) with tan(chi) = |X^B|/|X.B|; a = H <=> |X.B| = |X^B| (chi = 45 deg) "
   f"<=> Unruh part a/2pi = horizon part H/2pi; there T/H = {sp.nsimplify(sp.simplify(l / (2 * sp.pi * l / sp.sqrt(2))))} = sqrt2/(2 pi) = theta",
   sp.simplify(a2 - (1 / rho**2 - H**2)) == 0
   and sp.solve(sp.Eq(a2, H**2), rho) == [l / sp.sqrt(2)]
   and sp.simplify(l / (2 * sp.pi * (l / sp.sqrt(2))) - sp.sqrt(2) / (2 * sp.pi)) == 0)

# Q3 candidate selection by a minimal product: the family S^1(r) x S^{n-2}(sqrt(1-r^2)) in the unit S^n; its volume is extremal
# (Clifford-type minimal hypersurface) at r^2 = 1/(n-1). Only n = 3 gives r = 1/sqrt2; our Euclidean dS_4 is n = 4.
r = sp.symbols('r', positive=True)
sel = {}
for n in (3, 4, 5):
    V = r * (1 - r**2) ** sp.Rational(n - 2, 2)
    sel[n] = [s for s in sp.solve(sp.diff(V, r), r) if s.is_positive and s < 1][0]
ok(f"Q3 minimal S^1 x S^(n-2) in S^n: r = {sel}; n=3 (Clifford torus) gives 1/sqrt2 -> theta; n=4 (our S^4) gives 1/sqrt3 -> "
   f"T/H = sqrt3/(2 pi) = {float(sp.sqrt(3) / (2 * sp.pi)):.4f}, not theta",
   sel[3] == 1 / sp.sqrt(2) and sel[4] == 1 / sp.sqrt(3))

# Q4 Milgrom 1999 inertia from Delta T = T(a) - T(0): Delta T / a -> a/(2H) for a << H, i.e. a0 = 2 c H, not theta c H.
x = sp.symbols('x', positive=True)
dT = sp.sqrt(x**2 + 1) - 1                                         # in units H/2pi, x = a/H
ok(f"Q4 Milgrom-1999 deep limit Delta T ~ {sp.series(dT, x, 0, 3).removeO()} -> a0/(cH) = 2 (factor {float(2 / (sp.sqrt(2) / (2 * sp.pi))):.1f} above theta); "
   "the temperature picture alone does not give theta",
   sp.simplify(sp.series(dT, x, 0, 3).removeO() - x**2 / 2) == 0)

# Q5 numbers (H0 = 67.4 km/s/Mpc, Omega_L = 0.685, H_inf = H0 sqrt(Omega_L)); observed a0 = 1.20 +- 0.24 (sys) e-10 m/s^2.
c = 2.998e8; H0 = 67.4e3 / 3.0857e22; Hinf = H0 * 0.685 ** 0.5
cand = {'cH0/2pi': c * H0 / (2 * 3.14159265), 'cHinf/2pi (T at a=0)': c * Hinf / (2 * 3.14159265),
        'theta cHinf (T at a=H)': c * Hinf * 2 ** 0.5 / (2 * 3.14159265), 'sqrt3/2pi cHinf (S^4 minimal)': c * Hinf * 3 ** 0.5 / (2 * 3.14159265),
        '2 cHinf (Milgrom 1999)': 2 * c * Hinf}
for k, v in cand.items():
    print(f"   {k:32s} a0 = {v * 1e10:6.2f} e-10   ({(v * 1e10 - 1.20) / 0.24:+5.1f} sigma)")
ok("Q5 within 1 sigma: T at a=H (theta), and also cH0/2pi; T at a=0 with H_inf, sqrt3/2pi are 1.2-1.4 sigma off (data barely separate them), 2cH is excluded",
   abs(cand['theta cHinf (T at a=H)'] * 1e10 - 1.2) < 0.24 and abs(cand['cH0/2pi'] * 1e10 - 1.2) < 0.24
   and abs(cand['cHinf/2pi (T at a=0)'] * 1e10 - 1.2) > 0.24 and abs(cand['sqrt3/2pi cHinf (S^4 minimal)'] * 1e10 - 1.2) > 0.24)
