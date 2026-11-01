# Which dualities of the list map solutions to solutions? (papers/restart/37-dualities.md). T1-T7.
# "True" = maps solutions of an equation to solutions of the same equation; "correspondence" = between two equations; "kinematic" = only a map of
# numbers or profiles.
# Run: uv run --with sympy --with numpy --with scipy python ga_duality_test_check.py
import numpy as np
import sympy as sp
from scipy.optimize import brentq
from ga_core import ok

t, x, y, z = sp.symbols('t x y z', real=True)
X = (x, y, z)
grad = lambda f: [sp.diff(f, v) for v in X]
div = lambda F: sum(sp.diff(F[i], X[i]) for i in range(3))
curl = lambda F: [sp.diff(F[2], y) - sp.diff(F[1], z), sp.diff(F[0], z) - sp.diff(F[2], x), sp.diff(F[1], x) - sp.diff(F[0], y)]

# T1 Hodge F -> F I in vacuum Maxwell (E -> -B... E' = B, B' = -E): a generic vacuum solution (two crossed plane waves) stays a solution.
k = sp.symbols('k', positive=True)
E = [0, sp.cos(k * (x - t)), 0]; B = [0, 0, sp.cos(k * (x - t))]                              # plane wave along x
E2 = [sp.cos(k * (z - t)), 0, 0]; B2 = [0, sp.cos(k * (z - t)), 0]                            # plane wave along z
E = [E[i] + E2[i] for i in range(3)]; B = [B[i] + B2[i] for i in range(3)]
def maxwell(E, B):
    eqs = [div(E), div(B)] + [sp.diff(B[i], t) + curl(E)[i] for i in range(3)] + [sp.diff(E[i], t) - curl(B)[i] for i in range(3)]
    return all(sp.simplify(q) == 0 for q in eqs)
Ed, Bd = B, [-c for c in E]
ok(f"T1 Maxwell vacuum: solution {maxwell(E, B)}, Hodge dual (E,B) -> (B,-E) solution {maxwell(Ed, Bd)}: TRUE duality", maxwell(E, B) and maxwell(Ed, Bd))

# T2 inversion. Laplace: Kelvin K u = u(x/|x|^2)/|x| maps harmonic -> harmonic (mass of a point source changes). Deep MOND: weight-0 inversion
#    maps L3-solutions to L3-solutions with mass conserved (32, B1). Full MOND (simple mu, point mass): the inverted radial field does not solve it.
r = sp.sqrt(x**2 + y**2 + z**2)
u = 1 / sp.sqrt((x - 1)**2 + y**2 + z**2)
Ku = u.subs({x: x / r**2, y: y / r**2, z: z / r**2}, simultaneous=True) / r
lapK = sp.N(sum(sp.diff(Ku, v, 2) for v in X).subs({x: 0.3, y: -0.7, z: 0.4}))
a0, GM = 1.0, 1.0
gfull = lambda rr: brentq(lambda g: g * g / (g + a0) - GM / rr**2, 1e-12, 1e6)            # simple-mu field of a point mass
rr = np.logspace(-1.5, 1.5, 7)
ginv = np.array([gfull(1 / q) / q**2 for q in rr])                                          # radial field of phi(x/|x|^2) (weight 0)
mass_eff = ginv**2 / (ginv + a0) * rr**2                                                   # would be constant if it solved full MOND
ok(f"T2 Kelvin: Laplacian of K u = {float(lapK):.1e} (harmonic: TRUE); full MOND after inversion: effective mass varies x{mass_eff.max() / mass_eff.min():.1f} "
   "over radius (NOT a symmetry); deep MOND: TRUE with mass conserved (32 B1)", abs(lapK) < 1e-10 and mass_eff.max() / mass_eff.min() > 2)

# T3 the MOND point-mass duality g(r) g(r_M^2/r) = a0^2: exact in deep MOND, broken by the full law.
rM = np.sqrt(GM / a0)
prod = np.array([gfull(q) * gfull(rM**2 / q) for q in rr])
ok(f"T3 full simple-mu: g(r) g(r_M^2/r)/a0^2 ranges {prod.min():.2f}..{prod.max():.2f} (1 only in the deep limit): deep-only duality",
   abs(prod.min() - 1) > 0.05 or abs(prod.max() - 1) > 0.05)

# T4 the simple-mu relation is self-dual under (baryonic <-> dark): with g_D = g - g_N, g_D/g_N = a0/g, and g -> a0^2/g swaps g_N and g_D
#    (up to the factor a0^2/g^2). The empirical RAR form is only close to this.
g, A = sp.symbols('g a0', positive=True)
gN = g**2 / (g + A); gD = g - gN
gp_ = A**2 / g
ok(f"T4 simple mu: g_D/g_N = {sp.simplify(gD / gN)}; at g' = a0^2/g: g_D'/g_N' = {sp.simplify((gD / gN).subs(g, gp_))} = g_N/g_D: baryonic <-> dark "
   "exchange is an exact involution of the relation (curve-level, not a field-equation symmetry)",
   sp.simplify(gD / gN - A / g) == 0 and sp.simplify((gD / gN).subs(g, gp_) * (gD / gN) - 1) == 0)
gg = np.logspace(-2, 2, 81)                                                                  # g in units of a0
def gN_rar(gv):                                                                              # invert g = gN/(1-exp(-sqrt(gN)))
    return brentq(lambda q: q / (1 - np.exp(-np.sqrt(q))) - gv, 1e-14, gv)
ratio = np.array([(q - gN_rar(q)) / gN_rar(q) for q in gg])
f_prod = ratio * np.interp(np.log(1 / gg), np.log(gg), ratio)
print(f"   empirical RAR: (g_D/g_N)(g) * (g_D/g_N)(a0^2/g) in {f_prod.min():.3f}..{f_prod.max():.3f} (exact duality would give 1)")

# T5 rotor <-> dilation swap p -> -p as a profile map: v ~ r^p -> r^-p. Poisson density rho ~ (1+2p') r^(2p'-2) must be >= 0: p' = -p needs p <= 1/2.
#    Kepler (p=-1/2) -> p=+1/2 (rho ~ 1/r, fine); solid body (p=1) -> p=-1 (rho < 0, unphysical).
pp = sp.symbols('p', real=True)
rho_sign = lambda q: 1 + 2 * q                                                              # sign of rho for M ~ r^(1+2q)
ok(f"T5 p -> -p: Kepler -> rho sign {rho_sign(sp.Rational(1, 2))} (ok), flat -> flat, solid body -> rho sign {rho_sign(-1)} (negative): "
   "a profile map that changes the source and fails for p > 1/2: KINEMATIC", rho_sign(-1) < 0 and rho_sign(sp.Rational(1, 2)) > 0)

# T6 grade n <-> 4-n (superfluid phase phi <-> 3-form *d phi): a solution of the wave equation gives a conserved current (correspondence), but the
#    flow u ~ grad phi is not geodesic (dust needs u.grad u = 0): not a map onto dust solutions.
ph = t + sp.Rational(1, 10) * sp.sin(x) * sp.cos(t)                                         # exact wave solution
box = sp.simplify(-sp.diff(ph, t, 2) + sum(sp.diff(ph, v, 2) for v in X))
U = [sp.diff(ph, t)] + grad(ph)                                                             # u^mu ~ (phi_t, grad phi) (up to sign conventions)
norm2 = U[0]**2 - sum(c**2 for c in U[1:])
uu = [c / sp.sqrt(norm2) for c in U]
acc_x = sp.simplify((uu[0] * sp.diff(uu[1], t) + sum(uu[i + 1] * sp.diff(uu[1], X[i]) for i in range(3))))
ok(f"T6 box phi = {box} (wave solution); current conserved by construction; geodesic acceleration of u = grad phi: {sp.N(acc_x.subs({t: 0.3, x: 0.4, y: 0, z: 0})):.3f} "
   "!= 0: CORRESPONDENCE (superfluid -> conserved current), not a map onto dust", box == 0 and abs(sp.N(acc_x.subs({t: 0.3, x: 0.4, y: 0, z: 0}))) > 1e-3)

# T7 Hodge a <-> H^2/a (observer <-> horizon): maps the accelerated-observer family onto itself with T' = (H/a) T (31 D3); no isometry realises it
#    (B -> B I5 is not a versor action, 32), so it is a map of the family, not a symmetry of the field equations: KINEMATIC.
a_, H_ = sp.symbols('a H', positive=True)
T = lambda q: sp.sqrt(q**2 + H_**2) / (2 * sp.pi)
ok(f"T7 T(H^2/a) / T(a) = {sp.simplify(T(H_**2 / a_) / T(a_))} != 1: the dual observer is hotter/colder -- no symmetry of the detector problem",
   sp.simplify(T(H_**2 / a_) / T(a_) - H_ / a_) == 0)
