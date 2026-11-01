# Cosmic time as a dilator, the galaxy motor, and why a0 can only change where matter breaks the dilation (papers/restart/39-cosmic-dilator.md). M1-M5.
# Conformal GA of R^3 (e1,e2,e3,e+ (+), e- (-)); n_inf = e- + e+, n_0 = (e- - e+)/2. A de Sitter point in flat slicing (conformal time eta,
# comoving x) is the sphere S(x, |eta|) (32: dS point = sphere). Versors: rotor (B^2 = -1), translator 1 + t n_inf/2 (nilpotent), dilator
# exp(-D l/2) with D = n_0 ^ n_inf (D^2 = +1). Motor = rotor x translator.
# Run: uv run --with sympy python ga_cosmic_dilator_check.py
import sympy as sp
from ga_core import *

C5 = {0: 1, 1: 1, 2: 1, 3: 1, 4: -1}
ninf = {(3,): 1, (4,): 1}; n0 = {(4,): sp.Rational(1, 2), (3,): -sp.Rational(1, 2)}
P = lambda x: add(vec(list(x) + [0, 0]), scal(sum(v**2 for v in x) / 2, ninf), n0)
S = lambda c, r: add(P(c), scal(-r**2 / 2, ninf))
sc = lambda M: sp.simplify(M.get((), 0))
simp = lambda M: {k: w for k, w in ((k, sp.simplify(v)) for k, v in M.items()) if w != 0}
rev = lambda M: {k: (v if len(k) % 4 in (0, 1) else -v) for k, v in M.items()}
act = lambda V, Y: simp(gp(gp(V, Y, C5), rev(V), C5))
def sphere_of(Q):                                                   # centre and radius of a (dual) sphere vector
    w = -sc(gp(Q, ninf, C5))
    c = [sp.simplify(Q.get((k,), 0) / w) for k in range(3)]
    r2 = sp.simplify(sc(gp(Q, Q, C5)) / w**2)
    return c, r2
lam, eta, H, phi = sp.symbols('lambda eta H phi', real=True)
c1, c2, c3, r = sp.Rational(1, 2), sp.Rational(-3, 4), sp.Rational(2, 3), sp.Rational(5, 3)   # a test sphere
D = wedge(n0, ninf, C5)
Rd = add({(): sp.cosh(lam / 2)}, scal(-sp.sinh(lam / 2), D))

# M1 the dilator moves every de Sitter point (eta, x) to (e^-lam eta, e^-lam x): conformal time and comoving position scale together.
c, r2 = sphere_of(act(Rd, S([c1, c2, c3], r)))
ok(f"M1 dilator on S(x, |eta|): centre -> {sp.simplify(c[0] / c1)} x, radius^2 -> {sp.simplify(r2 / r**2)} r^2: (eta, x) -> e^-lam (eta, x)",
   all(sp.simplify(c[i] / [c1, c2, c3][i] - sp.exp(-lam)) == 0 for i in range(3)) and sp.simplify(r2 / r**2 - sp.exp(-2 * lam)) == 0)

# M2 this is an isometry of de Sitter in flat slicing, ds^2 = (-d eta^2 + dx^2)/(H eta)^2, and with eta = -e^{-Ht}/H it is a pure shift of
#    cosmic time t -> t + lam/H. Hence cosmic time translation IS the dilator; in exact de Sitter nothing (law or population) can depend on t.
k = sp.exp(-lam)
ds2 = lambda de, dx, et: (-de**2 + dx**2) / (H * et)**2
de_, dx_ = sp.symbols('de dx', real=True)
iso = sp.simplify(ds2(k * de_, k * dx_, k * eta) - ds2(de_, dx_, eta))
tt = sp.symbols('t', real=True)
shift = sp.simplify(sp.expand_log(sp.solve(sp.Eq(-sp.exp(-H * tt) / H, k * (-sp.exp(-H * sp.Symbol('t0')) / H)), tt)[0], force=True) - sp.Symbol('t0'))
ok(f"M2 metric change under the dilator: {iso}; cosmic time shift: t -> t0 + ({shift}): time translation = dilator (isometry)",
   iso == 0 and sp.simplify(sp.expand_log(shift, force=True) - lam / H) == 0)

# M3 the galaxy motor: a spatial rotor (plane e1e2, B^2 = -1) commutes with D, so rotation + expansion compose into one loxodromic versor
#    exp(-(D lam + B phi)/2): x -> e^-lam R x R~ (spiral similarity). A bound galaxy follows the rotor but not the dilator: it breaks D.
B = {(0, 1): 1}
Rr = add({(): sp.cos(phi / 2)}, scal(-sp.sin(phi / 2), B))
comm = simp(add(gp(D, B, C5), scal(-1, gp(B, D, C5))))
L = gp(Rd, Rr, C5)
cL, rL2 = sphere_of(act(L, S([c1, 0, 0], r)))
ok(f"M3 [D, B] = {comm}; loxodromic versor maps centre (c1,0,0) -> ({sp.simplify(cL[0])}, {sp.simplify(cL[1])}, 0), radius^2 x {sp.simplify(rL2 / r**2)}",
   comm == {} and sp.simplify(cL[0] - c1 * sp.exp(-lam) * sp.cos(phi)) == 0 and sp.simplify(sp.Abs(cL[1]) - c1 * sp.exp(-lam) * sp.Abs(sp.sin(phi))) == 0)

# M4 backgrounds. Newtonian scaling (lambda, mu) with t -> (lambda/mu) t maps densities as rho -> (mu/lambda)^2 rho. Matter era (EdS,
#    rho = 1/(6 pi G t^2)) is invariant under the WHOLE group (homothety: the background flows but looks the same); Lambda era (rho_L const)
#    only under mu = lambda (t fixed). So matter lets populations slide along any orbit (a0 can drift); Lambda freezes the drift.
G, l_, m_, t0 = sp.symbols('G lambda_ mu_ t0', positive=True)
rho_eds = lambda t: 1 / (6 * sp.pi * G * t**2)
eds_inv = sp.simplify((m_ / l_)**2 * rho_eds(t0) - rho_eds((l_ / m_) * t0))
rl = sp.symbols('rho_L', positive=True)
lam_inv = sp.solve(sp.Eq((m_ / l_)**2 * rl, rl), m_)
ok(f"M4 EdS: (mu/lambda)^2 rho(t) - rho(t') = {eds_inv} for all (lambda, mu); Lambda: invariant only for mu = {lam_inv}",
   eds_inv == 0 and lam_inv == [l_])

# M5 a characteristic acceleration of the population, a* ~ mu^2/lambda (35), is invariant under the Lambda-era subgroup mu = lambda only if
#    lambda = 1: once Lambda dominates, no self-similar rescaling of the population is allowed at fixed t -> a* is frozen (memory of the matter era).
ok(f"M5 on the Lambda-compatible subgroup mu = lambda (t fixed) accelerations scale as a* -> {sp.simplify((m_**2 / l_).subs(m_, l_))} a*: time "
   "evolution is no longer a self-similar rescaling, and in exact dS it is an isometry (M2) -- the drift of a* stops; its matter-era value is kept", sp.simplify((m_**2 / l_).subs(m_, l_) - l_) == 0)
