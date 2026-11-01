# Time as stretching instead of shifting (papers/restart/41-time-as-dilation.md). T1-T7.
# Claim: the three possible "times" are the three sorts of units of sl(2,R) = bivectors of Cl(2,1): parabolic (square 0: ordinary time
# translation), hyperbolic (square +1: dilation / boost / modular flow -- "time as stretching"), elliptic (square -1: rotation, periodic
# Euclidean or global time). Checked in mechanics, conformal QM, Rindler, the log map, and cosmology.
# Run: uv run --with sympy python ga_time_dilation_check.py
import sympy as sp
from ga_core import *

# T1 sl(2,R) as bivectors of Cl(2,1) (e1^2 = e2^2 = 1, e3^2 = -1): elliptic e1e2 (square -1), hyperbolic e1e3, e2e3 (square +1), parabolic
#    e1e2 + e2e3 (square 0). They close under the commutator (a 3-dim Lie algebra).
C21 = {0: 1, 1: 1, 2: -1}
E12, E13, E23 = {(0, 1): 1}, {(0, 2): 1}, {(1, 2): 1}
Npar = add(E12, E23)
sq = lambda A: gp(A, A, C21)
comm = lambda A, B: add(gp(A, B, C21), scal(-1, gp(B, A, C21)))
ok(f"T1 squares: e1e2 {sq(E12)}, e1e3 {sq(E13)}, e2e3 {sq(E23)}, e1e2+e2e3 {sq(Npar)}; [e1e2,e1e3] = {comm(E12, E13)}, [e1e3,e2e3] = {comm(E13, E23)}",
   sq(E12) == {(): -1} and sq(E13) == {(): 1} and sq(E23) == {(): 1} and sq(Npar) == {}
   and set(comm(E12, E13)) <= {(1, 2)} and set(comm(E13, E23)) <= {(0, 1)})

# T2 conformal mechanics (de Alfaro-Fubini-Furlan): H = p^2/2 + g/(2x^2), D = x p/2, K = x^2/2 form sl(2,R) under Poisson brackets:
#    {D,H} = H, {D,K} = -K, {H,K} = -2D. A generator aH + bD + cK has discriminant b^2 - 4ac: H -> 0 (parabolic, ordinary time),
#    D -> 1 (hyperbolic, time as stretching), H + K -> -4 (elliptic, compact time with discrete spectrum).
x, p, g = sp.symbols('x p g', positive=True)
PB = lambda A, B: sp.simplify(sp.diff(A, x) * sp.diff(B, p) - sp.diff(A, p) * sp.diff(B, x))
H = p**2 / 2 + g / (2 * x**2); D = x * p / 2; K = x**2 / 2
disc = lambda a, b, c: b**2 - 4 * a * c
ok(f"T2 {{D,H}} - H = {sp.simplify(PB(D, H) - H)}, {{D,K}} + K = {sp.simplify(PB(D, K) + K)}, {{H,K}} + 2D = {sp.simplify(PB(H, K) + 2 * D)}; "
   f"discriminants H {disc(1, 0, 0)}, D {disc(0, 1, 0)}, H+K {disc(1, 0, 1)}",
   sp.simplify(PB(D, H) - H) == 0 and sp.simplify(PB(D, K) + K) == 0 and sp.simplify(PB(H, K) + 2 * D) == 0
   and disc(1, 0, 0) == 0 and disc(0, 1, 0) > 0 and disc(1, 0, 1) < 0)

# T3 what each kind of time does to states (2x2 flows): parabolic = shift (one fixed direction), hyperbolic = stretch with two fixed null
#    directions (growing / decaying halves P+-: an arrow away from a 'Janus' point), elliptic = rotation, periodic (-> temperature / quantization).
s = sp.symbols('s', real=True)
Mp, Mh, Me = sp.Matrix([[0, 1], [0, 0]]), sp.Matrix([[1, 0], [0, -1]]) / 2, sp.Matrix([[0, 1], [-1, 0]])
Ep, Eh, Ee = (s * Mp).exp(), (s * Mh).exp(), (s * Me).exp()
ok(f"T3 parabolic exp = {Ep.tolist()} (shift); hyperbolic exp = {Eh.tolist()} (two eigen-directions, e^+-s/2); elliptic at s = 2pi: {sp.simplify(Ee.subs(s, 2 * sp.pi)).tolist()} (periodic)",
   Ep == sp.Matrix([[1, s], [0, 1]]) and Eh == sp.diag(sp.exp(s / 2), sp.exp(-s / 2)) and sp.simplify(Ee.subs(s, 2 * sp.pi)) == sp.eye(2))

# T4 mechanics: the dilatational momentum D = x.p is a clock exactly when the potential is scale-invariant (1/x^2): dD/dt = {D,H} = 2H = const,
#    so D grows linearly (Barbour-Koslowski-Mercati's arrow). For Newton (1/x) dD/dt = 2T + V = 2E - V (Lagrange-Jacobi): bound systems
#    oscillate (virial <dD/dt> = 0), only E >= 0 systems get a monotone clock.
Dfull = x * p
Hn = p**2 / 2 - g / x
ok(f"T4 1/x^2: dD/dt = {sp.simplify(PB(Dfull, H))} = 2H; Newton: dD/dt - (2E - V) = {sp.simplify(PB(Dfull, Hn) - (2 * Hn + g / x))}",
   sp.simplify(PB(Dfull, H) - 2 * H) == 0 and sp.simplify(PB(Dfull, Hn) - (2 * Hn + g / x)) == 0)

# T5 the log map r = e^tau makes stretching a shift: flat dr^2 + r^2 dphi^2 = e^{2 tau} (dtau^2 + dphi^2) (cylinder up to a conformal factor),
#    and r -> lam r is tau -> tau + ln lam (radial quantization: the dilation is the cylinder's Hamiltonian).
tau, phi, lam = sp.symbols('tau phi lambda', positive=True)
r = sp.exp(tau)
metric = sp.simplify(sp.diff(r, tau)**2)                                   # coefficient of dtau^2 ; r^2 is the dphi^2 coefficient
ok(f"T5 g_tautau = {metric}, g_phiphi = {sp.simplify(r**2)}: conformal to the cylinder; r -> lam r is tau -> tau + {sp.simplify(sp.log(lam * r) - tau)}",
   sp.simplify(metric - r**2) == 0 and sp.simplify(sp.log(lam * r) - tau - sp.log(lam)) == 0)

# T6 Rindler: the boost is the squeeze (u, v) -> (e^s u, e^-s v) of light-cone coordinates (hyperbolic, 35 S1); continued to s = i theta it is a
#    rotation with period 2 pi in Euclidean time -> the Unruh temperature a/2pi (time as stretching carries a temperature).
u, v, th, a_ = sp.symbols('u v theta a', positive=True)
keep = sp.simplify((sp.exp(s) * u) * (sp.exp(-s) * v) - u * v)
Erot = (sp.I * th * Mh * 2).exp()                                           # e^{i theta sigma_3}: Euclidean continuation of the squeeze
ok(f"T6 squeeze keeps u v: {keep == 0}; its Euclidean continuation is periodic: back to 1 at theta = 2 pi: "
   f"{sp.simplify(Erot.subs(th, 2 * sp.pi)) == sp.eye(2)} -> temperature a/(2 pi) per unit proper acceleration",
   keep == 0 and sp.simplify(Erot.subs(th, 2 * sp.pi)) == sp.eye(2))

# T7 cosmology: matter era a ~ t^(2/3) is invariant under stretching of time t -> lam t (a -> lam^(2/3) a: homothety; H t = 2/3 constant in
#    log time); de Sitter a ~ e^{Ht} is invariant under shifting of time, which stretches space (= the dilator, ga_cosmic_dilator_check M2).
#    LCDM flows from a time-stretching symmetry (matter) to a time-shift = space-stretching symmetry (Lambda).
t, Hc, c0 = sp.symbols('t H c0', positive=True)
a_m = t**sp.Rational(2, 3); a_L = sp.exp(Hc * t)
ok(f"T7 matter: a(lam t)/a(t) = {sp.simplify(a_m.subs(t, lam * t) / a_m)} (t-independent), H t = {sp.simplify(sp.diff(a_m, t) / a_m * t)}; "
   f"de Sitter: a(t + c)/a(t) = {sp.simplify(a_L.subs(t, t + c0) / a_L)} (t-independent)",
   sp.simplify(sp.diff(a_m.subs(t, lam * t) / a_m, t)) == 0 and sp.simplify(sp.diff(a_L.subs(t, t + c0) / a_L, t)) == 0)
