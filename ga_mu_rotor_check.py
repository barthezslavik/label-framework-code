# The Fermi-type mu from the dilation rotor (papers/restart/33-mu-transition.md, section "GA origin"). F1-F5.
# D = n_0 ^ n_inf (the dilation of 3-space = de Sitter static time, ga_theta_conformal_check C3), D^2 = 1, idempotents P+- = (1 +- D)/2.
# Under Milgrom's scaling (r, t) -> s (r, t) an acceleration goes g -> g/s, so the acceleration ratio x = g/a0 is a dilation s = x.
# Run: uv run --with sympy python ga_mu_rotor_check.py
import sympy as sp
from ga_core import *

C5 = {0: 1, 1: 1, 2: 1, 3: 1, 4: -1}
ninf = {(3,): 1, (4,): 1}; n0 = {(4,): sp.Rational(1, 2), (3,): -sp.Rational(1, 2)}
simp = lambda M: {k: w for k, w in ((k, sp.simplify(v)) for k, v in M.items()) if w != 0}
rev = lambda M: {k: (v if len(k) % 4 in (0, 1) else -v) for k, v in M.items()}
D = wedge(n0, ninf, C5)
one = {(): 1}
Pp, Pm = scal(sp.Rational(1, 2), add(one, D)), scal(sp.Rational(1, 2), add(one, scal(-1, D)))
lam, x = sp.symbols('lambda x', positive=True)

# F1 idempotents of the dilation
ok(f"F1 D^2 = {gp(D, D, C5)}; P+^2 = P+, P-^2 = P-, P+ P- = 0, P+ + P- = 1, <P+->_0 = 1/2",
   gp(D, D, C5) == one and simp(add(gp(Pp, Pp, C5), scal(-1, Pp))) == {} and simp(add(gp(Pm, Pm, C5), scal(-1, Pm))) == {}
   and simp(gp(Pp, Pm, C5)) == {} and simp(add(Pp, Pm)) == one and Pp[()] == sp.Rational(1, 2))

# F2 the dilation rotor splits into the idempotents with half-rapidity weights: R = exp(-D lam/2) = e^{-lam/2} P+ + e^{lam/2} P-,
#    and it scales points by e^{-lam} (ga_theta_conformal_check C3); a vector's null parts scale by e^{-+lam} (full rapidity).
R = add({(): sp.cosh(lam / 2)}, scal(-sp.sinh(lam / 2), D))
split = simp(add(scal(sp.exp(-lam / 2), Pp), scal(sp.exp(lam / 2), Pm), scal(-1, R)))
V = add(n0, ninf)
RV = simp(gp(gp(R, V, C5), rev(R), C5))
c0 = sp.simplify(-sp.Rational(1, 1) * (RV.get((3,), 0) - RV.get((4,), 0)))          # coefficient of n_0  (n_0 = (e- - e+)/2)
cinf = sp.simplify((RV.get((3,), 0) + RV.get((4,), 0)) / 2)                           # coefficient of n_inf
ok(f"F2 R - (e^-lam/2 P+ + e^lam/2 P-) = {split}; R (n_0 + n_inf) R~ = {c0} n_0 + {cinf} n_inf (weights e^+-lam: full rapidity)",
   split == {} and sp.simplify(c0 * cinf - 1) == 0 and (sp.simplify(c0 / cinf - sp.exp(2 * lam)) == 0 or sp.simplify(cinf / c0 - sp.exp(2 * lam)) == 0))

# F3 the two partitions. Rotor (spinor) fraction mu_R = <P- R>_0/<R>_0 with x = e^lam gives the SIMPLE mu = x/(1+x);
#    the vector fraction gives x^2/(1+x^2) = (standard mu)^2, the de Sitter/thermal one (5D Pythagoras).
muR = sp.simplify((gp(Pm, R, C5)[()] / R[()]).rewrite(sp.exp).subs(lam, sp.log(x)))
muV = sp.simplify((sp.exp(lam) / (sp.exp(lam) + sp.exp(-lam))).subs(lam, sp.log(x)))
ok(f"F3 rotor fraction <P- R>_0/<R>_0 = {muR} (simple mu); vector fraction = {muV} = (x/sqrt(1+x^2))^2 (standard mu squared)",
   sp.simplify(muR - x / (1 + x)) == 0 and sp.simplify(muV - (x / sp.sqrt(1 + x**2))**2) == 0)

# F4 the duality a <-> a0^2/a is reversion R -> R~ (lam -> -lam, P+ <-> P-); P+ + P- = 1 makes the partition linear: mu(x) + mu(1/x) = 1.
Rrev = rev(R)
muRrev = sp.simplify((gp(Pm, Rrev, C5)[()] / Rrev[()]).rewrite(sp.exp).subs(lam, sp.log(x)))
ok(f"F4 reversion R~ gives mu = {muRrev} = mu(1/x); mu(x) + mu(1/x) = {sp.simplify(muR + muRrev)}",
   sp.simplify(muRrev - 1 / (1 + x)) == 0 and sp.simplify(muR + muRrev - 1) == 0)

# F5 analytic structure: as functions of the rapidity lam, the rotor partition 1/(1+e^-lam) has poles at lam = i pi (2k+1) (antiperiodic over
#    2 pi i: the spinor sign R(2 pi) = -1, Fermi), the vector one 1/(1+e^-2lam) at lam = i pi/2 (2k+1). The Euclidean rotor at angle 2 pi is -1.
th = sp.symbols('theta', real=True)
DE = {(0, 1): 1}                                                      # a Euclidean rotation plane, DE^2 = -1
RE = add({(): sp.cos(th / 2)}, scal(-sp.sin(th / 2), DE))
z = sp.symbols('z')
poleR = sp.solve(sp.Eq(1 + sp.exp(-z), 0), z); poleV = sp.solve(sp.Eq(1 + sp.exp(-2 * z), 0), z)
ok(f"F5 Euclidean rotor at 2 pi: {simp({k: v.subs(th, 2 * sp.pi) for k, v in RE.items()})} (spinor sign); poles: rotor partition {poleR}, vector {poleV}",
   simp({k: v.subs(th, 2 * sp.pi) for k, v in RE.items()}) == {(): -1} and (sp.I * sp.pi in poleR or -sp.I * sp.pi in poleR) and sp.I * sp.pi / 2 in poleV)
# (Data: rar_mu_thermal_check U2 gives n = 0.98 +- 0.11, i.e. the rotor partition n = 1, against the vector/thermal n = 2.)
# (Scale: lam = 0 is the self-dual point x = 1 (mu = 1/2); the rotor fixes the shape, not a0 -- a0 stays the reference of the dilation.)
