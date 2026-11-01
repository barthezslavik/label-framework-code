# 78: making the rotor <-> dilation exchange of 35 (S6) concrete. Scaling group of collisionless gravity on the phase plane:
# lambda = s e^theta (positions), mu = s e^-theta (velocities), subgroup mu = lambda^p; Newtonian mass weight M ~ lambda mu^2. DU1-DU4.
# Structural; no data.  Run: ~/.venvs/jev/bin/python ga_scaling_duality_check.py
import sympy as sp

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

sp_, th, p = sp.symbols('s theta p', real=True)
lnl, lnm = sp_ + th, sp_ - th
# DU1 the exchange s <-> theta (log z -> K log z) is the velocity inversion mu -> 1/mu at fixed lambda, so p -> -p.
lnl2, lnm2 = lnl.subs({sp_: th, th: sp_}, simultaneous=True), lnm.subs({sp_: th, th: sp_}, simultaneous=True)
ok(f"DU1 exchange: ln lambda -> {sp.simplify(lnl2)}, ln mu -> {sp.simplify(lnm2)} (= -ln mu): velocity inversion at fixed positions, p -> -p",
   sp.simplify(lnl2 - lnl) == 0 and sp.simplify(lnm2 + lnm) == 0)

# DU2 what each subgroup keeps (Newtonian mass weight M ~ lambda mu^2): phase volume (lambda mu)^3 fixed <=> p = -1 (SDiff);
#     mass fixed <=> p = -1/2 (Kepler); acceleration mu^2/lambda = G Sigma fixed <=> p = 1/2 (k = 2 level set); density
#     M/lambda^3 = mu^2/lambda^2 fixed <=> p = 1 (k = 3 level set, rigid rotation).
lam = sp.symbols('lambda', positive=True); mu = lam**p
keep = {'phase volume': (lam * mu)**3, 'mass': lam * mu**2, 'acceleration / Sigma (k=2)': mu**2 / lam, 'density (k=3)': mu**2 / lam**2}
ps = {k: sp.solve(sp.Eq(sp.diff(sp.log(v), lam) * lam, 0), p)[0] for k, v in keep.items()}
ok(f"DU2 subgroup keeping each quantity: {ps}",
   ps == {'phase volume': -1, 'mass': sp.Rational(-1, 2), 'acceleration / Sigma (k=2)': sp.Rational(1, 2), 'density (k=3)': 1})

# DU3 the exchange pairs conservation laws of ONE system with level sets of a POPULATION:
#     mass conserved (p = -1/2)            <-> Sigma / acceleration fixed (p = +1/2, k = 2), population slope M ~ v^((1+2p)/p) = 4
#     phase volume conserved (p = -1, SDiff) <-> density fixed (p = +1, k = 3), population slope 3
#     p = 0 (deep-MOND scaling, v fixed) is self-dual.
slope = lambda q: sp.simplify((1 + 2 * sp.S(q)) / sp.S(q))
ok(f"DU3 dual pairs: mass <-> k=2 (slope {slope(sp.Rational(1, 2))}), phase volume <-> k=3 (slope {slope(1)}), p = 0 self-dual",
   -ps['mass'] == ps['acceleration / Sigma (k=2)'] and -ps['phase volume'] == ps['density (k=3)']
   and slope(sp.Rational(1, 2)) == 4 and slope(1) == 3)

# DU4 reading (hypothesis, not derived): if a population of systems lies on the dual of the orbit that conserves each system's own
#     invariant, then dissipative systems (baryons: phase-space SDiff broken by dissipation, mass conserved) form the k = 2 population,
#     M_b ~ v^4, and collisionless systems (dust halos: phase volume conserved) form the k = 3 population, M ~ v^3. The duality
#     principle itself is an assumption; the pairing (DU3) is algebra.
ok("DU4 recorded: duality principle 'population = dual of the conserved-invariant orbit' gives baryons -> slope 4, dust halos -> slope 3 (hypothesis)", True)

# DU5 why the dual: the log plane of the group, log z = s + K theta, carries the split-complex quadratic form s^2 - theta^2 (K^2 = +1).
#     For an invariant I = lambda^a mu^b kept by the subgroup p (a + b p = 0), ln I = (a + b) s + (a - b) theta; its gradient raised with
#     diag(1, -1) points along theta/s = (b - a)/(a + b), which is the subgroup -p. So the dual subgroup is the direction of steepest
#     change of the conserved invariant in the native metric (hyperbolic-orthogonal to the orbit).
a_, b_ = sp.symbols('a b', real=True)
grad = sp.Matrix([a_ + b_, -(a_ - b_)])                              # (d/ds, d/dtheta) of ln I, index raised with diag(1, -1)
ratio = sp.simplify((grad[1] / grad[0]).subs(a_, -b_ * p))            # theta/s of the gradient
ratio_of = lambda q: (1 - q) / (1 + q)                                # theta/s of the subgroup q
ok(f"DU5 Minkowski gradient of the invariant of subgroup p has theta/s = {ratio} = that of subgroup -p: dual = steepest change of the invariant",
   sp.simplify(ratio - ratio_of(-p)) == 0)

# DU6 the metric matters: with the Euclidean form the gradient of ln M (M ~ lambda mu^2) points to p = 2 (population slope 5/2), not 1/2.
#     The flat-curve subgroup p = 0 (theta/s = 1) is null, s^2 - theta^2 = 0: self-orthogonal, hence self-dual.
gE = sp.Matrix([3, -1])                                               # ln M = 3 s - theta, Euclidean gradient
pE = sp.solve(sp.Eq(ratio_of(p), gE[1] / gE[0]), p)[0]
ok(f"DU6 Euclidean gradient of ln M -> p = {pE} (slope {slope(pE)}); split-complex -> p = 1/2 (slope 4); p = 0 is null (1 - 1 = {1 - ratio_of(0)**2})",
   pE == 2 and 1 - ratio_of(0)**2 == 0)

# DU7 TEST of the DU4 principle against populations (literature numbers, verified 2026-10-21):
#     (a) collisionless-dominated galaxy clusters: M500 ~ T^alpha, alpha = 1.47-1.71 (Vikhlinin 2006/2009, Arnaud 2005), T ~ sigma^2 ->
#         M ~ sigma^(2 alpha) = 2.9-3.4: consistent with the k = 3 prediction (3).
#     (b) baryonic, low acceleration (<g_bar> < 0.6 a0; dwarfs and groups, Tian, Lelli et al. 2026, A&A 710, L39): M_bar ~ sigma^(4.19 +- 0.10):
#         consistent with the k = 2 prediction (4) at 1.9 sigma.
#     (c) baryonic, high acceleration (bright ellipticals, <g_bar> >~ 6 a0, same paper): no M_bar-sigma power law; they follow the Newtonian
#         virial fundamental plane with a radius term. DU4 predicts slope 4 for ALL dissipative (mass-conserving) systems, independent of
#         acceleration -> FALSIFIED for high-acceleration baryonic systems. The data select the k = 2 population by low acceleration,
#         not by dissipation.
alpha = (1.47, 1.71)
ok(f"DU7 clusters: M ~ sigma^{2*alpha[0]:.1f}-{2*alpha[1]:.1f} (k=3 -> 3, consistent); low-acc. baryonic 4.19 +- 0.10 (k=2 -> 4, 1.9 sigma); "
   "high-acc. ellipticals: no power law (fundamental plane) -> DU4 as stated is FALSIFIED: slope 4 is selected by low acceleration, not dissipation",
   2 * alpha[0] < 3 < 2 * alpha[1] + 0.1 and abs(4.19 - 4) / 0.10 < 2)

# DU8 (review 2026-10-22) the action on the sources matters: with the Vlasov-Poisson mass weight lambda mu^2, g and g_N both scale as
#     mu^2/lambda and C = g^2/g_N -> (mu^2/lambda) C, i.e. C/lambda at p = 0 (the deep-MOND curve is NOT invariant); with masses fixed,
#     g_N -> g_N/lambda^2, g -> (mu^2/lambda) g and C is invariant at p = 0; the law g^2 = a0 g_N needs mass weight mu^4.
l_, m_ = sp.symbols('l m', positive=True)
C_vp = sp.simplify((m_**2 / l_)**2 / (l_ * m_**2 / l_**2))
C_fix = sp.simplify((m_**2 / l_)**2 / (1 / l_**2))
w = sp.symbols('w', positive=True); wsol = sp.solve(sp.Eq((m_**2 / l_)**2, w / l_**2), w)[0]
ok(f"DU8 C scales as {C_vp} (VP weight; p=0: {C_vp.subs(m_, 1)}), as {C_fix} with fixed masses (p=0: {C_fix.subs(m_, 1)}); mass weight for g^2=a0 g_N: {wsol}",
   C_vp.subs(m_, 1) == 1 / l_ and C_fix.subs(m_, 1) == 1 and wsol == m_**4)
