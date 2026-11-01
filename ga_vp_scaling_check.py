# Scaling symmetries of collisionless self-gravity and where "a0" and slope 4 come from (papers/restart/34-sdiff-rotation.md, "Symmetries"). V1-V4.
# Vlasov-Poisson: f_t + v.grad_x f - grad phi . grad_v f = 0, lap phi = 4 pi G int f d^3v. Two-parameter scaling (lambda, mu):
# x -> lambda x, v -> mu v, t -> (lambda/mu) t, phi -> mu^2 phi, f -> f /(lambda^2 mu) (so M -> lambda mu^2 M).
# Run: uv run --with sympy python ga_vp_scaling_check.py
import sympy as sp
from ga_core import ok

lam, mu, G, M, r, v, eta, s, p = sp.symbols('lambda mu G M r v eta0 sigma p', positive=True)

# V1 invariance of Vlasov-Poisson (1D-in-each-variable check suffices for the powers): each term scales by the same factor.
fs = 1 / (lam**2 * mu)                       # f scaling
t_s, x_s, v_s, phi_s = lam / mu, lam, mu, mu**2
terms = {'f_t': fs / t_s, 'v f_x': v_s * fs / x_s, 'phi_x f_v': (phi_s / x_s) * fs / v_s}
poisson = {'lap phi': phi_s / x_s**2, '4 pi G int f d^3v': fs * v_s**3}
ok(f"V1 Vlasov terms scale as {set(sp.simplify(q) for q in terms.values())}, Poisson sides as {set(sp.simplify(q) for q in poisson.values())}: "
   "a symmetry for every (lambda, mu); mass M -> lambda mu^2 M, phase density eta0 -> eta0/(lambda^2 mu)",
   len(set(sp.simplify(q) for q in terms.values())) == 1 and len(set(sp.simplify(q) for q in poisson.values())) == 1)

# V2 accelerations scale as mu^2/lambda. Two one-parameter subgroups matter:
#    (A) mu = 1 (Milgrom's (r,t) -> lambda (r,t)): v fixed, a -> a/lambda  -- symmetry of the deep-MOND law, not of a fixed knee;
#    (B) mu = sqrt(lambda): a fixed (surface density Sigma = M/r^2 fixed)  -- the orbit on which a knee acceleration is constant.
acc = mu**2 / lam
ok(f"V2 a -> {acc} a; subgroup A (mu=1): a -> {acc.subs(mu, 1)} a, v fixed; subgroup B (mu=sqrt(lambda)): a -> {sp.simplify(acc.subs(mu, sp.sqrt(lam)))} a, "
   f"Sigma -> {sp.simplify((lam * mu**2 / lam**2).subs(mu, sp.sqrt(lam)))} Sigma",
   sp.simplify(acc.subs(mu, sp.sqrt(lam)) - 1) == 0 and sp.simplify((lam * mu**2 / lam**2).subs(mu, sp.sqrt(lam)) - 1) == 0)

# V3 along orbit B: M ~ lambda^2 and v ~ lambda^(1/2), so M ~ v^4 (Tully-Fisher slope 4), and eta0 ~ lambda^(-5/2) ~ sigma^-5 -- exactly the
#    relation that 34/R6 needs for a constant knee in Lynden-Bell spheres. "a0" is the invariant of orbit B, slope 4 is its exponent.
Mb, vb, eb = lam * mu**2, mu, 1 / (lam**2 * mu)
slope = sp.simplify(sp.log(Mb.subs(mu, sp.sqrt(lam))) / sp.log(vb.subs(mu, sp.sqrt(lam))))
eexp = sp.simplify(sp.log(eb.subs(mu, sp.sqrt(lam))) / sp.log(vb.subs(mu, sp.sqrt(lam))))
ok(f"V3 orbit B: d ln M / d ln v = {slope} (BTFR slope 4), d ln eta0 / d ln sigma = {eexp} (eta0 ~ sigma^-5, as in R6)",
   slope == 4 and eexp == -5)

# V4 a general orbit mu = lambda^p: slope s = (1 + 2p)/p and knee a ~ lambda^(2p-1). The measured BTFR slope 3.62 (18) gives p = 0.617
#    and a knee drifting as a ~ M^0.105 along the population: a weak, testable trend instead of a constant.
s_of_p = (1 + 2 * p) / p
p362 = sp.solve(sp.Eq(s_of_p, sp.Rational(362, 100)), p)[0]
a_exp_M = sp.simplify((2 * p - 1) / (1 + 2 * p))
ok(f"V4 slope(p) = {s_of_p}; slope 3.62 -> p = {float(p362):.3f}; knee a ~ M^{float(a_exp_M.subs(p, p362)):.3f} along the population "
   "(slope 4 <-> p = 1/2 <-> exactly constant a)",
   abs(float(p362) - 0.6173) < 1e-3 and sp.simplify(s_of_p.subs(p, sp.Rational(1, 2)) - 4) == 0 and a_exp_M.subs(p, sp.Rational(1, 2)) == 0)
