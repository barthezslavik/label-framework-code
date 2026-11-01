# Can the deep-MOND law g = sqrt(a0 g_N) with a0 = theta c H come out of T(a) = sqrt(a^2+H^2)/2pi and its duality invariant T/sqrt(aH)?
# (papers/restart/31-theta-sphere.md, section "MOND from the invariant?"). M1-M5.
# A "thermal rule" fixes the true acceleration g from the Newtonian g_N through temperatures; x = g/H, y = g_N/H.
# Run: uv run --with sympy python ga_theta_mond_check.py
import sympy as sp
from ga_core import ok

x, y, H, lam = sp.symbols('x y H lambda', positive=True)
T = lambda a: sp.sqrt(a**2 + H**2) / (2 * sp.pi)
TU = lambda a: a / (2 * sp.pi)
TGH = H / (2 * sp.pi)
theta = sp.sqrt(2) / (2 * sp.pi)
c, Hinf = 2.998e8, 67.4e3 / 3.0857e22 * 0.685 ** 0.5
sig = lambda k: (float(k) * c * Hinf * 1e10 - 1.20) / 0.24

def deep(rule):
    """rule(g) = g_N; return a0/H from the deep limit g_N ~ g^2/a0, and check the Newtonian limit g_N ~ g."""
    gN = sp.simplify(rule(x * H) / H)
    lo = sp.limit(gN / x**2, x, 0)                 # g_N/H ~ (x^2) / (a0/H)
    hi = sp.limit(gN / x, x, sp.oo)
    return (1 / lo if lo not in (0, sp.oo) else None), hi

# M1 rule g_N = g * T_U(g)/T(g): mu = sin(chi) = x/sqrt(1+x^2) is EXACTLY the standard MOND interpolating function, but a0 = H.
k1, hi1 = deep(lambda g: g * TU(g) / T(g))
ok(f"M1 g_N = g T_U/T (mu = x/sqrt(1+x^2), the 'standard' MOND function): Newton limit g_N/g -> {hi1}; deep a0 = {k1} c H_inf "
   f"= {float(k1) * c * Hinf * 1e10:.2f}e-10 ({sig(k1):+.0f} sigma)", hi1 == 1 and k1 == 1 and sig(k1) > 5)

# M2 Milgrom 1999 excess: T(g) - T_GH = T_U(g_N)  ->  a0 = 2 H.
k2, hi2 = deep(lambda g: 2 * sp.pi * (T(g) - TGH))
ok(f"M2 T(g) - T_GH = T_U(g_N): Newton limit {hi2}; deep a0 = {k2} c H_inf ({sig(k2):+.0f} sigma)", hi2 == 1 and k2 == 2 and sig(k2) > 5)

# M3 rules from the duality invariant I(g) = T/sqrt(gH) (minimum theta at g = H): g_N = g (theta/I)^2 gives no Newton limit;
#    g_N = g (1 - theta/I) ... the invariant enters only through theta/I = sqrt(2x/(1+x^2)), which is pi-free.
I = lambda a: T(a) / sp.sqrt(a * H)
r3 = sp.simplify(theta / I(x * H))
k3, hi3 = deep(lambda g: g * (theta / I(g))**2)
ok(f"M3 theta/I = {r3} (pi cancels); rule g_N = g (theta/I)^2: Newton limit g_N/g -> {hi3} (not 1, fails); deep a0 = {k3} H",
   sp.simplify(r3 - sp.sqrt(2 * x / (1 + x**2))) == 0 and hi3 == 0)

# M4 no-go: every rule that compares temperatures with temperatures (or accelerations with accelerations) is invariant under the
#    rescaling T -> lambda T of the 2 pi convention, so its a0/H is a pi-free number (1, 2, 1/2, ...). theta = sqrt2/(2 pi) is not:
#    a0 = theta c H needs an acceleration set equal to a bare temperature, the step that also gives Milgrom's cH/2pi.
rule_ratio = lambda g, s: g * (s * TU(g)) / (s * T(g))                 # same rule with all temperatures rescaled by s
inv = sp.simplify(rule_ratio(x * H, lam) - rule_ratio(x * H, 1)) == 0
knee_T = sp.sqrt(2) * H / (2 * sp.pi)                                 # T at the self-dual point
ok(f"M4 temperature-ratio rules are invariant under T -> lambda T: {inv}; their a0/H is pi-free; a0 = T(self-dual) = {knee_T} "
   f"scales as lambda under the same rescaling -> it is a unit identification, not a consequence of T(a)",
   inv and sp.simplify((lam * knee_T) / knee_T - lam) == 0 and not sp.simplify(theta).is_algebraic)

# M5 what the data need from a ratio rule: a0/(c H_inf) = 1.20e-10 / (c H_inf) = 0.221 +- 0.044; theta = 0.2251 sits in it, the pi-free
#    candidates 1, 2, 1/2, 1/sqrt2, 1/4 do not (1/4 = 0.25 is within 1 sigma, 1/(2 sqrt2) = 0.354 is not).
need = 1.20e-10 / (c * Hinf); s_need = 0.24e-10 / (c * Hinf)
cands = {'1': 1, '2': 2, '1/2': 0.5, '1/sqrt2': 2**-0.5, '1/4': 0.25, 'theta': float(theta)}
print("   needed a0/(c H_inf) = %.3f +- %.3f;  " % (need, s_need) + ", ".join(f"{k}: {(v - need) / s_need:+.1f} sigma" for k, v in cands.items()))
ok("M5 among simple pi-free factors only 1/4 is within 1 sigma (theta too); 1, 2, 1/2 are excluded: data alone do not demand a pi",
   abs(cands['1/4'] - need) < s_need and abs(cands['theta'] - need) < s_need and all(abs(cands[k] - need) > 5 * s_need for k in ('1', '2')))
