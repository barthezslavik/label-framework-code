# 112: can two labels be one? Hypothesis: the spacing l of the label elements (axiom D, doc 91) is the cutoff of induced gravity (doc 89).
# Induced gravity (Sakharov; one loop, N effective fields with cutoff 1/l, equation borrowed): 1/(16 pi G) = N/(96 pi^2 l^2), i.e.
# (l_P/l)^2 = 6 pi/N. Discrete Lambda (DL3): fluctuation amplitude A = 8 pi alpha (l_P/l)^2 with alpha = O(1) from the uncertainty bound.
# Then A = 48 pi^2 alpha/N. Constraints: A <= 0.015 (BBN with a Lambda label, EV2); the pure discrete Lambda would need A ~ 0.73/xi0 (DL5).
# LK1-LK4. Run: ~/.venvs/jev/bin/python ga_label_link_check.py
import sympy as sp

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

N, l, G, al = sp.symbols('N l G alpha', positive=True)
Gind = sp.solve(sp.Eq(1 / (16 * sp.pi * G), N / (96 * sp.pi**2 * l**2)), G)[0]
ratio = sp.simplify(Gind / l**2)                                   # (l_P/l)^2 with hbar = c = 1
A = sp.simplify(8 * sp.pi * al * ratio)
ok(f"LK1 induced gravity: (l_P/l)^2 = {ratio}; linked amplitude A = {A}", sp.simplify(A - 48 * sp.pi**2 * al / N) == 0)

Nmin = sp.N(48 * sp.pi**2 / 0.015)
ok(f"LK2 BBN (A <= 0.015) needs N >= {Nmin:.0f} alpha effective fields at the cutoff", Nmin > 1e4)

# LK3 the Standard Model count: induced-gravity weights per degree of freedom depend on spin and regularisation (known: scalars and
#     fermions enter with weights of order unity, gauge fields with model-dependent sign); an upper estimate is the number of degrees of
#     freedom, 118 (28 bosonic + 90 fermionic).
for Ne in (4, 28, 118):
    print(f"   N = {Ne}: A = {float(48 * sp.pi**2 / Ne):.1f} alpha")
ok(f"LK3 with Standard Model content (N <= 118) the link gives A >= {float(48 * sp.pi**2 / 118):.1f} alpha, above the BBN bound by "
   f"{float(48 * sp.pi**2 / 118 / 0.015):.0f} alpha times", 48 * float(sp.pi**2) / 118 > 0.015)

# LK4 the alpha needed with SM content, and the field count that would make the link harmless
ok(f"LK4 the link survives only if alpha <= {0.015 * 118 / float(48 * sp.pi**2):.1e} (no O(1) uncertainty coefficient) or N >= {float(Nmin):.0f} alpha "
   f"(a hidden sector of ~1e4-1e5 fields at the cutoff)", True)
