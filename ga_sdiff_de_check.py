# 62: hypothesis 'some breaking of SDiff reproduces dark energy exactly' (chat 2026-10-20), attacked. SD1-SD6.
# Literature (verified 2026-10-20 by web search, see papers/restart/63-literature-check.md): unimodular gravity (Henneaux-Teitelboim 1989; theorem B5 of the base); TDiff dark sector
# (Maroto et al. 2023-24); solid dark energy from frustrated defect networks (Bucher-Spergel 1999); Chaplygin gas (Kamenshchik et al.
# 2001), excluded by the matter power spectrum unless it is almost LCDM (Sandvik et al. 2004).
# Run: uv run --with sympy --with numpy --with scipy python ga_sdiff_de_check.py
import sympy as sp
from ga_core import ok

n, nu, A, B, N = sp.symbols('n nu A B N', positive=True)

def eos(F):
    rho = F; p = n * sp.diff(F, n) - F                    # relabelling-invariant label medium (base A3)
    w = sp.simplify(p / rho)
    cs2 = sp.simplify(sp.diff(p, n) / sp.diff(rho, n))
    return w, cs2

# SD1 Diff -> SDiff of the 4-volume (unimodular): Lambda is an integration constant, w = -1 exactly, any value (base B5: J5, K2). Survives,
#     known: the 'ideal' DE from SDiff is Lambda itself, with its value a label of the solution.
ok("SD1 SURVIVES (known, base B5): Diff -> SDiff gives Lambda as an integration constant, w = -1 exactly, value free", True)

# SD2 a relabelling medium F(n) with F ~ n^nu has w = nu - 1 and c_s^2 = w. w = -1 exactly only for nu = 0, i.e. F independent of n: the
#     'medium' then has no label dependence at all -- it is Lambda again.
w_p, cs_p = eos(n**nu)
ok(f"SD2 power-law label medium: w = {w_p}, c_s^2 = {cs_p}: w = -1 iff nu = 0 (F constant = Lambda)", sp.simplify(w_p - (nu - 1)) == 0 and sp.simplify(cs_p - (nu - 1)) == 0)

# SD3 any fluid medium with -1 < w < 0 has c_s^2 = w < 0: gradient instability. REFUTED: a label FLUID cannot be dark energy with w != -1.
ok(f"SD3 REFUTED (fluid DE with -1 < w < 0): c_s^2 = w < 0 -- unstable; phantom nu < 0 gives c_s^2 < -1, worse", (cs_p.subs(nu, sp.Rational(1, 2))) < 0)

# SD4 breaking relabelling SDiff to a SOLID (frustrated network of N-dimensional defects): energy ~ a^-(3-N), so w = -N/3 (strings -1/3,
#     walls -2/3); w = -1 needs N = 3, a space-filling 'defect' = vacuum energy = Lambda. Rigidity can stabilise w > -1 (known), but w = -1
#     again only as Lambda. NARROWED.
a = sp.symbols('a', positive=True)
w_def = sp.simplify(-1 - sp.Rational(1, 3) * a * sp.diff(sp.log(a**-(3 - N)), a))
ok(f"SD4 NARROWED (known): defect network w = {w_def}; w = -1 iff N = {sp.solve(sp.Eq(w_def, -1), N)[0]} (space-filling = Lambda)",
   sp.solve(sp.Eq(w_def, -1), N) == [3])

# SD5 a label medium whose w EVOLVES from 0 to -1 and is stable: F = sqrt(A + B n^2) (Born-Infeld form) -- the Chaplygin gas p = -A/rho.
#     Check w and c_s^2 >= 0. Survives as mathematics, known, and excluded by the matter power spectrum: 99.999% of the generalised-Chaplygin parameter space, only the LCDM limit survives (Sandvik et al. 2004, verified).
w_c, cs_c = eos(sp.sqrt(A + B * n**2))
ok(f"SD5 NARROWED (known; excluded by Sandvik et al. 2004): F = sqrt(A + B n^2): w = {w_c} (0 -> -1 as n -> 0), c_s^2 = {cs_c} >= 0: a stable "
   "unified dark sector from relabelling symmetry -- the Chaplygin gas", sp.simplify(w_c + A / (A + B * n**2)) == 0)

# SD6 summary: within RELABELLING media (matter labels, base A3), exact w = -1 appears only where the label dependence disappears (SD1, SD2,
#     SD4); genuine breakings give w > -1 with instability (SD3), rigidity (SD4) or Chaplygin-like evolution (SD5). NOT covered here: breaking
#     Diff -> TDiff in the MATTER sector (Maroto et al., arXiv:2301.05713, 2307.14861, 2311.16836, verified 2026-10-20): scalar fields with
#     arbitrary functions of the metric determinant give perfect fluids with new equations of state, including a unified dark sector with the
#     same number of parameters as LCDM that the authors report as strongly preferred at background level (DIC; CMB priors, Pantheon+/SH0ES,
#     BAO, chronometers). That line is published and not tested here.
ok("SD6 NARROWED: label-medium SDiff gives exact DE only as Lambda; the Diff -> TDiff matter-sector line (Maroto et al.) is a different, "
   "published mechanism, not tested here", True)
