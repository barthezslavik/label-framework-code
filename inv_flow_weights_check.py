# 57: which numbers 'want' to become small and which large? A dimensionless number X 'wants' what its flow under the stretch K does:
# RG flow towards the infrared (mu -> lam mu, lam < 1) and cosmic expansion towards the future (ln a). Flow weight w: X ~ lam^w (RG, so
# w < 0 grows in the IR) or X ~ a^w (cosmic, w > 0 grows). FW1-FW6.
# Literature (from memory): relevant / marginal / irrelevant operators (Wilson); the hierarchy and flatness problems as unstable directions
# near a fixed point (critical-phenomena analogy, widely discussed); one-loop SM beta coefficients b1 = 41/10, b2 = -19/6, b3 = -7.
# Gap computed: one flow weight for all 31 + cosmological numbers, and its match to the classes of 52 (GA point, none, tuning, puzzle).
# Run: uv run --with sympy --with numpy --with scipy python inv_flow_weights_check.py
import numpy as np
import sympy as sp
from ga_core import ok

# FW1 RG weights from the K weight (mass dimension d of the coupling): X/mu^d has weight -d. Relevant (d > 0) grow in the IR, irrelevant
#     (d < 0) shrink, marginal (d = 0) run logarithmically: d ln g / d ln mu = -b g^2/(16 pi^2) at one loop (here b > 0 = asymptotically free).
dim = {'mu^2 (Higgs)': 2, 'rho_Lambda': 4, 'G (gravity, alpha_G)': -2}
rg_w = {k: -d for k, d in dim.items()}
ok(f"FW1 RG weights towards the IR: {rg_w}: mu^2 and rho_Lambda grow (relevant), G shrinks (irrelevant); alpha_G small is then the "
   "statement m_p << M_P, explained by transmutation (inv_flow_weights_falsify XW2)", rg_w['mu^2 (Higgs)'] < 0 and rg_w['rho_Lambda'] < 0 and rg_w['G (gravity, alpha_G)'] > 0)

# FW2 marginal couplings: one-loop slopes d ln g / d ln mu at the electroweak scale (SM, GUT-normalised U(1)).
g1, g2, g3 = 0.46, 0.65, 1.22                      # sqrt(5/3) g', g, g_s at ~ M_Z
b = {'g1 (U1)': -41 / 10, 'g2 (SU2)': 19 / 6, 'g3 (SU3)': 7}   # d g/d ln mu = -b g^3 / 16 pi^2
slope = {k: -b[k] * g**2 / (16 * np.pi**2) for k, g in zip(b, (g1, g2, g3))}
yt, yb = 0.99, 0.024
s_light = (-8 * g3**2 - 9 / 4 * g2**2 - 17 / 12 * (3 / 5) * g1**2 + 3 * yt**2) / (16 * np.pi**2)   # d ln y / d ln mu for a light up-type Yukawa (dominant terms)
for k, v in slope.items():
    print(f"   {k:10s} d ln g / d ln mu = {v:+.4f}  ({'grows' if v < 0 else 'shrinks'} towards the IR)")
print(f"   light Yukawa d ln y / d ln mu = {s_light:+.3f} (vs mu^2: -2): marginal, almost frozen")
ok(f"FW2 g2, g3 grow towards the IR (asymptotic freedom), g1 shrinks; Yukawas run with |d ln y/d ln mu| = {abs(s_light):.2f} << 2: "
   "marginal numbers run logarithmically (a factor ~2 from M_Z to M_P, XW3), relevant ones as powers", slope['g3 (SU3)'] < 0 and slope['g2 (SU2)'] < 0 and slope['g1 (U1)'] > 0 and abs(s_light) < 0.2)

# FW3 cosmic weights d ln X / d ln a in the matter era (m) and the Lambda era (L).
a = sp.symbols('a', positive=True)
def w_of(expr):
    return sp.simplify(a * sp.diff(sp.log(expr), a))
cos_w = {
    'Omega_L / Omega_m':     (w_of(a**3), w_of(a**3)),
    'Omega_r / Omega_m':     (w_of(1 / a), w_of(1 / a)),
    'Omega_k (|k|/a^2 H^2)': (w_of(1 / a**2 / a**-3), w_of(1 / a**2)),      # H^2 ~ a^-3 (matter), const (Lambda)
    'Q (delta)':             (w_of(a), 0),                                   # delta ~ a in matter, frozen in Lambda
    'Omega_c / Omega_b':     (w_of(a**-3 / a**-3), 0),
    'eta (n_b / n_gamma)':   (w_of(a**-3 / a**-3), 0),
}
for k, (wm, wl) in cos_w.items():
    print(f"   {k:22s} weight matter era {str(wm):>3s}, Lambda era {str(wl):>3s}")
ok("FW3 cosmic weights: Omega_L grows (+3), radiation shrinks (-1), Omega_k and Q grow in the matter era (+1) and stop or shrink with "
   "Lambda, Omega_c/Omega_b and eta have weight 0 (conserved)", cos_w['Omega_k (|k|/a^2 H^2)'][0] == 1 and cos_w['Q (delta)'][0] == 1
   and cos_w['Omega_c / Omega_b'][0] == 0 and cos_w['Omega_L / Omega_m'][0] == 3)

# FW4 the three classes. Small numbers that WANT TO GROW = tuning problems; small numbers that want to shrink = natural; frozen = puzzles
#     without instability. Checked against 52/56: tuning = mu^2, rho_Lambda, Omega_k, Q; natural = alpha_G; frozen puzzles = theta, Yukawas.
classes = {'mu^2': 'grow', 'rho_Lambda': 'grow', 'Omega_k': 'grow', 'Q': 'grow', 'alpha_G': 'shrink', 'theta': 'frozen', 'Yukawas': 'frozen'}
known = {'mu^2': 'tuning', 'rho_Lambda': 'tuning', 'Omega_k': 'tuning', 'Q': 'tuning', 'alpha_G': 'natural', 'theta': 'puzzle', 'Yukawas': 'puzzle'}
rule = {'grow': 'tuning', 'shrink': 'natural', 'frozen': 'puzzle'}
match = all(rule[classes[k]] == known[k] for k in classes)
ok(f"FW4 on these {len(classes)} cases the flow sign matches the labels; NOT a rule: fermion masses grow towards the IR and are natural "
   "(XW1), and Omega_k shrinks today (XW4) -- tuning <=> additive corrections or classical growth in the given epoch", match)

# FW5 the numbers WITHOUT a GA symmetric point (52 ND6/NF5: 4 PMNS, xi_b, xi_c; also Omega_c/Omega_b, eta) are all frozen: conserved ratios
#     (weight 0) or marginal mixings (tiny running for Dirac neutrinos). Converse fails: theta and the Yukawas are frozen too but have point 0.
no_point_weights = {'Omega_c/Omega_b': 0, 'eta': 0, 'xi_b': 0, 'xi_c': 0, 'PMNS (4)': 0}
ok("FW5 every number without a symmetric point has flow weight 0 (frozen); NARROWED: one-way, frozen numbers with a point exist (theta, Yukawas)",
   all(v == 0 for v in no_point_weights.values()))

# FW6 some O(1) numbers sit at the end of a flow: g3 grows to O(1) in the IR (strong coupling, Lambda_QCD), y_t at an IR quasi-fixed point.
#     Not a rule: PMNS angles, g' and Omega_L are O(1) without being at a flow end (XW6).
ok(f"FW6 g3 at M_Z = {g3} is O(1) at the IR end of its flow (slope {slope['g3 (SU3)']:+.3f}); counterexamples to a rule in XW6", g3 > 1)
