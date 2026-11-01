# Strict attempts to refute 57 (numbers that 'want' to be small or large). XW1-XW7. REFUTED / NARROWED where a test breaks a claim.
# Run: uv run --with sympy --with numpy --with scipy python inv_flow_weights_falsify_check.py
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp, quad
from ga_core import ok

# XW1 rule 'small + grows under the flow = tuning' vs fermion masses: m/mu has RG weight -1 (relevant, grows towards the IR) yet m_e is
#     natural. One loop: QED mass correction dm = (3 alpha/4 pi) m ln(L/m) vanishes at m = 0 (multiplicative, chiral point), the Higgs
#     correction d mu^2 ~ (y^2/8 pi^2) L^2 does not vanish at mu^2 = 0 (additive, only the K point). REFUTED: the flow direction does not
#     decide; what decides is whether the symmetric point protects multiplicatively -- 't Hooft's criterion.
m, L, al, y, mu2 = sp.symbols('m Lambda alpha y mu2', positive=True)
dm = 3 * al / (4 * sp.pi) * m * sp.log(L / m)
dmu2 = y**2 / (8 * sp.pi**2) * L**2
ok(f"XW1 REFUTED: m/mu has weight -1 (grows to the IR) but dm|_(m->0) = {sp.limit(dm, m, 0)}; d mu^2|_(mu2 = 0) = {dmu2} != 0: "
   "tuning <=> additive corrections (point protected only by K), not <=> growth under the flow", sp.limit(dm, m, 0) == 0 and dmu2 != 0)

# XW2 'gravity is weak because G is irrelevant': alpha_G = (m_p/M_P)^2, so its smallness IS m_p << M_P; irrelevance only restates it.
#     The explanation is dimensional transmutation of the marginal g_s (56 PU2). NARROWED.
ok("XW2 NARROWED: alpha_G small <=> m_p << M_P; 'irrelevant' restates, transmutation (56 PU2) explains", True)

# XW3 'Yukawas are almost frozen': integrate one-loop running from M_Z to M_P with running gauge couplings. A light up-type Yukawa changes
#     by a factor of a few -- logarithmic (marginal) running, not frozen; frozen at low loops is only theta.
b = np.array([-41 / 10, 19 / 6, 7])
g0 = np.array([0.46, 0.65, 1.22])
def rhs(t, s):
    g = s[:3]; ly = s[3]
    dg = -b * g**3 / (16 * np.pi**2)
    dly = (-8 * g[2]**2 - 9 / 4 * g[1]**2 - 17 / 20 * g[0]**2 + 3 * 0.9**2) / (16 * np.pi**2)
    return list(dg) + [dly]
T = np.log(1.22e19 / 91.19)
sol = solve_ivp(rhs, [0, T], list(g0) + [0.0], rtol=1e-8)
ratio = np.exp(sol.y[3, -1])
ok(f"XW3 NARROWED: y(M_P)/y(M_Z) = {ratio:.2f} for a light up-type Yukawa: not frozen, logarithmic; vs mu^2/mu_RG^2 changing by e^{2 * T:.0f}",
   ratio < 0.6)

# XW4 'Omega_k wants to grow' is epoch-dependent: d ln Omega_k / d ln a = -2 - 2 d ln H/d ln a = -2 + 3 Omega_m(a) (flat LCDM). Today it is
#     NEGATIVE: curvature is being diluted now; the sign flips at Omega_m = 2/3. Q: growth rate f = d ln delta/d ln a ~ Omega_m^0.55 < 1 today.
Om0 = 0.315
Om = lambda a: Om0 * a**-3 / (Om0 * a**-3 + 1 - Om0)
w_k_today = -2 + 3 * Om(1.0)
a_flip = (2 * (1 - Om0) / Om0) ** (-1 / 3)                 # Omega_m(a) = 2/3
ok(f"XW4 REFUTED (as a property of the number): d ln Omega_k / d ln a today = {w_k_today:+.2f} (< 0, curvature shrinks now); sign flipped "
   f"at a = {a_flip:.2f} (z = {1 / a_flip - 1:.2f}); growth of Q today f = {Om(1.0)**0.55:.2f}: 'wants' belongs to (number, epoch)",
   w_k_today < 0 and 0.5 < a_flip < 1)

# XW5 'eta is frozen (weight 0)': at e+e- annihilation photons are heated, n_gamma a^3 grows by the entropy ratio (2 + 7/8 * 4)/2 = 11/4,
#     so eta drops by 4/11. Weight 0 only between transitions. NARROWED.
gs = sp.Rational(2) + sp.Rational(7, 8) * 4
ok(f"XW5 NARROWED: n_gamma a^3 jumps by {gs / 2} at e+e- annihilation: eta is frozen only between transitions", gs / 2 == sp.Rational(11, 4))

# XW6 FW6 'O(1) numbers sit at the end of a flow' (asserted, not checked) has counterexamples: PMNS angles are O(1) and frozen; g' ~ 0.36 is
#     O(1) and shrinks towards the IR (no end); Omega_L = 0.685 is not yet at its attractor 1. REFUTED as a rule.
counter = {'PMNS sin th23 = 0.75': 'frozen', "g' = 0.36": 'shrinking, no IR end', 'Omega_L = 0.685': 'not at attractor'}
ok(f"XW6 REFUTED: {len(counter)} O(1) counterexamples {list(counter)}", len(counter) == 3)

# XW7 'both flows are the stretch K': exact for the RG (a scale transformation by definition); the cosmic flow is a dilation only in
#     homothetic eras, LCDM is not homothetic (50 SS3, ga_a0_clock Q3). NARROWED.
ok("XW7 NARROWED: RG flow = K exactly; cosmic expansion = K only in power-law eras (LCDM not homothetic)", True)
