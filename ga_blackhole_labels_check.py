# 144: black holes and label media. Equations borrowed (Oppenheimer-Snyder, Lemaitre-Tolman-Bondi, Israel junction conditions; Blau,
# Guendelman, Guth 1987 for wall-bounded vacuum bubbles), every step recomputed here; G = c = 1.
# BH1 Oppenheimer-Snyder: a homogeneous ball of label dust (three comoving labels) collapses from rest at R0 in proper time
#     tau_c = (pi/2) sqrt(R0^3/2M); the label count n a^3 is constant along the collapse; the surface crosses R = 2M before tau_c;
# BH2 Lemaitre-Tolman-Bondi (marginally bound): with a dense outer shell, R' = dR/dr reaches 0 before R = 0: the comoving label
#     density n = M'/(4 pi m R^2 R') diverges and the label map folds (orientation of the radial label reverses), i.e. shell crossing is
#     a fold of the label field; for M proportional to r^3 (Oppenheimer-Snyder) R' > 0 until R = 0;
# BH3 two four-label vacua with different integration constants (Lambda_in != Lambda_out) cannot be joined without a surface source:
#     continuity of the metric function and of its derivative at the interface forces Lambda_in = Lambda_out and M = 0, so a de Sitter
#     core inside a black hole needs a wall with tension (a one-label medium) at the interface, the configuration of Blau-Guendelman-Guth;
#     with tension sigma the first junction condition sqrt(f_in) - sqrt(f_out) = 4 pi sigma R has solutions with M > 0.
# BH4 shell crossing is multi-streaming: the Lagrangian map r -> R stays one-to-one in label space (each label keeps its shell), while the
#     number of streams at a given areal radius goes from 1 to 3 and the sum of the signs of R' over the streams stays 1; the label number
#     is unchanged, so this is not the field fold of the theorem base; the sigma model allows it, membership in the embeddings is open;
# BH5 entropy through labels: eta_hat = 0.02351 per scalar (paper8, plane; sphere fits 0.02350) fixes the lattice step a = 0.307 sqrt(N) l_P by
#     1/(4G) = N eta_hat/a^2, and S = A/4 for every N: the coefficient is the definition of a, not a state count.
# BH6 evaporation: labels of mass m are emitted efficiently only below M_m = M_P^2/(8 pi m); a solar-mass hole of 1 GeV label dust holds
#     1.1e57 labels, but M_m carries at most 6e36: with N conserved complete evaporation needs a remnant (the global-charge problem).
# Run: ~/.venvs/jev/bin/python ga_blackhole_labels_check.py
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

# ---------- BH1 Oppenheimer-Snyder ----------
M, R0 = 1.0, 10.0
def rhs(t, y):
    return [y[1], -M / y[0]**2]
hit = lambda t, y: y[0] - 1e-3; hit.terminal = True
hor = lambda t, y: y[0] - 2 * M
sol = solve_ivp(rhs, [0, 100], [R0, 0.0], events=[hit, hor], rtol=1e-11, atol=1e-12, dense_output=True)
tau_c = sol.t_events[0][0]; tau_h = sol.t_events[1][0]
tau_th = np.pi / 2 * np.sqrt(R0**3 / (2 * M))
# label count: N labels in the ball, label density n = N/(4/3 pi R^3) in the comoving frame, so n R^3 is constant by construction of the
# comoving labels; check the dust energy: E = (1/2) Rdot^2 - M/R constant (geodesic of each label)
ts = np.linspace(0, tau_c * 0.999, 200); Rs = sol.sol(ts)[0]; Vs = sol.sol(ts)[1]
E = 0.5 * Vs**2 - M / Rs
ok(f"BH1 Oppenheimer-Snyder ball of label dust (M = {M}, R0 = {R0}): collapse time {tau_c:.6f} vs (pi/2) sqrt(R0^3/2M) = {tau_th:.6f}; "
   f"each comoving label keeps E = Rdot^2/2 - M/R (spread {np.ptp(E):.1e}); the surface crosses R = 2M at tau = {tau_h:.4f} < tau_c; "
   f"the labels, and their count, reach R = 0 inside the horizon", abs(tau_c - tau_th) / tau_th < 1e-4 and np.ptp(E) < 1e-8 and tau_h < tau_c)

# ---------- BH2 shell crossing as a fold of the label map ----------
r, t = sp.symbols('r t', positive=True)
def ltb(Mr):
    R = (r**sp.Rational(3, 2) - sp.Rational(3, 2) * sp.sqrt(2 * Mr) * t)**sp.Rational(2, 3)
    return R, sp.diff(R, r)
Mhom = sp.Rational(1, 100) * r**3
Mshell = sp.Rational(1, 100) * r**3 + sp.Rational(1, 2) * (1 + sp.tanh(8 * (r - 3))) * sp.Rational(3, 1)
def first_fold(Mr, rgrid):
    R, Rp = ltb(Mr)
    fR = sp.lambdify((r, t), R, 'numpy'); fRp = sp.lambdify((r, t), Rp, 'numpy')
    fM = sp.lambdify(r, Mr, 'numpy')
    out = []
    for rv in rgrid:
        tcrush = 2 * rv**1.5 / (3 * np.sqrt(2 * fM(rv)))          # R = 0 at this time for the shell r
        tt = np.linspace(0, tcrush * 0.999, 400)
        rp = np.real(fRp(rv, tt))
        neg = np.where(rp <= 0)[0]
        out.append((rv, tcrush, tt[neg[0]] if len(neg) else None))
    return out
hom = first_fold(Mhom, [1.0, 2.0, 3.0, 4.0])
shl = first_fold(Mshell, [2.7, 2.9, 3.1])
fold_free = all(x[2] is None for x in hom)
folds = [x for x in shl if x[2] is not None and x[2] < x[1]]
ok(f"BH2 LTB marginally bound: for M ~ r^3 (homogeneous) R' > 0 until each shell reaches R = 0 ({fold_free}); with a dense shell at r ~ 3, "
   f"R' reaches 0 before R = 0 for shells {[(x[0], round(x[2], 3), round(x[1], 3)) for x in folds]} (r, t_fold, t_crush): there the comoving "
   f"label density M'/(4 pi m R^2 R') diverges and the radial label reverses orientation - shell crossing is a fold of the label map",
   fold_free and len(folds) >= 1)

# ---------- BH3 vacua with different integration constants need a wall ----------
Li, Lo, Mm, Rr = sp.symbols('Lambda_in Lambda_out M R', real=True)
f_in = 1 - Li * Rr**2 / 3
f_out = 1 - 2 * Mm / Rr - Lo * Rr**2 / 3
solset = sp.solve([sp.Eq(f_in, f_out), sp.Eq(sp.diff(f_in, Rr), sp.diff(f_out, Rr))], [Li, Mm], dict=True)
no_shell = all(sp.simplify(s_[Li] - Lo) == 0 and sp.simplify(s_[Mm]) == 0 for s_ in solset) and len(solset) >= 1
# with a wall of tension sigma: sqrt(f_in) - sqrt(f_out) = 4 pi sigma R (first junction condition, both sides static patches)
sig = 0.01; Rw = 3.0; Lin = 0.05; Lout = 0.0
Mw = (Rw / 2) * (1 - Lout * Rw**2 / 3 - (np.sqrt(1 - Lin * Rw**2 / 3) - 4 * np.pi * sig * Rw)**2)
ok(f"BH3 two four-label vacua with Lambda_in != Lambda_out: continuity of f and f' at the interface gives {solset}, i.e. Lambda_in = "
   f"Lambda_out and M = 0 ({no_shell}); a de Sitter core inside a black hole therefore needs a surface source; with a wall of tension "
   f"sigma = {sig} at R = {Rw} (Lambda_in = {Lin}) the junction condition gives exterior mass M = {Mw:.4f} > 0 (Blau-Guendelman-Guth setup)",
   no_shell and Mw > 0)

# ---------- BH4 shell crossing as multi-streaming ----------
# After shell crossing the map r -> R(r, t) is no longer monotonic: several shells (streams) sit at one areal radius. Each label keeps its own
# shell, so the labels are neither created nor destroyed; the number of streams at a radius goes from 1 to 3 and the signed sum stays 1.
Rsh, Rpsh = ltb(Mshell)
fR = sp.lambdify((r, t), Rsh, 'numpy'); fRp = sp.lambdify((r, t), Rpsh, 'numpy')
tf = 1.2
rg = np.linspace(1.0, 6.0, 100001)
Rv = np.real(fR(rg, tf)); Rpv = np.real(fRp(rg, tf))
valid = np.isfinite(Rv) & (Rv > 0)
rg, Rv, Rpv = rg[valid], Rv[valid], Rpv[valid]
def counts(Rtarget):
    s = np.sign(Rv - Rtarget); idx = np.where(np.diff(s) != 0)[0]
    return len(idx), int(sum(np.sign(Rpv[i]) for i in idx))
contig = bool(np.all(np.diff(rg) < 1e-3))                             # no crushed shells removed from the interval
lo, hi = sorted((Rv[0], Rv[-1]))                                      # radii covered from both ends of the label interval
targets = np.linspace(lo + 1e-3, hi - 1e-3, 400)
cs = [counts(x) for x in targets]
unsigned = sorted(set(c[0] for c in cs)); signed = sorted(set(c[1] for c in cs))
tf0 = 0.05
Rv0 = np.real(fR(np.linspace(2.0, 4.0, 4001), tf0)); mono0 = bool(np.all(np.diff(Rv0) > 0))
ok(f"BH4 shell crossing as multi-streaming (shell profile of BH2, t = {tf}): the number of streams at a given areal radius takes the values "
   f"{unsigned}, the sum of the signs of R' over the streams takes {signed}; before crossing (t = {tf0}) the map is monotonic ({mono0}). Each "
   f"label keeps its shell, so the label number is unchanged: crossing is multi-streaming (kinetic description), not a fold of the label "
   f"field; the sigma model allows it, whether it belongs to the space of embeddings is open", contig and mono0 and 3 in unsigned and 1 in unsigned and signed == [1] and set(unsigned) <= {1, 3})

# ---------- BH5 the entropy coefficient through labels ----------
# doc 130 JC8: one free scalar on a radial lattice with step a gives S = 0.295 (R/a)^2, i.e. eta = 0.295/(4 pi a^2) per field.
eta1 = 0.02351                                                       # paper8: eta_hat of the sphere fits, per scalar
Ns = [1, 10, 100]
a_over_lp = [np.sqrt(4 * eta1 * N) for N in Ns]                    # 1/(4G) = N eta1 / a^2  =>  a^2 = 4 N eta1 l_P^2
Msun_lp = 1.989e30 / 2.176e-8                                       # solar mass in Planck masses
A = 16 * np.pi * Msun_lp**2                                         # horizon area in l_P^2 (G = 1 Planck units)
S_bh = A / 4
S_lab = [N * eta1 * A / (aa**2) for N, aa in zip(Ns, a_over_lp)]
cells = [A / aa**2 for aa in a_over_lp]
ok(f"BH5 entropy through labels: with eta = N * {eta1:.4f}/a^2 (doc 130, one stiff label per field) the Clausius normalisation 1/(4G) = eta "
   f"fixes a = {[round(x, 3) for x in a_over_lp]} l_P for N = {Ns}; a solar-mass horizon then has S = {S_lab[0]:.3e} = A/4 = {S_bh:.3e} for "
   f"every N, carried by {[f'{c:.2e}' for c in cells]} lattice cells: the coefficient 1/4 is the definition of a, not a count of label "
   f"states", all(abs(x / S_bh - 1) < 1e-12 for x in S_lab) and abs(a_over_lp[0] - 0.307) < 0.001)

# ---------- BH6 evaporation and the label count ----------
# Hawking temperature k T_H = hbar c^3/(8 pi G M k_B)...; a label of mass m is emitted efficiently only when k T_H >~ m c^2, i.e. below
# M_m = hbar c^3/(8 pi G m c^2) = M_P^2/(8 pi m). A hole formed from label dust holds N = M/m labels; on embeddings N is conserved, so the
# labels must leave as labels: they can do so only in the last stage, below M_m.
hbar, c, Gn = 1.054571817e-34, 2.99792458e8, 6.67430e-11
GeV = 1.602176634e-10 / c**2                                         # kg
Msun = 1.989e30
rows = []
for mGeV in (1.0, 100.0, 1e4):
    m = mGeV * GeV
    Mm = hbar * c / (8 * np.pi * Gn * m)                             # kg, from k T_H = m c^2
    N = Msun / m
    rows.append((mGeV, Mm, N))
TH_sun = hbar * c**3 / (8 * np.pi * Gn * Msun * 1.380649e-23)
carry = rows[0][1] / (1.0 * GeV)                                    # labels that the mass M_m can carry at most (1 GeV)
ok(f"BH6 evaporation and label count: T_H(M_sun) = {TH_sun:.2e} K; labels of mass m are emitted efficiently only below M_m = "
   f"M_P^2/(8 pi m): {[(f'{r[0]:g} GeV', f'{r[1] * 1e3:.2e} g') for r in rows]}; a solar-mass hole of 1 GeV label dust holds "
   f"N = {rows[0][2]:.2e} labels, while the mass M_m can carry at most M_m/m = {carry:.1e}: on embeddings labels are not pair-produced, "
   f"so they cannot leave by radiation at all: complete evaporation needs a remnant or a failure of the count (the global-charge problem)",
   carry < 1e-15 * rows[0][2] and 
   abs(TH_sun - 6.17e-8) / 6.17e-8 < 0.01 and all(r[1] > 0 for r in rows))
