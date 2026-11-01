# 71: the flat part of a rotation curve as 'uniform mass per unit radius' (a radial label with uniform measure). FC1-FC7.
# Literature (verified 2026-10-21): singular isothermal sphere (Binney-Tremaine); Lynden-Bell 1967 violent relaxation;
# Milgrom 2009 (arXiv:0810.4065) deep MOND from space-time scale invariance.
# Run: ~/.venvs/jev/bin/python ga_flat_curve_label_check.py
import sympy as sp
from ga_core import ok

r, G, v, A, lam, m, sig, a0, Mb = sp.symbols('r G v A lambda m sigma a0 M_b', positive=True)
p = sp.symbols('p', real=True)
s = sp.symbols('s', positive=True)

# FC1 Newton, spherical: v^2 = G M(r)/r. v = const <=> M = v^2 r/G <=> dM/dr = v^2/G <=> rho = v^2/(4 pi G r^2)
M = v**2 * r / G
rho = sp.diff(M, r) / (4 * sp.pi * r**2)
ok(f"FC1 flat curve <=> dM/dr = v^2/G = const <=> rho = {rho}: constant mass per unit RADIUS, not per volume",
   sp.simplify(sp.sqrt(G * M / r) - v) == 0 and sp.simplify(rho - v**2 / (4 * sp.pi * G * r**2)) == 0)

# FC2 one label: SDiff of a 1D label space (dphi~/dphi = 1) is the shifts phi -> phi + c; the only shift-invariant measure is
#     lambda dphi. With the label phi = r (geometric radius) this is dM = lambda dr: FC1 with lambda = v^2/G.
phi, c = sp.symbols('phi c', real=True)
f = sp.Function('f')
shift_inv = sp.dsolve(sp.Eq(f(phi).diff(phi), 0))       # f(phi + c) = f(phi) for all c <=> f' = 0
ok(f"FC2 1D SDiff = shifts; invariant label measure: {shift_inv.rhs} dphi (uniform); with phi = r: dM/dr = const = flat curve",
   shift_inv.rhs.free_symbols <= {sp.Symbol('C1')})

# FC3 NEGATIVE: realise the radial label as a local k = 1 label medium (walls r = const), phi = g(r), F = -m n, n = |grad phi| = g'.
#     rho = m g'(r); mass per unit label dM/dphi = 4 pi r^2 m: never uniform (grows as r^2), and the tangential pressure is -rho (walls,
#     w = -2/3), not a pressure-supported halo. A local action cannot give 'equal mass per shell'.
g = sp.Function('g')
rho_w = m * sp.diff(g(r), r)
dMdphi = sp.simplify(4 * sp.pi * r**2 * rho_w / sp.diff(g(r), r))
ok(f"FC3 local k=1 wall medium: dM/dphi = {dMdphi} (not uniform for any g); pressure -rho: walls, not a halo -> the radial label is NOT a local label medium",
   sp.diff(dMdphi, r) != 0)

# FC4 relabelling freedom (Moser): for dust with any rho(r), the Lagrangian mass coordinate mu = M(r) has uniform measure dM = dmu.
#     So 'uniform measure' selects nothing unless the label is tied to the geometric radius. Check: rho = A r^p, mu = M(r): dM/dmu = 1.
Mp = sp.integrate(4 * sp.pi * s**2 * A * s**p, (s, 0, r), conds='none')
ok("FC4 any profile has uniform measure in its own mass label (dM/dmu = 1): uniformity is a statement about the label phi = r, not about SDiff",
   sp.simplify(sp.diff(Mp, r) / sp.diff(Mp, r)) == 1)

# FC5 scale invariance: rho = A r^p, v^2 = G M/r. v(lambda r) = v(r) for all lambda (solution invariant under r -> lambda r at fixed
#     velocity) iff p = -2. Same scaling (t, x) -> l (t, x) as Milgrom's deep-MOND symmetry; in Newton it is a property of the SOLUTION.
v2 = sp.simplify(G * 4 * sp.pi * A * r**(p + 3) / (p + 3) / r)
expo = sp.simplify(sp.diff(sp.log(v2), r) * r)
ok(f"FC5 v^2 ~ r^({expo}) for rho ~ r^p: scale-free (flat) iff p = {sp.solve(expo, p)}",
   sp.solve(expo, p) == [-2])

# FC6 the scale-free dust solution exists dynamically: isotropic Jeans equation d(rho sigma^2)/dr = -rho G M/r^2 with sigma = const
#     is solved by the singular isothermal sphere rho = sigma^2/(2 pi G r^2), v_c^2 = 2 sigma^2 (known). Violent relaxation
#     (phase-space incompressibility + mixing, Lynden-Bell 1967) drives collisionless dust towards this state.
rho_sis = sig**2 / (2 * sp.pi * G * r**2)
M_sis = sp.integrate(4 * sp.pi * s**2 * rho_sis.subs(r, s), (s, 0, r))
jeans = sp.simplify(sp.diff(rho_sis * sig**2, r) + rho_sis * G * M_sis / r**2)
ok(f"FC6 singular isothermal sphere solves Jeans: residual {jeans}; v_c^2 = {sp.simplify(G * M_sis / r)}",
   jeans == 0 and sp.simplify(G * M_sis / r - 2 * sig**2) == 0)

# FC7 restatement (not a derivation): with lambda = dM/dr, v^2 = G lambda. The BTFR v^4 = G a0 M_b then reads lambda = sqrt(a0 M_b/G):
#     the halo's mass per unit radius is set by the baryonic mass. Nothing in FC1-FC6 fixes lambda; this is the open link to baryons.
lam_btfr = sp.solve(sp.Eq((G * lam)**2, G * a0 * Mb), lam)
ok(f"FC7 BTFR in label form: lambda = {lam_btfr} -- a restatement; the baryon link is NOT derived",
   len(lam_btfr) == 1 and sp.simplify(lam_btfr[0] - sp.sqrt(a0 * Mb / G)) == 0)

# FC8 (user: 'not a halo, a different, almost Lambda-like geometry') the k = 2 rung with the two ANGLES as labels and the uniform
#     measure on the sphere of directions (f = sin theta): radial strings, n = 1/r^2, so rho = m/r^2 automatically (M ~ r: the flat-
#     curve profile with no length scale), p_r = -rho (tension along the strings), p_t = 0. Same stress as a global monopole.
t, th, ph_ = sp.symbols('t theta phi_', positive=True)
Xs = [t, r, th, ph_]
gs = sp.diag(1, -1, -r**2, -r**2 * sp.sin(th)**2); gsi = gs.inv()
dXs = sp.Matrix(4, 2, lambda a_, b_: sp.diff([th, ph_][b_], Xs[a_]))
Bs = dXs.T * gsi * dXs
ns = sp.simplify(sp.sin(th) * sp.sqrt(Bs.det())).subs(sp.Abs(sp.sin(th)), sp.sin(th))  # 0 < theta < pi
Ts = sp.simplify(gsi * (m * ns * (gs - dXs * Bs.inv() * dXs.T)))
ok(f"FC8 radial string cloud from angular labels: n = {ns}, rho = {Ts[0,0]}, p_r = {-Ts[1,1]}, p_t = {-Ts[2,2]}",
   sp.simplify(ns - 1 / r**2) == 0 and sp.simplify(Ts[0, 0] - m / r**2) == 0 and sp.simplify(Ts[1, 1] - m / r**2) == 0 and Ts[2, 2] == 0)

# FC9 active gravitational density on the ladder: rho + 3 p_mean = rho (1 + 3 w) = (k - 2) rho. Only dust (k = 3) attracts;
#     strings (k = 2) are neutral, walls and the vacuum repel.
k_ = sp.symbols('k')
ok(f"FC9 active density rho(1 + 3w) with w = k/3 - 1: {sp.expand(1 + 3 * (k_ / 3 - 1))} * rho -> k = 2 gives zero Newtonian force",
   sp.expand(1 + 3 * (k_ / 3 - 1)) == k_ - 2)

# FC10 NEGATIVE in GR: point mass M in the radial string cloud: g_tt = 1 - a - 2GM/r, a = 8 pi G m (Letelier 1979; Barriola-Vilenkin
#     1989 form). Einstein tensor G^t_t = G^r_r = a/r^2, G^th_th = 0 matches FC8. Circular speed v^2 = GM/((1 - a) r - 2GM):
#     Keplerian, only rescaled; NO flat curve. The cloud changes the geometry (solid-angle deficit a, constant light deflection) but
#     not the force. Flat curves need extra ingredients: Soleng 1995 (other string-fluid EOS), Nucamendi-Salgado-Sudarsky 2001
#     (non-minimal coupling).
a_ = sp.symbols('a', positive=True)
Mm = sp.symbols('M', positive=True)
Ag = 1 - a_ - 2 * G * Mm / r
gg = sp.diag(Ag, -1 / Ag, -r**2, -r**2 * sp.sin(th)**2); ggi = gg.inv()
Gam = [[[sum(ggi[i, s_] * (sp.diff(gg[s_, j], Xs[k2]) + sp.diff(gg[s_, k2], Xs[j]) - sp.diff(gg[j, k2], Xs[s_])) for s_ in range(4)) / 2
         for k2 in range(4)] for j in range(4)] for i in range(4)]
Ric = sp.Matrix(4, 4, lambda b_, d_: sp.simplify(sum(sp.diff(Gam[a2][b_][d_], Xs[a2]) - sp.diff(Gam[a2][b_][a2], Xs[d_])
      + sum(Gam[a2][a2][e] * Gam[e][b_][d_] - Gam[a2][d_][e] * Gam[e][b_][a2] for e in range(4)) for a2 in range(4))))
Rsc = sp.simplify(sum(ggi[i, j] * Ric[i, j] for i in range(4) for j in range(4)))
Ein = sp.simplify(ggi * (Ric - Rsc * gg / 2))
v2s = sp.simplify(r * sp.diff(Ag, r) / (2 * Ag))
ok(f"FC10 GR: G^t_t = {Ein[0,0]}, G^r_r = {Ein[1,1]}, G^th_th = {Ein[2,2]}; v^2 = {v2s} ~ 1/r: string cloud bends geometry, not orbits",
   sp.simplify(Ein[0, 0] - a_ / r**2) == 0 and sp.simplify(Ein[1, 1] - a_ / r**2) == 0 and Ein[2, 2] == 0
   and sp.limit(v2s * r, r, sp.oo) != 0)

# FC11 the rotation curve as the shape of LABEL space: dust with labels (psi(r), theta, phi), uniform sphere measure sin(theta):
#      rho = m psi'(r)/r^2. Ball (psi = r^3/3): rho = const, solid-body; cylinder R x S^2 (psi = r): rho = m/r^2, flat; ball inside and
#      cylinder outside (psi = r - r_c atan(r/r_c)): rho = m/(r^2 + r_c^2), solid-body core then plateau. Dropping the radial label
#      (k = 2) keeps the 1/r^2 profile but the active density is (k - 2) rho = 0 (FC9): one label separates 'geometry' from 'halo'.
rc = sp.symbols('r_c', positive=True)
def dust_rho(psi):
    dXd = sp.Matrix(4, 3, lambda a_, b_: sp.diff([psi, th, ph_][b_], Xs[a_]))
    Bd = dXd.T * gsi * dXd
    return sp.simplify((m * sp.sin(th) * sp.sqrt((-Bd).det())).subs(sp.Abs(sp.sin(th)), sp.sin(th)))
rb, rcyl, rmix = dust_rho(r**3 / 3), dust_rho(r), dust_rho(r - rc * sp.atan(r / rc))
rmix = sp.factor(sp.simplify(rmix.subs(sp.Abs(sp.sin(th)), sp.sin(th))**2))
rmix = sp.sqrt(rmix).simplify()
ok(f"FC11 label-space shape -> curve: ball rho = {rb}, cylinder rho = {rcyl}, ball+cylinder rho = {sp.simplify(rmix)}",
   rb == m and sp.simplify(rcyl - m / r**2) == 0 and sp.simplify(rmix - m / (r**2 + rc**2)) == 0)

# FC12 where does the ball (Newtonian, baryon-dominated) turn into the cylinder (plateau)? Suppose the transition radius r_t is set
#      by a threshold on a k-label density of the baryons, M_b / r_t^k = const (k = 3: volume density, k = 2: mass per area of the
#      sphere of directions, k = 1: mass per radius). Continuity at r_t: v^2 = G M_b / r_t. Then M_b ~ v^s with s = 2k/(k - 1):
#      k = 3 -> 3, k = 2 -> 4, k = 1 -> no relation (v independent of M_b).
k3 = sp.symbols('k', positive=True)
Cth = sp.symbols('C', positive=True)
rt = (Mb / Cth)**(1 / k3)
vsq = G * Mb / rt
s_exp = sp.simplify(1 / (sp.diff(sp.log(vsq), Mb) * Mb / 2))           # M_b ~ v^s, s = dlnM/dlnv
ok(f"FC12 transition at a threshold on the k-label density M_b/r^k: M_b ~ v^s, s = {s_exp}; k=3 -> {s_exp.subs(k3, 3)}, k=2 -> {s_exp.subs(k3, 2)}",
   sp.simplify(s_exp - 2 * k3 / (k3 - 1)) == 0 and s_exp.subs(k3, 2) == 4 and s_exp.subs(k3, 3) == 3)

# FC13 data: measured BTFR slope s = 3.85 +- 0.09 (Lelli 2019; systematic range 3.5-4.0; bulge/colour-corrected ~4; our deep-regime
#      selection-corrected 3.7-3.9) -> k = s/(s - 2) = 2.08 +- 0.05 (2.0-2.33). A threshold on the k = 3 (volume) density, the
#      CDM virial-type criterion, gives s = 3; the data select the k = 2 rung: the mass per unit area of the sphere of directions,
#      i.e. the density of the angular (string) labels of FC8.
sv, dsv = 3.85, 0.09
kd = sv / (sv - 2); dkd = 2 * dsv / (sv - 2)**2
ok(f"FC13 k = s/(s-2) = {kd:.2f} +- {dkd:.2f}; systematic range s 3.5-4.0 -> k {4.0/2:.2f}-{3.5/1.5:.2f}; k = 3 (s = 3) is {(sv-3)/dsv:.0f} sigma away",
   abs(kd - 2) < 2 * dkd and (sv - 3) / dsv > 5)

# FC14 with the k = 2 threshold written as an acceleration, G M_b / r_t^2 = a0: lambda = M_b / r_t = sqrt(a0 M_b / G) (FC7) --
#      the halo's mass per unit radial label equals the baryonic mass per transition radius; the dark mass on the cylinder up to r_t
#      equals M_b. a0/G is the threshold surface density (a label, doc 53).
rt2 = sp.sqrt(G * Mb / a0)
ok(f"FC14 k=2 threshold G M_b/r_t^2 = a0 gives lambda = M_b/r_t = {sp.simplify(Mb / rt2)} = FC7 BTFR; lambda r_t = M_b",
   sp.simplify(Mb / rt2 - sp.sqrt(a0 * Mb / G)) == 0)

# FC15 why k = 2 is an acceleration: the angular label blade with the uniform sphere measure, sin(theta) grad(theta) ^ grad(phi), is
#      in 3D the dual of r_hat/r^2 (wedge -> cross product), i.e. of the Newtonian field of a point mass divided by -GM. Closedness of
#      the blade = div g = 0 outside the mass. So the density of angular labels IS the field strength (Gauss): a threshold on the k = 2
#      density is a threshold on the acceleration g.
xc, yc, zc = sp.symbols('x y z', real=True)
R3 = sp.sqrt(xc**2 + yc**2 + zc**2)
TH = sp.acos(zc / R3); PH = sp.atan2(yc, xc)
grad = lambda F_: sp.Matrix([sp.diff(F_, q) for q in (xc, yc, zc)])
blade_dual = sp.simplify((sp.sin(TH) * grad(TH).cross(grad(PH))))
rhat_r2 = sp.Matrix([xc, yc, zc]) / R3**3
divb = sp.simplify(sum(sp.diff(blade_dual[i], q) for i, q in enumerate((xc, yc, zc))))
ok(f"FC15 sin(th) grad th ^ grad phi = dual of r_hat/r^2 (= -g/(GM)): {sp.simplify(blade_dual - rhat_r2) == sp.zeros(3, 1)}; closed (div = {divb})",
   sp.simplify(blade_dual - rhat_r2) == sp.zeros(3, 1) and divb == 0)

# FC16 field-line count: N(r) = r^2 g(r)/G = enclosed mass (Gauss). Newton (ball, baryons inside): N = M_b. Deep regime
#      g = sqrt(a0 G M_b)/r: N = lambda r with lambda = sqrt(a0 M_b/G) -- the cylinder: field lines (enclosed mass) grow by a
#      constant amount per unit radius; N(r_t) = M_b at r_t = sqrt(G M_b/a0) (continuity, FC14).
gN = G * Mb / r**2; gD = sp.sqrt(a0 * G * Mb) / r
ND = sp.simplify(r**2 * gD / G)
ok(f"FC16 N = r^2 g/G: Newton {sp.simplify(r**2*gN/G)}, deep {ND} = lambda r; N(r_t) = {sp.simplify(ND.subs(r, sp.sqrt(G*Mb/a0)))}",
   sp.simplify(r**2 * gN / G - Mb) == 0 and sp.simplify(ND - sp.sqrt(a0 * Mb / G) * r) == 0
   and sp.simplify(ND.subs(r, sp.sqrt(G * Mb / a0)) - Mb) == 0)

# FC17 the framework's own acceleration scale: the de Sitter sphere (Lambda as a label) has c^2/l = c H_inf. Numbers (Planck 2018:
#      H0 = 67.4, Omega_L = 0.685; RAR g+ = 1.20 +- 0.02 (stat) +- 0.24 (sys) e-10 m/s^2, McGaugh-Lelli-Schombert 2016):
#      a0 / (c H_inf / 2 pi) = 1.39 +- 0.28. Recorded as a comparison of two labels, NOT a derivation.
import math
cH = 2.998e8 * 67.4e3 / 3.0857e22
cHinf = cH * math.sqrt(0.685)
ratio = 1.20e-10 / (cHinf / (2 * math.pi)); dratio = ratio * math.hypot(0.02, 0.24) / 1.20
ok(f"FC17 c H0 = {cH:.3g}, c H_inf = {cHinf:.3g} m/s^2; a0/(c H_inf/2pi) = {ratio:.2f} +- {dratio:.2f} (comparison only)",
   0.5 < ratio < 2.5)

# FC18 test inside the framework: H(z) is state, H_inf the form constant (doc 65). If the k = 2 threshold is tied to the de Sitter
#      sphere, a0 is the same at every z; if it were tied to H(z), a0(z)/a0(0) = H(z)/H0 = 1.79 at z = 1, 3.03 at z = 2. High-z rotation
#      curves discriminate.
Ez = lambda z_: math.sqrt(0.315 * (1 + z_)**3 + 0.685)
ok(f"FC18 a0 tied to H_inf: constant in z; tied to H(z): x{Ez(1):.2f} at z=1, x{Ez(2):.2f} at z=2 -- discriminating test",
   abs(Ez(1) - 1.79) < 0.01 and abs(Ez(2) - 3.03) < 0.01)

# FC19 mechanism candidate inside the framework: the static potential Phi as ONE label (k = 1) of 3D space, action int d^3x F(n),
#      n = |grad Phi|. The 3D analogue of the radiation theorem: under x -> lam x with Phi fixed, d^3x -> lam^3, n -> n/lam, so the
#      action is dilation-blind iff F ~ n^(3/k) = n^3. Newton is F ~ n^2 (weight lam^1, not blind).
lamx, pp = sp.symbols('lambda_x p', positive=True)
nn = sp.symbols('n', positive=True)
weight = sp.simplify(sp.log((lamx**3 * (nn / lamx)**pp) / nn**pp) / sp.log(lamx))
ok(f"FC19 3D dilation weight of int F(n), F = n^p: {weight}; blind iff p = {sp.solve(weight, pp)} (k = 1, d = 3); Newton p = 2 has weight 1",
   sp.solve(weight, pp) == [3] and weight.subs(pp, 2) == 1)

# FC20 field equation div(F'(n) grad Phi / n) = source: for a point mass the flux is conserved, r^2 n^(p-1) = K M, so
#      g = n ~ r^(-2/(p-1)) and v^2 = g r ~ r^((p-3)/(p-1)): flat iff p = 3. (p = 4, the 4D-conformal power, gives v^2 ~ r^(1/3).)
Kc = sp.symbols('K', positive=True)
gp_ = (Kc * Mb / r**2)**(1 / (pp - 1))
vexp = sp.simplify(sp.diff(sp.log(gp_ * r), r) * r)
ok(f"FC20 point source: v^2 ~ r^({vexp}); flat iff p = {sp.solve(vexp, pp)}; p = 4 gives r^({vexp.subs(pp, 4)})",
   sp.solve(vexp, pp) == [3] and vexp.subs(pp, 4) == sp.Rational(1, 3))

# FC21 normalising the p = 3 medium as div(|grad Phi| grad Phi / a0) = 4 pi G rho: g^2 r^2 = a0 G M_b -> g = sqrt(a0 G M_b)/r,
#      v^4 = G a0 M_b: the BTFR with s = 4. a0 is the ratio of the coefficients of the n^2 (Newton) and n^3 (dilation-blind) terms;
#      the transition n = a0 is where the two balance -- the k = 2 threshold of FC12/FC15 (the flux of the k = 1 label is the k = 2
#      field-line blade). This is the AQUAL structure (Bekenstein-Milgrom 1984) read on the label ladder.
gsol = sp.sqrt(a0 * G * Mb) / r
ok(f"FC21 p = 3: g^2 r^2/a0 = G M_b -> g = {gsol}, v^4 = {sp.simplify((gsol * r)**2)}",
   sp.simplify(gsol**2 * r**2 / a0 - G * Mb) == 0 and sp.simplify((gsol * r)**2 - G * a0 * Mb) == 0)

# FC22 constraint (doc 66): in this reading the interpolation between n^2 and n^3 must be sharp; the standard simple-mu, RAR and
#      nu = sqrt(1 + 1/y) forms give Q2 = 2-4e-26 s^-2 in the Solar System, 4-11x above Cassini/DE440 (mond_solar_q2_check K1-K7).
ok("FC22 recorded: the k = 1 potential-label reading inherits the Solar-System bound of doc 66 -- the tested smooth interpolations (simple-mu, RAR, sqrt form) fail", True)

# FC23 which stretch selects which law: S = int d^3x [ -|grad Phi|^p / c_p - rho Phi ] under x -> lam x with Phi fixed and
#      rho -> lam^-q rho. Weights: gradient term 3 - p, source term 3 - q; the action is covariant (one overall factor) iff p = q.
#      Masses fixed (q = 3, the deep-MOND scaling; masses are label counts, conserved) -> p = 3, deep MOND. Masses scaling as lam
#      (q = 2, the matter-era homothety) -> p = 2, Newton.
qq = sp.symbols('q', positive=True)
wgrad, wsrc = 3 - pp, 3 - qq
ok(f"FC23 covariance iff {sp.solve(sp.Eq(wgrad, wsrc), pp)} = q: masses fixed (q=3) -> p=3 deep MOND; homothety (q=2) -> p=2 Newton",
   sp.solve(sp.Eq(wgrad, wsrc), pp) == [qq])

# FC24 FALSIFICATION of the FC17/FC18 tie a0 ~ c H_inf (a0 constant in z): against the published MUSE-DARK line a0(z) = 1.00 + 1.59 z
#      (0.33 < z < 1.44) a constant a0 deviates by up to 10.2 sigma (inv_clock_combos_check P4: clock 1/t 1.0 sigma, H(z) 2.7 sigma).
#      If that evolution is confirmed, a0 is state, not the de Sitter label; the tie to H_inf is then refuted.
ok("FC24 recorded: constant a0 (tie to H_inf) is 10.2 sigma from the published a0(z) line; the tie survives only if that evolution is not confirmed",
   True)

# FC25 weights under the SPACETIME dilation K: g -> lam^2 g, labels fixed, label measure f -> lam^wf f (wf = 0: K acts on spacetime only,
#      internal data fixed; wf = 1: K combined with an internal rescaling of the measure, masses ~ lam). Point particle / dust
#      -m int d^3phi f int dtau: weight 1 + wf (dtau -> lam dtau). Coupling -m int dtau Phi (Phi weight 0): the same. Potential as a
#      k = 1 label medium, int sqrt(-g) |g^{mu nu} d Phi d Phi|^(p/2): weight 4 - p. Einstein-Hilbert int sqrt(-g) R: weight 2.
#      Homogeneous total action iff 4 - p = 1 + wf: wf = 0 -> p = 3 (deep MOND, in 4D as well), wf = 1 -> p = 2 (Newton; = EH weight 2,
#      i.e. GR's own scaling g -> lam^2 g, M -> lam M). Supersedes the FC20 remark: the requirement is equal weight with matter, not zero.
lam_, wf = sp.symbols('lambda_ w_f', positive=True)
gL = sp.diag(1, -1, -1, -1); sqrtg = lambda gg: sp.sqrt(-gg.det())
w_sqrtg = sp.simplify(sp.log(sqrtg(lam_**2 * gL) / sqrtg(gL)) / sp.log(lam_))         # 4
w_X = sp.simplify(sp.log((lam_**-2)) / sp.log(lam_))                                   # g^{mu nu} -> lam^-2: X weight -2
w_pot = sp.simplify(w_sqrtg + w_X * pp / 2)                                            # 4 - p
w_EH = sp.simplify(w_sqrtg + w_X)                                                      # R ~ g^{-1} d d g: weight -2 -> 2
w_dust = 1 + wf
sol = sp.solve(sp.Eq(w_pot, w_dust), pp)[0]
ok(f"FC25 K-weights: potential medium {w_pot}, dust/coupling {w_dust}, Einstein-Hilbert {w_EH}; homogeneous iff p = {sol}: "
   f"wf=0 -> p={sol.subs(wf, 0)} (deep MOND), wf=1 -> p={sol.subs(wf, 1)} (Newton, = EH weight)",
   w_sqrtg == 4 and sol.subs(wf, 0) == 3 and sol.subs(wf, 1) == 2 and w_EH == 2 and w_EH == w_dust.subs(wf, 1))

# FC26 in flat space the metric dilation g -> lam^2 g is, up to the diffeomorphism x -> lam x, the coordinate scaling (t, x) -> lam (t, x)
#      at fixed velocities: the deep-MOND scaling is K with internal data fixed (wf = 0); the matter-era homothety (M ~ lam) is K with
#      wf = 1. Check on the static 3D+time reduction: int dt d^3x |grad Phi|^p has weight 4 - p, int dt M Phi has weight 1 + wf.
w3 = (1 + 3 - pp); ok(f"FC26 static reduction: field weight {w3}, source weight 1 + wf -> same selection p = 3 - wf as in 4D",
                      sp.solve(sp.Eq(w3, 1 + wf), pp)[0] == sol)
