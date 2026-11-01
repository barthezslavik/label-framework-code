# 64: (a) is there a 'volume' (spherical) rotor that acts on the SDiff pseudoscalar? (b) does a pure spherical symmetry, like SDiff,
# reproduce dark energy? VS1-VS7. Literature (verified 2026-10-20): de Sitter relativity, Lambda = 3/l^2 for the pseudo-sphere radius
# (Aldrovandi, Beltran Almeida, Pereira gr-qc/0606122; arXiv:0812.3438); TDiff = Jacobian 1 / divergence-free generators (Maroto et al.).
# Run: uv run --with sympy --with numpy --with scipy python ga_sphere_de_check.py
import sympy as sp
from ga_core import gp, add, scal, clean, ok

def rev(X):
    return {k: v * (-1)**(len(k) * (len(k) - 1) // 2) for k, v in X.items()}
def sandwich(R, X, S):
    return clean({k: sp.simplify(v) for k, v in gp(gp(R, X, S), rev(R), S).items()})

th, ph, lam, r = sp.symbols('theta phi lambda r', real=True)

# VS1 every ordinary rotor (rotation; the rotor group of 3D is the 3-sphere of unit even multivectors) keeps the pseudoscalar: R I R~ = I.
S3 = {0: 1, 1: 1, 2: 1}; I3 = {(0, 1, 2): 1}
R1 = add({(): sp.cos(th / 2)}, {(0, 1): -sp.sin(th / 2)})
R2 = add({(): sp.cos(ph / 2)}, {(1, 2): -sp.sin(ph / 2)})
R = gp(R1, R2, S3)
S4 = {0: 1, 1: -1, 2: -1, 3: -1}; I4 = {(0, 1, 2, 3): 1}
Rb = add({(): sp.cosh(lam / 2)}, {(0, 1): sp.sinh(lam / 2)})       # a boost rotor
ok(f"VS1 rotors keep the pseudoscalar: 3D R I R~ = {sandwich(R, I3, S3)}, 4D boost: {sandwich(Rb, I4, S4)}: no rotation, 'spherical' or "
   "not, can act on the SDiff pseudoscalar", sandwich(R, I3, S3) == I3 and sandwich(Rb, I4, S4) == I4)

# VS2 a rotor of a BIGGER algebra can: the conformal dilator of Cl(4,1), R = exp(lambda D/2), D = n0 ^ ninf, maps each sphere centred at the
#     origin to a concentric sphere of radius r e^{+-lambda} (a 'spherical volume rotor') and rescales the base volume by e^{3 lambda}.
C = {0: 1, 1: 1, 2: 1, 3: 1, 4: -1}                                  # e1 e2 e3, e+ (3), e- (4)
ninf = {(3,): 1, (4,): 1}; n0 = {(4,): sp.Rational(1, 2), (3,): -sp.Rational(1, 2)}
Dbiv = clean(gp(n0, ninf, C)); Dbiv = {k: v for k, v in Dbiv.items() if len(k) == 2}
sqD = clean({k: sp.simplify(v) for k, v in gp(Dbiv, Dbiv, C).items()})
Rd = add({(): sp.cosh(lam / 2)}, scal(sp.sinh(lam / 2), Dbiv))
Sph = add(n0, scal(-r**2 / 2, ninf))
Sp_ = sandwich(Rd, Sph, C)
# extract radius: write Sp_ = alpha n0 + beta ninf  ->  r'^2 = -2 beta/alpha
alpha, beta = sp.symbols('alpha beta')
sol = sp.solve([sp.Eq(alpha / 2 + beta, Sp_.get((4,), 0)), sp.Eq(-alpha / 2 + beta, Sp_.get((3,), 0))], [alpha, beta])
r2new = sp.simplify(-2 * sol[beta] / sol[alpha])
ok(f"VS2 D^2 = {sqD}; dilator maps the sphere of radius r to radius^2 = {r2new}: concentric spheres go to concentric spheres, base volume "
   "x e^{+-3 lambda}; in Cl(4,1) it is a rotor, on R^3 it changes the pseudoscalar", sqD == {(): 1} and sp.simplify(r2new / r**2 - sp.exp(2 * lam)) * sp.simplify(r2new / r**2 - sp.exp(-2 * lam)) == 0)

# VS3 which generators change the base volume: divergence of the generating field. Rotations, translations: 0 (inside SDiff / TDiff).
#     Dilation: 3 lambda (uniform). Special conformal (transversion): 6 b.x (non-uniform). Diff -> TDiff breaks exactly the generators with
#     non-zero divergence; their local part is K.
x, y, z, b1, b2, b3 = sp.symbols('x y z b1 b2 b3', real=True)
X = [x, y, z]; bb = [b1, b2, b3]
div = lambda v: sp.simplify(sum(sp.diff(v[i], X[i]) for i in range(3)))
bx = sum(bi * xi for bi, xi in zip(bb, X)); x2 = x**2 + y**2 + z**2
fields = {'rotation': [-y, x, 0], 'translation': [1, 0, 0], 'dilation': [lam * x, lam * y, lam * z],
          'transversion': [2 * bx * X[i] - x2 * bb[i] for i in range(3)]}
dv = {k: div(v) for k, v in fields.items()}
ok(f"VS3 divergences {dv}: only dilation and transversion change volume -- the 'volume rotors' are the conformal K-type generators that "
   "Diff -> TDiff removes", dv['rotation'] == 0 and dv['translation'] == 0 and dv['dilation'] == 3 * lam and sp.simplify(dv['transversion'] - 6 * bx) == 0)

# VS4 (b) the pure spherical symmetry: de Sitter = pseudo-sphere X^2 = l^2 in R^{1,4}, group O(1,4) (rotors of Cl(1,4)). In flat slicing
#     a = e^{t/l}: G_mu_nu = (3/l^2) g_mu_nu, i.e. exactly Lambda = 3/l^2, w = -1, constant density.
t, l = sp.symbols('t l', positive=True)
a = sp.exp(t / l); H = sp.diff(a, t) / a
G00 = 3 * H**2
Gii = -(2 * sp.diff(a, t, 2) / a + H**2) * a**2                   # with g = diag(1, -a^2, -a^2, -a^2)
ok(f"VS4 SURVIVES (computed, known): de Sitter sphere gives G_00 = {sp.simplify(G00)} g_00, G_ii = {sp.simplify(Gii / (-a**2))} g_ii: "
   "Lambda = 3/l^2, w = -1 exactly", sp.simplify(G00 - 3 / l**2) == 0 and sp.simplify(Gii / (-a**2) - 3 / l**2) == 0)

# VS4a the sign of the embedding. Flat-slicing chart X0 = l sinh(t/l) + r^2 e^{t/l}/(2l), X4 = l cosh(t/l) - r^2 e^{t/l}/(2l), Xi = e^{t/l} x_i.
#     In R^{1,4} with eta = diag(+,-,-,-,-) (Cl(1,4), the (+,-,-,-) convention extended): X.X = -l^2 and the induced metric is
#     dt^2 - e^{2t/l} dx^2 (Lorentzian de Sitter). The surface X.X = +l^2 there is the two-sheeted hyperboloid with a Riemannian induced
#     metric. In R^{4,1} (the opposite overall sign, Cl(4,1)) the same de Sitter is X.X = +l^2.
x1, x2, x3 = sp.symbols('x1 x2 x3', real=True)
rr = x1**2 + x2**2 + x3**2; ee = sp.exp(t / l)
X = [l * sp.sinh(t / l) + rr * ee / (2 * l), ee * x1, ee * x2, ee * x3, l * sp.cosh(t / l) - rr * ee / (2 * l)]
eta = [1, -1, -1, -1, -1]
XX = sp.simplify(sum(e_ * c**2 for e_, c in zip(eta, X)).rewrite(sp.exp))
crd = [t, x1, x2, x3]
gind = sp.Matrix(4, 4, lambda i, j: sp.simplify(sum(e_ * sp.diff(c, crd[i]) * sp.diff(c, crd[j]) for e_, c in zip(eta, X)).rewrite(sp.exp)))
ok(f"VS4a embedding sign: in R^(1,4) (+,-,-,-,-) de Sitter is X.X = {XX} (= -l^2), induced metric diag{tuple(sp.simplify(gind[i, i]) for i in range(4))}: "
   "Lorentzian; X.X = +l^2 is the right form only in R^(4,1)",
   sp.simplify(XX + l**2) == 0 and sp.simplify(gind - sp.diag(1, -ee**2, -ee**2, -ee**2)) == sp.zeros(4, 4))

# VS5 does the spherical symmetry fix the VALUE? The group O(1,4) has no scale; l is the radius of the orbit (a label), exactly like Lambda as
#     an integration constant in SDiff/unimodular gravity. REFUTED: the value is not fixed.
ok("VS5 REFUTED (value from symmetry): O(1,4) is the same group for every l; the radius is a label -- the same status as Lambda in unimodular "
   "gravity: two views of one constant (volume: SDiff; curvature radius: de Sitter)", True)

# VS6 can a 'breathing' sphere l(t) give evolving w (DESI-like w0 > -1, wa < 0)? With H = 1/l: w = -1 - (2/3) H'/H^2 = -1 + (2/3) l'.
#     Any w(t) maps to some l(t): no prediction; and a breathing sphere is no longer O(1,4)-symmetric. REFUTED: pure spherical symmetry gives
#     only constant Lambda; DESI-like evolution needs l shrinking in the past and growing now, put in by hand.
lt = sp.Function('l')(t)
Ht = 1 / lt
w_eff = sp.simplify(-1 - sp.Rational(2, 3) * sp.diff(Ht, t) / Ht**2)
ok(f"VS6 REFUTED (evolving DE from the sphere): w = {w_eff}: a relabelling of H(t), symmetric only for l' = 0", sp.simplify(w_eff - (-1 + sp.Rational(2, 3) * sp.diff(lt, t))) == 0)

# VS7 the bridge to TDiff: a TDiff matter Lagrangian depends on det g; it is invariant under divergence-free generators (VS3: rotations,
#     translations) and responds to the volume-changing ones (dilation, transversion). So the 'volume rotor' is exactly what TDiff matter
#     feels and Diff matter does not: under the dilation, sqrt(-g) -> lam^4 sqrt(-g) in 4D.
lm = sp.symbols('lambda_s', positive=True)
w_det = sp.simplify(sp.log(sp.sqrt(lm**8)) / sp.log(lm))          # det(lam^2 g) = lam^8 det g
ok(f"VS7 (interpretation, consistent with VS3): sqrt(-g) has weight lam^{w_det} under the dilator; TDiff matter, built from functions of "
   "det g, feels the volume rotor -- that is where TDiff dark-sector models get their new equation of state", w_det == 4)

# VS8 sign conventions for paper5 (review 2026-10-21): signature (+,-,-,-), Gamma and Riemann R^r_{s m n} = d_m G^r_{n s} - d_n G^r_{m s}
#     + G^r_{m l} G^l_{n s} - G^r_{n l} G^l_{m s}, Ricci R_{s n} = R^r_{s r n}. For a = e^{t/l}: G_00 = 3/l^2 = +(3/l^2) g_00, R = -12/l^2.
#     Hence Einstein's equation in this convention is G = kappa T + Lambda g (Lambda = 3/l^2 > 0), its trace -R = kappa T + 4 Lambda, and the
#     trace-free route gives R + kappa T = -4 Lambda.
tt, X1, X2, X3 = sp.symbols('t x1 x2 x3', real=True); ll = sp.symbols('l', positive=True)
co = [tt, X1, X2, X3]; aa = sp.exp(tt / ll)
gm = sp.diag(1, -aa**2, -aa**2, -aa**2); gi = gm.inv()
Gam = [[[sp.simplify(sum(gi[r, q] * (sp.diff(gm[q, m], co[n]) + sp.diff(gm[q, n], co[m]) - sp.diff(gm[m, n], co[q])) for q in range(4)) / 2)
         for n in range(4)] for m in range(4)] for r in range(4)]
def Riem(r, s, m, n):
    return (sp.diff(Gam[r][n][s], co[m]) - sp.diff(Gam[r][m][s], co[n])
            + sum(Gam[r][m][q] * Gam[q][n][s] - Gam[r][n][q] * Gam[q][m][s] for q in range(4)))
Ric = sp.Matrix(4, 4, lambda s, n: sp.simplify(sum(Riem(r, s, r, n) for r in range(4))))
Rs = sp.simplify(sum(gi[i, j] * Ric[i, j] for i in range(4) for j in range(4)))
Gt = sp.simplify(Ric - Rs * gm / 2)
ok(f"VS8 conventions: R = {Rs}, G_00 = {Gt[0, 0]}, G_11/g_11 = {sp.simplify(Gt[1, 1] / gm[1, 1])}: G = +(3/l^2) g, so G = kappa T + Lambda g "
   "with Lambda = 3/l^2, and the trace-free integration gives R + kappa T = -4 Lambda",
   sp.simplify(Rs + 12 / ll**2) == 0 and sp.simplify(Gt[0, 0] - 3 / ll**2) == 0 and sp.simplify(Gt[1, 1] / gm[1, 1] - 3 / ll**2) == 0)
