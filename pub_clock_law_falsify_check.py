# Attack on the last open point of paper2: 'the clock law a0 ~ 1/t needs a common local a0 and points at z ~ 1-2.5 with one method'. CL1-CL4.
# Run: uv run --with sympy --with numpy --with scipy python pub_clock_law_falsify_check.py
import numpy as np
from scipy.integrate import quad
from ga_core import ok

H0 = 67.4e3 / 3.0857e22; Om, OL = 0.315, 0.685
E = lambda z: np.sqrt(Om * (1 + z)**3 + OL)
t_of_z = lambda z: quad(lambda y: 1 / (y * np.sqrt(Om / y**3 + OL)), 1e-9, 1 / (1 + z))[0] / H0
t0 = t_of_z(0)
Ht = lambda z: E(z) * H0 * t_of_z(z)

# CL1 where the two laws differ in SHAPE: clock/H(z), normalised at z = 0, equals (H t)(z)/(H0 t0). It falls from 1 to ~0.70 and saturates by
#     z ~ 1.5 (matter era, H t -> 2/3). Above z ~ 1.5 the shapes are parallel: high-z points only test the normalisation against the anchor.
zs = [0, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
R = {z: Ht(z) / Ht(0) for z in zs}
print("   clock/H shape ratio: " + ", ".join(f"z={z}: {v:.3f}" for z, v in R.items()))
ok(f"CL1 REFUTED (my own requirement): the shape difference sits at z < 1.5 (ratio {R[0]:.2f} -> {R[1.5]:.2f}); from z = 1.5 to 5 it changes only "
   f"{R[1.5] / R[5.0]:.3f}x -- points at z ~ 1-2.5 cannot separate the laws by shape", R[1.5] / R[5.0] < 1.06 and R[0] - R[1.0] > 0.2)

# CL2 the covariance theorem (pub_paper2_falsify PF4) holds only where the matter-era homothety exists, i.e. where H t = 2/3 exactly; there
#     1/t = (3/2) H, so the theorem supports 'a0 ~ 1/t' and 'a0 ~ H' equally. Where they differ (Lambda era) there is no homothety and the
#     theorem is silent. NARROWED: the clock law as distinct from H(z) is an empirical hypothesis, not a consequence of the theorem.
ok(f"CL2 NARROWED: H t = {Ht(5.0):.4f} at z = 5 (= 2/3 to {abs(Ht(5.0) - 2 / 3):.4f}): in the homothetic era clock and H coincide; the theorem cannot "
   "prefer one; in the Lambda era it does not apply", abs(Ht(5.0) - 2 / 3) < 0.01)

# CL3 the z = 2 bound (Milgrom 2017: ~4 a0 all but excluded, i.e. a0(2) < 4.8e-10 for a local 1.2) turns into a ceiling on the LOCAL a0 for each
#     law: clock a0(0) < 4.8 t(2)/t0, H(z) a0(0) < 4.8/E(2). Compare with the anchors SPARC 1.20 +- 0.24 (sys), MIGHTEE 1.69 +- 0.13.
cap_clock = 4.8 * t_of_z(2) / t0; cap_H = 4.8 / E(2)
ok(f"CL3 ceilings on the local a0 from z = 2: clock {cap_clock:.2f}, H(z) {cap_H:.2f}. SPARC 1.20: clock {(1.20 - cap_clock) / 0.24:+.2f} sigma above the "
   f"ceiling (marginal), H fine; MIGHTEE 1.69 exceeds the clock ceiling by {(1.69 - cap_clock) / 0.13:+.1f} sigma and the H ceiling by {(1.69 - cap_H) / 0.13:+.1f} sigma "
   "(not significant): the anchor decides, and the high anchor is in tension with the z = 2 bound", 1.69 > cap_H and 1.69 > cap_clock and cap_clock < 1.2 < cap_H)

# CL4 what would decide: one method, a common local anchor and points at z <~ 1. Precision for a 3 sigma separation at z = 0.5 and z = 1:
for z in (0.5, 1.0):
    d = 1 - R[z]
    print(f"   z = {z}: laws differ by {100 * d:.1f}% relative to the local anchor -> each of a0(0), a0(z) needs ~{100 * d / 3 / np.sqrt(2):.1f}% precision")
ok("CL4 CORRECTED requirement: a common local anchor plus points at z ~ 0.3-1 with one method, each a0 to ~2-5%; points at z > 1.5 only test the "
   "normalisation", 1 - R[1.0] > 0.2)
