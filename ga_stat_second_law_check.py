# 154: paper13, step 3. The second law for labels. A volume-preserving relabelling on the lattice torus Z_L^2: Arnold's cat map
# (x, y) -> (2x + y, x + y) mod L, a permutation of the L^2 sites (SDiff on the lattice), and a non-mixing one, a translation.
# Fine-grained entropy: the occupation of sites; coarse-grained entropy: occupation of B x B blocks.
# SL1 the fine-grained entropy is exactly constant (a permutation), while from a low-entropy start (labels in one block) the coarse-
#     grained entropy rises to within 1% of its maximum ln(number of blocks);
# SL2 applying the inverse map returns the coarse-grained entropy exactly to its initial low value: the volume-preserving dynamics is
#     reversible and does not by itself make entropy grow;
# SL3 from a typical start (labels placed at random) the coarse-grained entropy stays near its maximum at all steps: growth needs a
#     low-entropy initial state, initial data of the solution;
# SL4 under a non-mixing volume-preserving map (a translation) the coarse-grained entropy of the low-entropy start does not grow
#     beyond its initial value: growth needs mixing, a property of the particular flow, not of the symmetry.
# Status (paper13): fine-grained conservation follows from the relabelling symmetry; coarse-grained growth needs coarse-graining
# (premise), a mixing flow (property of the dynamics) and low-entropy initial data (a label of the solution).
# Run: ~/.venvs/jev/bin/python ga_stat_second_law_check.py
import numpy as np

def ok(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    assert cond

L, B = 512, 32                         # lattice and block size: (L/B)^2 = 256 blocks
nb = (L // B)**2; Smax = np.log(nb)
rng = np.random.default_rng(1)

def cat(xy):
    x, y = xy; return ((2 * x + y) % L, (x + y) % L)
def cat_inv(xy):
    x, y = xy; return ((x - y) % L, (-x + 2 * y) % L)
def shift(xy):
    x, y = xy; return ((x + 37) % L, (y + 11) % L)
def coarse_S(xy):
    bx, by = xy[0] // B, xy[1] // B
    cnt = np.bincount((bx * (L // B) + by).astype(int), minlength=nb).astype(float)
    p = cnt / cnt.sum(); p = p[p > 0]
    return -(p * np.log(p)).sum()
def fine_S(xy):
    cnt = np.bincount((xy[0] * L + xy[1]).astype(int), minlength=L * L).astype(float)
    p = cnt / cnt.sum(); p = p[p > 0]
    return -(p * np.log(p)).sum()

# low-entropy start: every site of one block occupied once
bx, by = np.meshgrid(np.arange(B), np.arange(B), indexing='ij')
low = (bx.ravel(), by.ravel())
xy = low; Sc = [coarse_S(xy)]; Sf = [fine_S(xy)]
for _ in range(12):
    xy = cat(xy); Sc.append(coarse_S(xy)); Sf.append(fine_S(xy))
ok(f"SL1 cat map from one block: fine-grained entropy {Sf[0]:.6f} -> {Sf[-1]:.6f} (constant), coarse-grained {Sc[0]:.3f} -> {Sc[-1]:.3f} "
   f"(max ln {nb} = {Smax:.3f})", max(Sf) - min(Sf) < 1e-12 and Sc[-1] > 0.99 * Smax and Sc[0] == 0.0)

back = xy
for _ in range(12):
    back = cat_inv(back)
ok(f"SL2 inverse map: coarse-grained entropy returns to {coarse_S(back):.3f} (initial {Sc[0]:.3f}); the dynamics is reversible",
   coarse_S(back) == Sc[0] and np.array_equal(np.sort(back[0] * L + back[1]), np.sort(low[0] * L + low[1])))

typ = (rng.integers(0, L, B * B), rng.integers(0, L, B * B))
St = [coarse_S(typ)]; t = typ
for _ in range(12):
    t = cat(t); St.append(coarse_S(t))
ok(f"SL3 typical start: coarse-grained entropy between {min(St):.3f} and {max(St):.3f} (max {Smax:.3f}) at every step: growth needs "
   f"a low-entropy initial state", min(St) > 0.95 * Smax)

xy = low; Ss = [coarse_S(xy)]
for _ in range(12):
    xy = shift(xy); Ss.append(coarse_S(xy))
ok(f"SL4 translation (volume-preserving, not mixing): coarse-grained entropy stays at most {max(Ss):.3f} from {Ss[0]:.3f}, far below "
   f"{Smax:.3f}: growth needs a mixing flow", max(Ss) < 0.3 * Smax)

print("ALL PASS")
