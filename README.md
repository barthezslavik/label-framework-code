# Code and checks for the label-framework series

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23871789.svg)](https://doi.org/10.5281/zenodo.23871789)
[![Licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)

The label framework describes matter as a field of comoving scalar labels on spacetime whose relabellings preserve the label volume.
Under premises stated in each paper, the series obtains the media of cosmology (dust, radiation, strings, walls, the vacuum) from the
number of labels and a condition on the action, treats the cosmological constant as a constant of integration and compares it with DESI DR2,
describes galaxy rotation curves, the vacuum as q-theory and black holes built from label dust, and collects the results as theorems in
geometric algebra. This repository holds the Python scripts (numpy, scipy, sympy) that compute or check those results.

The scripts in this repository compute or check the statements of eleven texts, all open access on Zenodo:

- [Volume-preserving relabelling and a label-count classification of cosmic media](https://doi.org/10.5281/zenodo.24021766) (paper1, version 4): which medium (dust, radiation, strings, walls, vacuum) each condition on the action gives
- [Rotation curves as the shape of a label space](https://doi.org/10.5281/zenodo.23871775) (paper3, version 3): galaxy rotation curves and the baryonic Tully–Fisher slope
- [The cosmological constant as a label](https://doi.org/10.5281/zenodo.24021767) (paper5, version 3): Λ as a constant of integration; fit to DESI DR2
- [What the label framework predicts: a catalogue of tests across physics](https://doi.org/10.5281/zenodo.23871778) (paper7): 62 testable items with status and data
- [The volume axis, horizon equilibrium and spin as a relation](https://doi.org/10.5281/zenodo.23871780) (paper8): signature (1,3), the trace-free Einstein equation, half-integer spin
- [A dimensionless theorem base in geometric algebra](https://doi.org/10.5281/zenodo.24023762) (theorem base, version 2): 100 theorems with premises, proof ideas and checks
- [The vacuum in the label framework](https://doi.org/10.5281/zenodo.24023150) (paper10, version 2): the four-label vacuum as q-theory; its rigidity
- [Black holes in the label framework](https://doi.org/10.5281/zenodo.24023151) (paper11, version 2): collapse of label dust, shell crossing, entropy step, remnants
- [Quantum mechanics in the label framework: one body, many bodies, statistics, spin and internal quantum numbers](https://doi.org/10.5281/zenodo.24047587) (paper12): what labels derive and postulate in quantum mechanics; many bodies from densities on configuration space; internal quantum numbers
- [Statistical mechanics in the label framework: entropy as a label count, the second law as initial data](https://doi.org/10.5281/zenodo.24055978) (paper13): entropy as a count of labels; the second law not derived; gradient flows and the Fisher term; equilibrium
- [The language of labels in physics: an overview of a series](https://doi.org/10.5281/zenodo.24055979) (overview, version 4): reading order, premises of the main results, the classes of numbers

The papers state results and proofs without naming scripts. The map from statements to scripts and the labels of their checks
is in `checks/`: `theorem-base.md` (every theorem of the theorem base), `paper8.md`, `paper10.md`, `paper11.md`, `paper12.md`, `paper13.md`, `paper14.md` (in preparation) and `overview.md`. Every check prints a labelled
`PASS`/`FAIL` line (for example `ga_rindler_eta_check` RE3). The repository holds only the scripts the texts rest on, the scripts they call,
their stored outputs and one small table of our own; the papers themselves are on Zenodo (DOIs below).

## Layout

| Files | Content |
|---|---|
| `ga_*.py`, `inv_*.py`, `pub_*.py`, `paper2_cs_check.py` | symbolic and numerical checks without external data; each prints labelled `PASS`/`FAIL` lines |
| `de_*.py`, `btfr_*.py`, `rar_*.py`, `mond_*.py`, `csdr_*.py` | checks against published data (the data directory is a command-line argument) |
| `ga_core.py` | the geometric-algebra helpers used by the checks |
| `run_all.py`, `checkcache.py` | run every data-free check in parallel, with a cache keyed by the content of each script and of what it reads (`--full` recomputes everything) |
| `checks/*.md` | which script and which labels check each statement |
| `theorem_base_check.py` | runs the 100 theorems of the theorem base and the falsification pass and asserts that every listed label prints `PASS` |
| `figures/fig_paper*.py` | draw the figures of the papers (byte for byte reproducible) |
| `*.out`, `*.txt`, `results/` | stored outputs read by some checks (300 realisations of the fluctuating Λ, entanglement sums) |
| `data/a0z_points.csv` | the a₀(z) points used by the clock checks, transcribed from the cited papers |

## Running

Python 3.11 or later with `numpy`, `scipy`, `sympy` and `matplotlib`.

```
python run_all.py              # every data-free check; prints a TOTAL line with 0 FAIL
python theorem_base_check.py   # the theorem base: 100 theorems and the checks of paper12, paper13 and paper14, 574 labelled checks
```

`ga_rindler_eta_check.py` takes about nine minutes; with the other checks in parallel a full run takes about ten minutes, and a cached rerun seconds.

Checks against data take the data directory as an argument. The data are third-party files: run these scripts with `python -I`,
from a directory that holds no other code, e.g.

```
python -I de_desi_fit_check.py DATA_ROOT
python -I de_h3_s8_fit_check.py DATA_ROOT
python -I figures/fig_paper3_btfr.py SPARC_DIR OUT.pdf
```

## Data (not included)

| Directory under `DATA_ROOT` | Source |
|---|---|
| `desi_dr2_bao/` | DESI DR2 BAO means and covariance, CobayaSampler `bao_data` repository (DESI Collaboration, arXiv:2503.14738) |
| `planck2018_distance_priors/` | Planck 2018 distance priors (Chen, Huang & Wang 2019, arXiv:1808.05724) |
| `pantheonplus/` | Pantheon+ data release, github.com/PantheonPlusSH0ES/DataRelease |
| `des_sn5yr_dovekie/` | DES-SN5YR, Dovekie recalibration, github.com/des-science/DES-SN5YR (arXiv:2511.07517) |
| `hsc_2301.04664/`, `hsc_chaurasiya2023/` | HSC weak-lensing stellar-to-halo relations (arXiv:2301.04664; Chaurasiya et al.) |
| `kids_brouwer2021/` | KiDS-1000 lensing profiles (Brouwer et al. 2021) |
| `mandelbaum2016/` | SDSS blue centrals, Mandelbaum et al. 2016, MNRAS 457, 3200, table B1 |
| `desi_des_dwarfs2025/` | DES and DESI dwarf lensing (Treiber et al. 2025) |
| `li2020_sparc_halos/` | halo fits to SPARC (Li et al. 2020, arXiv:2001.10538) |
| `velander2014/` | CFHTLenS (Velander et al. 2014) |
| SPARC directory | SPARC and the BTFR tables of Lelli et al. 2016, 2019 (astroweb.cwru.edu/SPARC) |

## Papers

| Text | DOI |
|---|---|
| Volume-preserving relabelling and a label-count classification of cosmic media (paper1, version 4) | [10.5281/zenodo.24021766](https://doi.org/10.5281/zenodo.24021766) |
| Rotation curves as the shape of a label space (paper3, version 3) | [10.5281/zenodo.23871775](https://doi.org/10.5281/zenodo.23871775) |
| The cosmological constant as a label (paper5, version 3) | [10.5281/zenodo.24021767](https://doi.org/10.5281/zenodo.24021767) |
| What the label framework predicts: a catalogue of tests across physics (paper7) | [10.5281/zenodo.23871778](https://doi.org/10.5281/zenodo.23871778) |
| The volume axis, horizon equilibrium and spin as a relation (paper8) | [10.5281/zenodo.23871780](https://doi.org/10.5281/zenodo.23871780) |
| A dimensionless theorem base in geometric algebra (theorem base, version 2) | [10.5281/zenodo.24023762](https://doi.org/10.5281/zenodo.24023762) |
| The vacuum in the label framework (paper10, version 2) | [10.5281/zenodo.24023150](https://doi.org/10.5281/zenodo.24023150) |
| Black holes in the label framework (paper11, version 2) | [10.5281/zenodo.24023151](https://doi.org/10.5281/zenodo.24023151) |
| The language of labels in physics: an overview of a series (overview, version 2) | [10.5281/zenodo.24021765](https://doi.org/10.5281/zenodo.24021765) |

This code: [10.5281/zenodo.24021768](https://doi.org/10.5281/zenodo.24021768) (release v1.1); all versions: [10.5281/zenodo.23871789](https://doi.org/10.5281/zenodo.23871789); v1.0: [10.5281/zenodo.23871790](https://doi.org/10.5281/zenodo.23871790).

## Citing

Cite the paper whose result you use (DOIs above) and, for the checks, this code:

```
V. Loginov, Code and checks for the label-framework series, version 1.1, Zenodo (2026), doi:10.5281/zenodo.24021768.
```

## Licence

MIT (see `LICENSE`).
