# Code and checks for the label-framework series

The scripts in this repository compute or check the statements of eight texts:

- Volume-preserving relabelling and a label-count classification of cosmic media (paper1, version 3)
- Rotation curves as the shape of a label space (paper3, version 3)
- The cosmological constant as a label (paper5, version 2)
- What the label framework predicts: a catalogue of tests across physics (paper7, version 2)
- The volume axis, horizon equilibrium and spin as a relation (paper8)
- Where a number stops being a label (paper9)
- A dimensionless theorem base in geometric algebra (theorem base)
- The language of labels in physics: an overview of a series

Each paper names, next to a statement, the script and the label of the check (for example `ga_rindler_eta_check` RE3).
The repository holds only those scripts, the scripts they call, their stored outputs and one small table of our own;
the papers themselves are on Zenodo (DOIs below).

## Layout

| Files | Content |
|---|---|
| `ga_*.py`, `inv_*.py`, `pub_*.py`, `paper2_cs_check.py` | symbolic and numerical checks without external data; each prints labelled `PASS`/`FAIL` lines |
| `de_*.py`, `btfr_*.py`, `rar_*.py`, `mond_*.py`, `csdr_*.py` | checks against published data (the data directory is a command-line argument) |
| `ga_core.py` | the geometric-algebra helpers used by the checks |
| `run_all.py` | runs every data-free check |
| `theorem_base_check.py` | runs the 100 theorems of the theorem base and the falsification pass and asserts that every listed label prints `PASS` |
| `figures/fig_paper*.py` | draw the figures of the papers (byte for byte reproducible) |
| `*.out`, `*.txt`, `results/` | stored outputs read by some checks (300 realisations of the fluctuating Λ, entanglement sums) |
| `data/a0z_points.csv` | the a₀(z) points used by the clock checks, transcribed from the cited papers |

## Running

Python 3.11 or later with `numpy`, `scipy`, `sympy` and `matplotlib`.

```
python run_all.py              # every data-free check; prints a TOTAL line with 0 FAIL
python theorem_base_check.py   # the theorem base: 100 theorems, 475 labelled checks
```

`ga_rindler_eta_check.py` takes about nine minutes; the rest of `run_all.py` a few minutes.

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

DOIs are added when the records are published.

## Licence

MIT (see `LICENSE`).
