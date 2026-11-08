# paper14: checks

Scripts that compute the statements of *Continuum Mechanics in the Label Framework: Fluids, Liquid Crystals, Solids and Superfluids by
the Symmetry of Their Labels* (doi:10.5281/zenodo.24060747); all print `PASS`.

- `ga_cont_media_check` (CM1–CM8): groups of label maps (SDiff, smectic, columnar, ISO(3), cubic), their invariants and reflections;
  stress; sound modes by direction; zero-frequency modes as local relabellings; smectic second sound; fluid and superfluid sound;
  a shear relabelling changes the solid action; layer stress and undulation
- `ga_cont_elastic_check` (EL1–EL5): numbers of elastic moduli by point group and for label media; isotropic stability and CRC sound
  speeds; Born stability and Kittel stiffnesses of cubic crystals; the Cauchy relation
- `ga_cont_superfluid_check` (SF1–SF4): Euler allows any circulation; integer winding on a circle; h/m against Rayfield–Reif; vortex
  lattice and rigid rotation
- `ga_cont_series_check` (CX1–CX3): forms and numbers independent of the values of moduli; the solid as mu -> 0; Son's superfluid as one
  timelike label
