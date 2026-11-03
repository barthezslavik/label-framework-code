# Theorem base: theorem → check

Each theorem of *A Dimensionless Theorem Base in Geometric Algebra* (doi:10.5281/zenodo.23871786) and the scripts and labels
that check it. `python theorem_base_check.py` runs them all and asserts that every listed label prints `PASS`.

| Section | Theorem | Check |
|---|---|---|
| Matter from relabelling | Three labels | `ga_selection_check` E9, E10, E11 |
| Matter from relabelling | Identical conservation | `ga_selection_check` E12; `ga_dark_sector_check` K6 |
| Matter from relabelling | Equation of state | `pub_paper1_check` Z2b; `ga_phase_space_check` K1c; `ga_dark_sector_check` K3 |
| Matter from relabelling | Dust is additivity | `ga_phase_space_check` K1a, K1c |
| Matter from relabelling | Isotropy and slip | `pub_paper1_check` Z3; `ga_multistream_check` M3, M4 |
| Matter from relabelling | Dilution exponent | `ga_einstein_check` J3, J3b |
| Matter from relabelling | Liouville | `ga_phase_space_check` K4a; `ga_dark_sector_check` K4 |
| Matter from relabelling | SDiff as a rotor field | `ga_sdiff_rotor_check` SR1, SR2, SR3, SR4, SR5, SR6, SR6b, SR7 |
| Selection of field equations | Duality circle | `ga_selection_check` E2, E3 |
| Selection of field equations | Maxwell collapse | `ga_selection_check` E5, E13 |
| Selection of field equations | Middle-grade duality | `ga_conformal_check` G2, G6 |
| Selection of field equations | Einstein form | `ga_einstein_check` J1, J2 |
| Selection of field equations | $\Lambda$ is a label | `ga_einstein_check` J5; `ga_dark_sector_check` K2, K2' |
| Selection of field equations | Vacuum $w=-1$ | `ga_dark_sector_check` K1, K1' |
| Conformal de Sitter space | Point = sphere | `ga_theta_conformal_check` C2 |
| Conformal de Sitter space | Conformal weights | `ga_theta_conformal_check` C1; `ga_symmetry_breaking_check` B1, B2 |
| Conformal de Sitter space | Cosmic time is a dilator | `ga_cosmic_dilator_check` M1, M2, M3 |
| Conformal de Sitter space | Unruh–de Sitter | `inv_misc_check` M4; `ga_theta_sphere_check` Q2; `ga_theta_duality_check` D2, D3, D4 |
| The scaling group | Group | `ga_vp_scaling_check` V1, V2; `ga_scaling_group_check` S1, S2, S3, S4, S5 |
| The scaling group | Fixed points and orbits | `ga_scaling_fixed_points_check` G1, G2, G3; `ga_vp_scaling_check` V3, V4 |
| The scaling group | Lynden-Bell | `ga_sdiff_rotation_check` R2, R3, R4, R5, R6 |
| The scaling group | Kelvin | `ga_sdiff_rotation_check` R1 |
| The scaling group | Mass weights | `ga_symmetry_links_check` LK1, LK2 |
| The scaling group | A subgroup writes the law, it does not select it | `ga_symmetry_map_check` SM1, SM2, SM3; `ga_symmetry_map_falsify_check` RF1, RF2, RF3 |
| The scaling group | Hodge with sources | `ga_symmetry_links_check` LK3 |
| Units, partitions and time | $GL=SL\times K$ through $I$ | `ga_sl_k_hodge_check` HK1, HK2, HK3, HK5, HK6; `pub_paper2_falsify_check` PF1 |
| Units, partitions and time | Two faces of $I_5$ | `ga_deep_units_check` U1, U2, U3; `pub_chiral_check` C3 |
| Units, partitions and time | Middles are null | `ga_deep_units_check` U4, U5 |
| Units, partitions and time | Partitions of unity | `ga_mu_rotor_check` F1–F5; `ga_duality_test_check` T4; `mond_solar_q2_check` K1–K6 (Hees et al. 2014, 2016; Park et al. 2026) |
| Units, partitions and time | Three kinds of time | `ga_time_dilation_check` T1, T2, T3, T4, T5, T6, T7 |
| Units, partitions and time | Time, strictly | `ga_time_strict_check` TS1, TS2, TS3, TS4; `pub_paper2_falsify_check` PF2 |
| Units, partitions and time | Self-similarity | `ga_selfsimilar_check` SS1, SS2, SS3, SS4, SS5, SS6; `pub_paper2_falsify_check` PF3 |
| Units, partitions and time | No scale from symmetry | `ga_theta_conformal_check` C6; `ga_scaling_fixed_points_check` G4; `ga_theta_mond_check` M4 |
| Dimensionless observables | Redshift and time dilation | `inv_doppler_check` G1, G2, G3, G4 |
| Dimensionless observables | Charge of label matter | `ga_dark_sector_check` K7, K7'; `ga_label_count_quantum_check` QL7 |
| Dimensionless observables | Conserved ratios | `ga_dark_sector_check` K8, K5 |
| Dimensionless observables | True dualities | `ga_duality_test_check` T1, T2, T3 |
| Principles that set numbers (second layer) | Four kinds of principle | `inv_number_principles_check` N1, N2, N3, N4, N5 |
| Principles that set numbers (second layer) | Discrete labels and $\Lambda$ | `ga_discrete_lambda_check` DL1–DL5; `de_everpresent_check` EV1, EV2; `de_phantom_compare_check` PC1; `de_h3_discrete_s8_check` HD1 |
| Principles that set numbers (second layer) | Two native phantom mechanisms | `ga_growth_coupling_check` GC2; `ga_wbar_definition_check` WD1–WD4; `de_phantom_compare_check` PC1, PC2; `de_phantom_bins_check` PB1, PB2; `de_h3_s8_fit_check` HS3–HS5; `de_h3_discrete_s8_check` HD1, HD4 |
| Principles that set numbers (second layer) | Compact charges are counts | `ga_compact_integers_check` CI1, CI2, CI3, CI4, CI5, CI6, CI7, CI8 |
| Principles that set numbers (second layer) | Attractors and zero modes | `ga_attractor_check` AT1, AT2, AT3, AT4 |
| Principles that set numbers (second layer) | Catalogue of structural numbers | `ga_number_catalog_check` NC1–NC3, NC5–NC12; `de_btfr_running_check` BR1–BR3 |
| Principles that set numbers (second layer) | Label budget | `ga_label_budget_check` LB1, LB2, LB3 |
| Principles that set numbers (second layer) | The level $a_0$ and the baryon share | `ga_kappa_check` KA1–KA8; `de_p1p2_fits_check` PF4 |
| Principles that set numbers (second layer) | Formation epoch of the level | `ga_formation_epoch_check` FE1, FE2, FE3, FE5; `de_halo_share_check` HS2, HS3, HS7 |
| Principles that set numbers (second layer) | Clock combinations | `inv_clock_combos_check` P1, P2, P3, P4, P5, P6, P7 |
| Principles that set numbers (second layer) | Clock covariance | `ga_a0_clock_check` Q1, Q2, Q3, Q4 |
| Symmetric points of the parameters | Constants as licences | `inv_all_numbers_check` ND1, ND2, ND3 |
| Symmetric points of the parameters | Symmetric points | `inv_all_numbers_check` ND4, ND5; `inv_numbers_full_check` NF1, NF2, NF3, NF5 |
| Symmetric points of the parameters | Stability of symmetric points | `inv_numbers_full_check` NF4; `inv_numbers_falsify_check` XN2; `ga_pure_universe_falsify_check` PU2, PU3 |
| Symmetric points of the parameters | Pure GA universe | `inv_numbers_falsify_check` XN1; `inv_all_numbers_check` ND7; `ga_pure_universe_falsify_check` PU1 |
| The map: one algebra, one split, chosen vectors | The split by the geometric product | `ga_conformal_split_check` CS1, CS2, CS3, CS4, CS5, CS6 |
| The map: one algebra, one split, chosen vectors | Rigidity needs $\Lambda$ | `ga_rehab_hypotheses_check` RH12; `ga_all_falsify_check` AF2 |
| The map: one algebra, one split, chosen vectors | Phase and hand | `ga_all_falsify_check` AF1; `ga_rehab_hypotheses_check` RH11 |
| The map: one algebra, one split, chosen vectors | Volume rotors | `ga_sphere_de_check` VS1, VS2, VS3 |
| The map: one algebra, one split, chosen vectors | $\Lambda$, two views | `ga_sphere_de_check` VS4, VS4a; `ga_sdiff_de_check` SD1 |
| The map: one algebra, one split, chosen vectors | Dark energy from relabelling media | `ga_sdiff_de_check` SD2, SD3, SD4, SD5 |
| The map: one algebra, one split, chosen vectors | Entropy and the split | `ga_rehab_hypotheses_check` RH9; `ga_entropy_falsify_check` EN1, EN2, EN3 |
| The map: one algebra, one split, chosen vectors | Time as a dilation angle | `inv_hubble_norm_check` HN1, HN2 |
| The map: one algebra, one split, chosen vectors | Zeros and flows | `ga_rehab_hypotheses_check` RH16, RH7; `inv_flow_weights_check` FW1, FW3; `inv_flow_weights_falsify_check` XW1, XW4 |
| The map: one algebra, one split, chosen vectors | Shear is not dark energy | `ga_shear_de_a0_check` SH1, SH2, SH3, SH7 |
| The map: one algebra, one split, chosen vectors | Internal vector choice | `ga_rehab_hypotheses_check` RH13 |
| Label media in two signatures, galaxies and the dark sector | Stress of a label medium | `ga_label_ladder_check` LL7; `ga_timelike_rungs_check` TR1 |
| Label media in two signatures, galaxies and the dark sector | Two-dimensional ladder | `ga_label_ladder_check` LL1, LL2, LL3, LL6; `ga_timelike_rungs_check` TR2, TR4 |
| Label media in two signatures, galaxies and the dark sector | Dilation-blind media | `ga_radiation_k_check` RK1, RK2; `ga_timelike_rungs_check` TR3 |
| Label media in two signatures, galaxies and the dark sector | Null energy condition | `ga_timelike_ladder_check` TL1, TL7 |
| Label media in two signatures, galaxies and the dark sector | Lowest-order dust–scalar coupling | `ga_timelike_ladder_check` TL5; `ga_coupling_consequences_check` CC1, CC3, CC4 |
| Label media in two signatures, galaxies and the dark sector | Rotation curves as label space | `ga_flat_curve_label_check` FC11, FC15; `ga_label_factor_check` LF1, LF2, LF3, LF4 |
| Label media in two signatures, galaxies and the dark sector | One isothermal state | `ga_isothermal_label_check` IS1, IS2, IS3, IS4 |
| Label media in two signatures, galaxies and the dark sector | Population slopes from the spectrum | `ga_flat_curve_label_check` FC12; `ga_p2_spectrum_check` HS1, HS2, HS3; `de_core_scaling_check`; `de_p1p2_fits_check` PF1–PF3; `de_li2020_check` |
| Label media in two signatures, galaxies and the dark sector | The angular rung alone | `ga_flat_curve_label_check` FC8, FC9, FC10 |
| Label media in two signatures, galaxies and the dark sector | Growth with the coupling | `ga_growth_coupling_check` GC1, GC2, GC3 |
| Label media in two signatures, galaxies and the dark sector | Growth and the common baryon share | `ga_label_growth_check` LG1, LG3; `ga_p1_check` PA1, PA6 |
| Label media in two signatures, galaxies and the dark sector | Galaxy cores and a self-interaction label | `ga_si_scales_check` SS1–SS3; `de_phase_density_check`; `de_self_interaction_check` SI, SI3, SI4 |
| Label media in two signatures, galaxies and the dark sector | Population law at the halo radius | `ga_kappa_check` KA1; `de_share_growth_check` SG1–SG9; `de_hsc_shmr_check` VH1–VH5; `de_vflat_v200_check` VR1–VR6; `de_dwarf_lensing_check` DL1–DL5 |
| Label media in two signatures, galaxies and the dark sector | Evolution of the level | `ga_a0_evolution_check` AE1, AE2, AE3, AE4 |
| Label media in two signatures, galaxies and the dark sector | Ladder of preserved structures | `ga_ladder_check` LD1, LD2, LD3, LD4, LD5 |
| Label media in two signatures, galaxies and the dark sector | Which media can induce gravity | `ga_induced_gravity_check` IG1, IG2, IG3 |
| Label media in two signatures, galaxies and the dark sector | Depth of the Maxwell selection | `ga_em_selection_check` EM1, EM3, EM5, EM5b, EM7, EM8 |
| Label media in two signatures, galaxies and the dark sector | Geometry is an independent axiom | `ga_matter_geometry_check` MG1, MG3 |
| Label media in two signatures, galaxies and the dark sector | Horizon equilibrium gives the trace-free equation | `ga_jacobson_check` JC1, JC2, JC3, JC4, JC5, JC5c, JC6, JC7, JC8 |
| Label media in two signatures, galaxies and the dark sector | $G$ from one extra dimension | `ga_g_curvature_check` GX1, GX1b, GX2, GX2b, GX2c, GX3, GX3b, GX3c |
| Predictions across physics | $S_8$ line of the coupling | `ga_s8_wbar_forecast_check` SW1, SW3, SW7, SW8 |
| Predictions across physics | Level evolution with the spectrum epoch | `ga_a0_host_epoch_check` HE1, HE2, HE2b, HE3 |
| Predictions across physics | Dust and baryons fall differently | `ga_dust_baryon_ep_check` EP2, EP6 |
| Predictions across physics | Cores under the self-interaction axiom | `ga_si_cluster_cores_check` SC1, SC4, SC5; `ga_f5_cluster_cores_check` FX1, FX2 |
| Predictions across physics | Crater II needs tidal mass loss | `ga_crater2_tides_check` CR1, CR2, CR3, CR4, CR5 |
| Predictions across physics | Couplings of label matter to the Standard Model | `ga_label_sm_couplings_check` LC1, LC2, LC3 |
| Predictions across physics | Label count at the quantum level | `ga_label_count_quantum_check` QL1, QL2, QL3, QL4, QL5, QL7, QL8 |
| Predictions across physics | No thermal history of the label number | `ga_thermal_nuclear_check` TN1, TN3, TN4; `ga_llr_mapping_check` LL3 |
| Predictions across physics | Lunar ranging and the baryon coupling | `ga_llr_mapping_check` LL1, LL2 |
| Predictions across physics | Compact integers in condensed matter | `ga_solid_state_compact_check` SS1, SS2 |
| Foundations, fermions and ordinary matter | Indistinguishable label elements | `ga_fermion_relation_check` FR1, FR3, FR4, FR5, FR6, FR7 |
| Foundations, fermions and ordinary matter | Label number as a degree, $J=I$ | `ga_label_soliton_spin_check` LS1, LS2, LS2b, LS3, LS4, LS5 |
| Foundations, fermions and ordinary matter | Compact labels replace no axiom | `ga_compact_labels_check` CL1, CL2, CL3, CL4, CL4b, CL5, CL6 |
| Foundations, fermions and ordinary matter | Signature from the label medium | `ga_signature_from_labels_check` SG1, SG2, SG3, SG4, SG4b, SG4c, SG5, SG6 |
| Foundations, fermions and ordinary matter | Vacuum, orientation and the quantum level | `ga_postulates_reduction_check` VC1, VC1b, VC2, VC3, OR1, OR2, OR3, OR4, QS1, QS2 |
| Foundations, fermions and ordinary matter | Plane and sphere share the area coefficient within the fits | `ga_rindler_eta_check` RE1, RE2, RE3, RE4, RE5, RE6, RE7, RE8 |
| Foundations, fermions and ordinary matter | Internal group of a rotor field | `ga_ordinary_matter_check` OM1, OM2, OM3, OM4, OM5, OM6, OM7, OM8a, OM8, OM9 |

## Falsification pass

| Pass | Check |
|---|---|
| theorem by theorem | `ga_theorem_falsify_check` XA1–XF3 |
| first | `ga_symmetry_map_falsify_check` RF4–RF7, `inv_numbers_falsify_check` XN1–XN5 |
| second | `ga_pure_universe_falsify_check` PU1–PU6, `btfr_selection_bias_check` PU7 |
| third | `ga_all_falsify_check` AF1–AF8 |
| fourth | `ga_rehab_hypotheses_check`, `ga_shear_de_a0_check`, `ga_sphere_de_check` |
| fifth | `ga_scaling_duality_check` DU7, `ga_angmom_radius_check` AM3, coupling CC1, stable scalar TL7 |
