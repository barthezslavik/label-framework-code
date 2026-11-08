# Machine check of the GA theorem base (papers/restart/44-theorem-base.md): every theorem lists (script, labels); this runs each script once
# and asserts that every listed label printed PASS. Data-dependent scripts (rar_*) are not part of the base; observations enter only through
# published dimensionless numbers in the inv_* scripts.
# Run: uv run --with sympy --with numpy --with scipy python theorem_base_check.py
import os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = {
    # A. matter from relabelling (SDiff)
    'A1 three labels':            {'ga_selection_check.py': ['E9', 'E10', 'E11']},
    'A2 identical conservation':  {'ga_selection_check.py': ['E12'], 'ga_dark_sector_check.py': ['K6']},
    'A3 equation of state':       {'pub_paper1_check.py': ['Z2b'], 'ga_phase_space_check.py': ['K1c'], 'ga_dark_sector_check.py': ['K3']},
    'A4 dust = reparametrisation':{'ga_phase_space_check.py': ['K1a', 'K1c']},
    'A5 isotropy and slip':       {'pub_paper1_check.py': ['Z3'], 'ga_multistream_check.py': ['M3', 'M4']},
    'A6 dilution exponent 3':     {'ga_einstein_check.py': ['J3', 'J3b']},
    'A7 Liouville':               {'ga_phase_space_check.py': ['K4a'], 'ga_dark_sector_check.py': ['K4']},
    # B. selection of field equations
    'B1 duality circle':          {'ga_selection_check.py': ['E2', 'E3']},
    'B2 Maxwell collapse':        {'ga_selection_check.py': ['E5', 'E13']},
    'B3 middle-grade duality':    {'ga_conformal_check.py': ['G2', 'G6']},
    'B4 Einstein form':           {'ga_einstein_check.py': ['J1', 'J2']},
    'B5 Lambda as a label':       {'ga_einstein_check.py': ['J5'], 'ga_dark_sector_check.py': ['K2', "K2'"]},
    'B6 vacuum w = -1':           {'ga_dark_sector_check.py': ['K1', "K1'"]},
    # C. conformal de Sitter
    'C1 point = sphere':          {'ga_theta_conformal_check.py': ['C2']},
    'C2 conformal weights':       {'ga_theta_conformal_check.py': ['C1'], 'ga_symmetry_breaking_check.py': ['B1', 'B2']},
    'C3 cosmic time = dilator':   {'ga_cosmic_dilator_check.py': ['M1', 'M2', 'M3']},
    'C4 Unruh-de Sitter':         {'inv_misc_check.py': ['M4'], 'ga_theta_sphere_check.py': ['Q2'], 'ga_theta_duality_check.py': ['D2', 'D3', 'D4']},
    # D. scaling group
    'D1 scaling group':           {'ga_vp_scaling_check.py': ['V1', 'V2'], 'ga_scaling_group_check.py': ['S1', 'S2', 'S3', 'S4', 'S5']},
    'D2 fixed points and orbits': {'ga_scaling_fixed_points_check.py': ['G1', 'G2', 'G3'], 'ga_vp_scaling_check.py': ['V3', 'V4']},
    'D3 Lynden-Bell':             {'ga_sdiff_rotation_check.py': ['R2', 'R3', 'R4', 'R5', 'R6']},
    'D4 Kelvin minimum':          {'ga_sdiff_rotation_check.py': ['R1']},
    'E6 SL x K through I':        {'ga_sl_k_hodge_check.py': ['HK1', 'HK2', 'HK3', 'HK5', 'HK6'], 'pub_paper2_falsify_check.py': ['PF1']},
    'A8 SDiff as rotor field':    {'ga_sdiff_rotor_check.py': ['SR1', 'SR2', 'SR3', 'SR4', 'SR5', 'SR6', 'SR6b', 'SR7']},
    # E. units, partitions, time
    'E1 two faces of I5':         {'ga_deep_units_check.py': ['U1', 'U2', 'U3'], 'pub_chiral_check.py': ['C3']},
    'E2 middles are null':        {'ga_deep_units_check.py': ['U4', 'U5']},
    'E3 partitions of unity':     {'ga_mu_rotor_check.py': ['F1', 'F2', 'F3', 'F4', 'F5'], 'ga_duality_test_check.py': ['T4']},
    'E4 three kinds of time':     {'ga_time_dilation_check.py': ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7']},
    'E4c self-similarity':        {'ga_selfsimilar_check.py': ['SS1', 'SS2', 'SS3', 'SS4', 'SS5', 'SS6'], 'pub_paper2_falsify_check.py': ['PF3']},
    'E4b time strictly':          {'ga_time_strict_check.py': ['TS1', 'TS2', 'TS3', 'TS4'], 'pub_paper2_falsify_check.py': ['PF2']},
    'E5 no scale from symmetry':  {'ga_theta_conformal_check.py': ['C6'], 'ga_scaling_fixed_points_check.py': ['G4'], 'ga_theta_mond_check.py': ['M4']},
    # F. dimensionless observables
    'F1 redshift and time dilation': {'inv_doppler_check.py': ['G1', 'G2', 'G3', 'G4']},
    'F2 charge of label matter': {'ga_dark_sector_check.py': ['K7', "K7'"], 'ga_label_count_quantum_check.py': ['QL7']},
    'F3 conserved ratios':        {'ga_dark_sector_check.py': ['K8', 'K5']},
    'F4 true dualities':          {'ga_duality_test_check.py': ['T1', 'T2', 'T3']},
    # H. principles that set numbers (second layer)
    'H number principles':        {'inv_number_principles_check.py': ['N1', 'N2', 'N3', 'N4', 'N5']},
    'H2 discrete labels and Lambda': {'ga_discrete_lambda_check.py': ['DL1', 'DL2', 'DL3', 'DL4', 'DL5']},
    'H3 compact charges are counts': {'ga_compact_integers_check.py': ['CI1', 'CI2', 'CI3', 'CI4', 'CI5', 'CI6', 'CI7', 'CI8']},
    'H4 attractors and zero modes': {'ga_attractor_check.py': ['AT1', 'AT2', 'AT3', 'AT4']},
    'H5 catalogue of structural numbers': {'ga_number_catalog_check.py': ['NC1', 'NC2', 'NC3', 'NC5', 'NC6', 'NC7', 'NC8', 'NC9', 'NC10', 'NC11', 'NC12']},
    'H6 label budget':            {'ga_label_budget_check.py': ['LB1', 'LB2', 'LB3']},
    'H7 level a0 and the baryon share': {'ga_kappa_check.py': ['KA1', 'KA2', 'KA3', 'KA4', 'KA5', 'KA6', 'KA7', 'KA8']},
    'H8 two phantom mechanisms':  {'ga_growth_coupling_check.py': ['GC2'], 'ga_wbar_definition_check.py': ['WD1', 'WD2', 'WD3', 'WD4']},
    'H9 formation epoch of the level': {'ga_formation_epoch_check.py': ['FE1', 'FE2', 'FE3', 'FE5']},
    'H2 clock combinations':      {'inv_clock_combos_check.py': ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']},
    'H3 why a0 ~ 1/t':            {'ga_a0_clock_check.py': ['Q1', 'Q2', 'Q3', 'Q4']},
    # I. numbers of the universe and the symmetry web (papers/restart/51-54)
    'I1 constants as licences':   {'inv_all_numbers_check.py': ['ND1', 'ND2', 'ND3']},
    'I2 symmetric points':        {'inv_all_numbers_check.py': ['ND4', 'ND5'], 'inv_numbers_full_check.py': ['NF1', 'NF2', 'NF3', 'NF5']},
    'I3 stability of symmetric points': {'inv_numbers_full_check.py': ['NF4'], 'inv_numbers_falsify_check.py': ['XN2'], 'ga_pure_universe_falsify_check.py': ['PU2', 'PU3']},
    'I4 pure GA universe':        {'inv_numbers_falsify_check.py': ['XN1'], 'inv_all_numbers_check.py': ['ND7'], 'ga_pure_universe_falsify_check.py': ['PU1']},
    'I5 mass weights of the stretch group': {'ga_symmetry_links_check.py': ['LK1', 'LK2']},
    'I6 subgroup writes, does not select': {'ga_symmetry_map_check.py': ['SM1', 'SM2', 'SM3'], 'ga_symmetry_map_falsify_check.py': ['RF1', 'RF2', 'RF3']},
    'I7 Hodge with sources':      {'ga_symmetry_links_check.py': ['LK3']},
    # J. the map: one algebra, one split, chosen vectors (papers/restart/59-65)
    'J1 rigidity needs Lambda':   {'ga_rehab_hypotheses_check.py': ['RH12'], 'ga_all_falsify_check.py': ['AF2']},
    'J2 phase and hand':          {'ga_all_falsify_check.py': ['AF1'], 'ga_rehab_hypotheses_check.py': ['RH11']},
    'J3 volume rotors':           {'ga_sphere_de_check.py': ['VS1', 'VS2', 'VS3']},
    'J4 Lambda two views':        {'ga_sphere_de_check.py': ['VS4', 'VS4a'], 'ga_sdiff_de_check.py': ['SD1']},
    'J5 DE from relabelling media': {'ga_sdiff_de_check.py': ['SD2', 'SD3', 'SD4', 'SD5']},
    'J6 entropy and the split':   {'ga_rehab_hypotheses_check.py': ['RH9'], 'ga_entropy_falsify_check.py': ['EN1', 'EN2', 'EN3']},
    'J7 time = dilation angle':   {'inv_hubble_norm_check.py': ['HN1', 'HN2']},
    'J8 zeros and flows':         {'ga_rehab_hypotheses_check.py': ['RH16', 'RH7'], 'inv_flow_weights_check.py': ['FW1', 'FW3'],
                                   'inv_flow_weights_falsify_check.py': ['XW1', 'XW4']},
    'J9 shear is not dark energy': {'ga_shear_de_a0_check.py': ['SH1', 'SH2', 'SH3', 'SH7']},
    'J10 internal vector choice': {'ga_rehab_hypotheses_check.py': ['RH13']},
    # K. label media in two signatures, galaxies and the dark sector (docs 68-87)
    'K1 stress of a label medium':  {'ga_label_ladder_check.py': ['LL7'], 'ga_timelike_rungs_check.py': ['TR1']},
    'K2 two-dimensional ladder':    {'ga_label_ladder_check.py': ['LL1', 'LL2', 'LL3', 'LL6'], 'ga_timelike_rungs_check.py': ['TR2', 'TR4']},
    'K3 dilation-blind media':      {'ga_radiation_k_check.py': ['RK1', 'RK2'], 'ga_timelike_rungs_check.py': ['TR3']},
    'K4 null energy condition':     {'ga_timelike_ladder_check.py': ['TL1', 'TL7']},
    'K5 dust-scalar coupling':      {'ga_timelike_ladder_check.py': ['TL5'], 'ga_coupling_consequences_check.py': ['CC1', 'CC3', 'CC4']},
    'K6 rotation curves as label space': {'ga_flat_curve_label_check.py': ['FC11', 'FC15'], 'ga_label_factor_check.py': ['LF1', 'LF2', 'LF3', 'LF4']},
    'K7 one isothermal state':      {'ga_isothermal_label_check.py': ['IS1', 'IS2', 'IS3', 'IS4']},
    'K8 population slopes from the spectrum': {'ga_flat_curve_label_check.py': ['FC12'], 'ga_p2_spectrum_check.py': ['HS1', 'HS2', 'HS3']},
    'K9 angular rung alone':        {'ga_flat_curve_label_check.py': ['FC8', 'FC9', 'FC10']},
    'K10 growth with the coupling': {'ga_growth_coupling_check.py': ['GC1', 'GC2', 'GC3']},
    'K16 galaxy cores need a new label': {'ga_si_scales_check.py': ['SS1', 'SS2', 'SS3']},
    'K17 population law at the halo radius': {'ga_kappa_check.py': ['KA1']},
    'K18 evolution of the level': {'ga_a0_evolution_check.py': ['AE1', 'AE2', 'AE3', 'AE4']},
    'K12 growth and the common baryon share': {'ga_label_growth_check.py': ['LG1', 'LG3'], 'ga_p1_check.py': ['PA1', 'PA6']},
    'K13 ladder of preserved structures': {'ga_ladder_check.py': ['LD1', 'LD2', 'LD3', 'LD4', 'LD5']},
    'K14 which media can induce gravity': {'ga_induced_gravity_check.py': ['IG1', 'IG2', 'IG3']},
    'K15 depth of the Maxwell selection': {'ga_em_selection_check.py': ['EM1', 'EM3', 'EM5', 'EM5b', 'EM7', 'EM8']},
    'K11 geometry is an independent axiom': {'ga_matter_geometry_check.py': ['MG1', 'MG3']},
    'K17 horizon equilibrium gives the trace-free equation': {'ga_jacobson_check.py': ['JC1', 'JC2', 'JC3', 'JC4', 'JC5', 'JC5c', 'JC6', 'JC7', 'JC8']},
    'K19 G from one extra dimension': {'ga_g_curvature_check.py': ['GX1', 'GX1b', 'GX2', 'GX2b', 'GX2c', 'GX3', 'GX3b', 'GX3c']},
    'K20 conformal generators by the geometric product': {'ga_conformal_split_check.py': ['CS1', 'CS2', 'CS3', 'CS4', 'CS5', 'CS6']},
    'K21 indistinguishability of label elements': {'ga_fermion_relation_check.py': ['FR1', 'FR3', 'FR4', 'FR5', 'FR6', 'FR7']},
    'K22 label number as a degree, J = I': {'ga_label_soliton_spin_check.py': ['LS1', 'LS2', 'LS2b', 'LS3', 'LS4', 'LS5']},
    'K23 compact labels do not replace D or self-interaction': {'ga_compact_labels_check.py': ['CL1', 'CL2', 'CL3', 'CL4', 'CL4b', 'CL5', 'CL6']},
    'K24 signature (1,3) from the label medium': {'ga_signature_from_labels_check.py': ['SG1', 'SG2', 'SG3', 'SG4', 'SG4b', 'SG4c', 'SG5', 'SG6']},
    'K25 vacuum, orientation, quantum level': {'ga_postulates_reduction_check.py': ['VC1', 'VC1b', 'VC2', 'VC3', 'OR1', 'OR2', 'OR3', 'OR4', 'QS1', 'QS2']},
    'K26 plane and sphere share the area coefficient within the fits': {'ga_rindler_eta_check.py': ['RE1', 'RE2', 'RE3', 'RE4', 'RE5', 'RE6', 'RE7', 'RE8']},
    'K27 internal group of a rotor field is U(2)': {'ga_ordinary_matter_check.py': ['OM1', 'OM2', 'OM3', 'OM4', 'OM5', 'OM6', 'OM7', 'OM8a', 'OM8', 'OM9']},
    # L. predictions across physics (docs 116-124)
    'L1 S8 line of the coupling': {'ga_s8_wbar_forecast_check.py': ['SW1', 'SW2', 'SW3', 'SW4', 'SW6', 'SW7', 'SW8']},
    'L2 level evolution with the spectrum epoch': {'ga_a0_host_epoch_check.py': ['HE1', 'HE2', 'HE2b', 'HE3']},
    'L3 dust and baryons fall differently': {'ga_dust_baryon_ep_check.py': ['EP1', 'EP2', 'EP3', 'EP4', 'EP6']},
    'L4 cores under the self-interaction axiom': {'ga_si_cluster_cores_check.py': ['SC1', 'SC4', 'SC5'], 'ga_f5_cluster_cores_check.py': ['FX1', 'FX2']},
    'L7b lunar ranging and the baryon coupling': {'ga_llr_mapping_check.py': ['LL1', 'LL2']},
    'L5 Crater II needs tidal mass loss': {'ga_crater2_tides_check.py': ['CR1', 'CR2', 'CR3', 'CR4', 'CR5']},
    'L6 couplings of label matter to the SM': {'ga_label_sm_couplings_check.py': ['LC1', 'LC2', 'LC3']},
    'L6b label count at the quantum level': {'ga_label_count_quantum_check.py': ['QL1', 'QL2', 'QL3', 'QL4', 'QL5', 'QL8']},
    'L7 no thermal history of the label number': {'ga_thermal_nuclear_check.py': ['TN1', 'TN1b', 'TN3', 'TN4']},
    'L8 compact integers in condensed matter': {'ga_solid_state_compact_check.py': ['SS1', 'SS2']},
    # P12. companion checks of paper12, quantum mechanics in the label framework (doc 145); not counted among the theorems of the base text
    'P12 Schrodinger field of a label medium': {'ga_qm_labels_check.py': ['QM1', 'QM2', 'QM3', 'QM7']},
    'P12 linearity is the pressureless point': {'ga_qm_labels_check.py': ['QM4'], 'ga_qm_modifications_check.py': ['QD5']},
    'P12 vorticity and the circle are postulated': {'ga_qm_labels_check.py': ['QM5', 'QM6']},
    'P12 labels in space carry no entanglement': {'ga_qm_labels_check.py': ['QM8'], 'ga_qm_modifications_check.py': ['QD4']},
    'P12 verdicts on testable modifications': {'ga_qm_modifications_check.py': ['QD1', 'QD2', 'QD3']},
    'P12 half-integer spin under the sign premise': {'ga_qm_modifications_check.py': ['HH1']},
    'P12 statuses of the twelve interpretations': {'ga_qm_interpretations_check.py': ['QI1', 'QI2', 'QI3', 'QI4', 'QI5', 'QI6']},
    # P12 (continued). many bodies, statistics and internal quantum numbers of paper12 (docs 146-150); not counted among the theorems
    'P12 unimodularity gives the gauge algebra and hypercharges': {'ga_internal_check.py': ['IN1', 'IN2', 'IN3']},
    'P12 colour number: free under unimodularity, odd by the global anomaly': {'ga_internal_check.py': ['IN4', 'IN5']},
    'P12 coupling relation under the spectral action': {'ga_internal_check.py': ['IN6', 'IN7']},
    'P12 unimodularity is not volume preservation': {'ga_internal_dictionary_check.py': ['ID1', 'ID2', 'ID3']},
    'P12 Lambda and theta conjugate to exact top forms': {'ga_internal_dictionary_check.py': ['ID4']},
    'P12 half-integer spin from an odd colour number': {'ga_internal_spin_check.py': ['SP1', 'SP2', 'SP3']},
    'P12 internal label budget': {'ga_internal_budget_check.py': ['IB1', 'IB2', 'IB3']},
    'P12 Fisher-Rao is not unique under Diff(M)^N x S_N': {'ga_dens_metric_check.py': ['DM1', 'DM2', 'DM3', 'DM4', 'DM5']},
    'P12 marginal terms: Bohm potential of marginals, non-linear, hbar for products': {'ga_dens_dynamics_check.py': ['DD1', 'DD2', 'DD3', 'DD4', 'DD5']},
    'P12 no signalling selects linear many-body QM': {'ga_dens_signal_check.py': ['DS1', 'DS2', 'DS3', 'DS4']},
    'P12 link of the configuration-space picture with the series': {'ga_dens_series_check.py': ['DX1', 'DX2', 'DX3']},
    'P12 statistics is a phase condition': {'ga_dens_series_check.py': ['ST1', 'ST2', 'ST3', 'ST4']},
    # G. falsification pass (counterexamples and necessity of hypotheses)
    'G falsification':            {'ga_theorem_falsify_check.py': ['XA1', 'XA2', 'XA3', 'XA4', 'XA5', 'XA6', 'XA7', 'XB1', 'XB2', 'XB3', 'XB4',
                                    'XB5', 'XB6', 'XC1', 'XC2', 'XC3', 'XC4', 'XD1', 'XD2', 'XD3', 'XE1', 'XE2', 'XE3', 'XE4', 'XF1', 'XF3'],
                                   'ga_symmetry_map_falsify_check.py': ['RF4', 'RF6', 'RF7'], 'ga_symmetry_map_check.py': ['SM4'],
                                   'inv_numbers_falsify_check.py': ['XN3', 'XN4', 'XN5'], 'inv_all_numbers_check.py': ['ND6'],
                                   'inv_numbers_full_check.py': ['NF6'], 'ga_pure_universe_falsify_check.py': ['PU4', 'PU5', 'PU6'],
                                   'ga_all_falsify_check.py': ['AF4', 'AF5', 'AF6'],
                                   'ga_rehab_hypotheses_check.py': ['RH5', 'RH8'], 'ga_shear_de_a0_check.py': ['SH5', 'SH6'],
                                   'ga_sphere_de_check.py': ['VS6'], 'inv_flow_weights_falsify_check.py': ['XW6'],
                                   'ga_scaling_duality_check.py': ['DU7'], 'ga_angmom_radius_check.py': ['AM3']},
}

sys.path.insert(0, HERE)
import checkcache                                                        # same content-keyed cache as run_all.py
res = checkcache.results(sorted({s for t in BASE.values() for s in t}), full='--full' in sys.argv)
passed = {}
for script, r in res.items():
    passed[script] = set(re.findall(r"^PASS (\S+)", r['stdout'], re.M))
    if r['rc'] != 0:
        print(f"   {script}: exit {r['rc']}")
missing = []
for thm, refs in BASE.items():
    for script, labels in refs.items():
        for lab in labels:
            if lab not in passed[script]:
                missing.append((thm, script, lab))
n_lab = sum(len(l) for r in BASE.values() for l in r.values())
for m in missing:
    print("   missing:", m)
n_thm = sum(1 for k in BASE if not k.startswith(("G falsification", "P12 ")))
n_cmp = sum(1 for k in BASE if k.startswith("P12 "))
print(("PASS " if not missing else "FAIL ") + f"theorem base: {n_thm} theorems + falsification pass + {n_cmp} companion checks (paper12), {n_lab} labelled checks in {len(passed)} scripts, all PASS: {not missing}")
assert not missing
