# Support-Aware Objective Comparison

- Compared legacy global objective vs new support-aware objective on tiny benchmark-style targets.
- Truncation: n_max=5, chi=-2.84 MHz, duration=1.0 us.
- Case E is the default expressive ansatz in support-aware mode; Case D is retained as fallback/ablation.

## Active Support: S01
- Levels: [0, 1]

| Target | Objective | Case | active weighted mean fid | active min fid | active theta RMS | active phase RMS | active pre-Z RMS | active post-Z RMS | support-leak mean/max | spectral-leak mean/max | state mean/min | phase-super RMS | global mean/min fid | global phase-sensitive RMS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| x90_x90 | global | E | 0.844367 | 0.830092 | 0.238991 | 0.377990 | 0.729269 | 0.627687 | 1.4140e-09/5.6460e-09 | 4.1427e-05/4.1427e-05 | 0.831553/0.815409 | 0.202024 | 0.461037/0.028221 | 1.863154 |
| x90_x90 | support | E | 0.894967 | 0.791086 | 0.117477 | 0.356111 | 0.368011 | 0.399584 | 2.9056e-09/5.9698e-09 | 9.2232e-05/9.2232e-05 | 0.849518/0.791481 | 0.044922 | 0.478237/0.025777 | 2.082482 |
- Delta (support-aware - global objective): active mean fid=+0.0506, active min fid=-0.0390, state min=-0.0239, support-state leak mean=+1.492e-09.
- Active-boundary crosstalk (Case D diagnostic): global boundary spill=6.390e-01, support boundary spill=1.014e+00, global max active->inactive=1.382e+00, support max active->inactive=1.827e+00.
- Restricted active->inactive crosstalk matrices (rows=inactive, cols=active): global=[[0.894, 0.639], [1.382, 1.08], [1.194, 0.979], [0.718, 0.59]], support=[[1.197, 1.014], [1.827, 1.592], [1.55, 1.366], [0.929, 0.818]].
- Dominant support-aware loss terms (primary case): active_infidelity=1.050e-01, active_state_mean=8.277e-02, active_phase_axis=4.439e-02, active_post_z=3.992e-02
- Case E vs Case D (support-aware ablation): active mean fid delta=+0.0000, active min fid delta=+0.0000, state min delta=+0.0000.
| x90_y90 | global | E | 0.901494 | 0.812645 | 0.122574 | 0.367562 | 0.202699 | 0.122172 | 2.1722e-09/4.7065e-09 | 7.8721e-05/7.8721e-05 | 0.829754/0.770512 | 0.166407 | 0.481872/0.029175 | 1.905001 |
| x90_y90 | support | E | 0.993286 | 0.989944 | 0.008168 | 0.039119 | 0.047364 | 0.047364 | 1.4313e-09/5.6991e-09 | 7.7068e-05/7.7068e-05 | 0.957674/0.918822 | 0.195343 | 0.512996/0.031726 | 1.883863 |
- Delta (support-aware - global objective): active mean fid=+0.0918, active min fid=+0.1773, state min=+0.1483, support-state leak mean=-7.409e-10.
- Active-boundary crosstalk (Case D diagnostic): global boundary spill=1.014e+00, support boundary spill=1.081e+00, global max active->inactive=1.684e+00, support max active->inactive=1.855e+00.
- Restricted active->inactive crosstalk matrices (rows=inactive, cols=active): global=[[1.1, 1.014], [1.684, 1.592], [1.435, 1.366], [0.86, 0.818]], support=[[1.216, 1.081], [1.855, 1.685], [1.573, 1.438], [0.942, 0.861]].
- Dominant support-aware loss terms (primary case): active_state_mean=2.328e-02, inactive_infidelity=1.454e-02, phase_superposition=1.336e-02, active_infidelity=6.714e-03
- Case E vs Case D (support-aware ablation): active mean fid delta=+0.0400, active min fid delta=+0.0783, state min delta=+0.0549.
| x180_identity | global | E | 0.767080 | 0.592766 | 0.434078 | 0.080869 | 0.451957 | 0.321260 | 9.3721e-10/3.6723e-09 | 0.0000e+00/0.0000e+00 | 0.779858/0.597820 | 0.265729 | 0.437080/0.042008 | 1.303209 |
| x180_identity | support | E | 0.971144 | 0.942507 | 0.342444 | 0.021368 | 0.171452 | 0.171592 | 6.0258e-10/2.4103e-09 | 0.0000e+00/0.0000e+00 | 0.970542/0.915058 | 0.161184 | 0.504648/0.037634 | 2.176699 |
- Delta (support-aware - global objective): active mean fid=+0.2041, active min fid=+0.3497, state min=+0.3172, support-state leak mean=-3.346e-10.
- Active-boundary crosstalk (Case D diagnostic): global boundary spill=0.000e+00, support boundary spill=0.000e+00, global max active->inactive=7.927e-01, support max active->inactive=8.746e-01.
- Restricted active->inactive crosstalk matrices (rows=inactive, cols=active): global=[[0.483, 0.0], [0.793, 0.0], [0.741, 0.0], [0.451, 0.0]], support=[[0.541, 0.0], [0.875, 0.0], [0.802, 0.0], [0.486, 0.0]].
- Dominant support-aware loss terms (primary case): active_infidelity=2.886e-02, active_theta=2.345e-02, active_state_mean=1.620e-02, inactive_infidelity=1.457e-02
- Case E vs Case D (support-aware ablation): active mean fid delta=+0.0000, active min fid delta=+0.0000, state min delta=+0.0000.

## Active Support: S0123
- Levels: [0, 1, 2, 3]

| Target | Objective | Case | active weighted mean fid | active min fid | active theta RMS | active phase RMS | active pre-Z RMS | active post-Z RMS | support-leak mean/max | spectral-leak mean/max | state mean/min | phase-super RMS | global mean/min fid | global phase-sensitive RMS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| x90_x90 | global | E | 0.530742 | 0.028221 | 1.663877 | 1.163628 | 0.975249 | 0.939012 | 8.8444e-08/5.3060e-07 | 0.0000e+00/0.0000e+00 | 0.812683/0.394580 | 0.244656 | 0.461037/0.028221 | 1.863154 |
| x90_x90 | support | E | 0.547386 | 0.045740 | 1.583653 | 0.723552 | 0.976282 | 1.200502 | 1.2370e-07/7.3571e-07 | 0.0000e+00/0.0000e+00 | 0.791025/0.426882 | 0.309630 | 0.464772/0.045740 | 2.292890 |
- Delta (support-aware - global objective): active mean fid=+0.0166, active min fid=+0.0175, state min=+0.0323, support-state leak mean=+3.525e-08.
- Active-boundary crosstalk (Case D diagnostic): global boundary spill=0.000e+00, support boundary spill=0.000e+00, global max active->inactive=1.194e+00, support max active->inactive=1.083e+00.
- Restricted active->inactive crosstalk matrices (rows=inactive, cols=active): global=[[1.194, 0.979, 0.0, 0.0], [0.718, 0.59, 0.0, 0.0]], support=[[1.083, 0.891, 0.0, 0.0], [0.652, 0.539, 0.0, 0.0]].
- Dominant support-aware loss terms (primary case): worst_block=6.983e-01, active_theta=5.016e-01, active_infidelity=4.526e-01, active_post_z=3.603e-01
- Case E vs Case D (support-aware ablation): active mean fid delta=+0.0165, active min fid delta=-0.0086, state min delta=+0.0735.
| x90_y90 | global | E | 0.562160 | 0.029175 | 1.649494 | 1.036249 | 0.849533 | 0.815017 | 8.7793e-08/5.2281e-07 | 0.0000e+00/0.0000e+00 | 0.824512/0.374173 | 0.146814 | 0.481872/0.029175 | 1.905001 |
| x90_y90 | support | E | 0.541534 | 0.028084 | 1.666515 | 0.537561 | 0.872362 | 0.879447 | 1.2439e-07/7.4532e-07 | 0.0000e+00/0.0000e+00 | 0.814129/0.423872 | 0.213453 | 0.468824/0.028084 | 2.144756 |
- Delta (support-aware - global objective): active mean fid=-0.0206, active min fid=-0.0011, state min=+0.0497, support-state leak mean=+3.659e-08.
- Active-boundary crosstalk (Case D diagnostic): global boundary spill=0.000e+00, support boundary spill=0.000e+00, global max active->inactive=1.435e+00, support max active->inactive=1.905e+00.
- Restricted active->inactive crosstalk matrices (rows=inactive, cols=active): global=[[1.435, 1.366, 0.0, 0.0], [0.86, 0.818, 0.0, 0.0]], support=[[1.905, 1.289, 0.0, 0.0], [1.14, 0.773, 0.0, 0.0]].
- Dominant support-aware loss terms (primary case): worst_block=7.249e-01, active_theta=5.555e-01, active_infidelity=4.585e-01, active_post_z=1.934e-01
- Case E vs Case D (support-aware ablation): active mean fid delta=+0.0199, active min fid delta=-0.0004, state min delta=+0.0164.
| x180_identity | global | E | 0.505734 | 0.042008 | 1.630705 | 0.065176 | 0.862203 | 0.832365 | 1.2336e-07/7.4011e-07 | 0.0000e+00/0.0000e+00 | 0.791833/0.378511 | 0.754908 | 0.437080/0.042008 | 1.303209 |
| x180_identity | support | E | 0.409370 | 0.035355 | 1.783441 | 0.040503 | 0.922865 | 0.913725 | 1.2348e-07/7.4087e-07 | 0.0000e+00/0.0000e+00 | 0.697652/0.271318 | 0.960730 | 0.375695/0.035355 | 2.207614 |
- Delta (support-aware - global objective): active mean fid=-0.0964, active min fid=-0.0067, state min=-0.1072, support-state leak mean=+1.191e-10.
- Active-boundary crosstalk (Case D diagnostic): global boundary spill=0.000e+00, support boundary spill=0.000e+00, global max active->inactive=7.415e-01, support max active->inactive=7.897e-01.
- Restricted active->inactive crosstalk matrices (rows=inactive, cols=active): global=[[0.741, 0.0, 0.0, 0.0], [0.451, 0.0, 0.0, 0.0]], support=[[0.79, 0.0, 0.0, 0.0], [0.479, 0.0, 0.0, 0.0]].
- Dominant support-aware loss terms (primary case): worst_block=7.139e-01, active_theta=6.361e-01, active_infidelity=5.906e-01, phase_superposition=3.231e-01
- Case E vs Case D (support-aware ablation): active mean fid delta=-0.0883, active min fid delta=-0.0012, state min delta=-0.1163.

## Main Question
- Support-aware optimization can improve the experimentally occupied subspace metrics even when global equal-weight metrics move less.
- The largest practical gains typically come from worst-block protection and active-ensemble fidelity terms; leakage terms are most impactful near support boundaries.