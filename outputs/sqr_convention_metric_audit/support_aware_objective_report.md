# Support-Aware Objective Comparison

- Compared legacy global objective vs new support-aware objective on tiny benchmark-style targets.
- Truncation: n_max=5, chi=-2.84 MHz, duration=1.0 us.
- Case E is the default expressive ansatz in support-aware mode; Case D is retained as fallback/ablation.

## Active Support: S01
- Levels: [0, 1]

| Target | Objective | Case | active weighted mean fid | active min fid | active theta RMS | active phase RMS | active pre-Z RMS | active post-Z RMS | support-leak mean/max | spectral-leak mean/max | state mean/min | phase-super RMS | global mean/min fid | global phase-sensitive RMS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| x90_x90 | global | E | 0.476903 | 0.453842 | 1.288986 | 0.000000 | 0.980303 | 0.543726 | 9.9987e-09/1.3611e-08 | 1.1095e-04/1.1095e-04 | 0.530100/0.370381 | 0.363001 | 0.741670/0.453842 | 2.374853 |
| x90_x90 | support | E | 0.483794 | 0.472032 | 1.245994 | 0.000000 | 0.181557 | 0.181557 | 1.3127e-08/1.9116e-08 | 3.3255e-05/3.3255e-05 | 0.949861/0.914120 | 0.176584 | 0.682606/0.472032 | 2.278475 |
- Delta (support-aware - global objective): active mean fid=+0.0069, active min fid=+0.0182, state min=+0.5437, support-state leak mean=+3.129e-09.
- Active-boundary crosstalk (Case D diagnostic): global boundary spill=1.656e-01, support boundary spill=1.509e-01, global max active->inactive=5.119e+02, support max active->inactive=4.642e+02.
- Restricted active->inactive crosstalk matrices (rows=inactive, cols=active): global=[[0.047, 0.166], [92.425, 12.84], [511.811, 25.536], [511.933, 26.371]], support=[[0.048, 0.151], [6.775, 82.824], [42.447, 463.319], [42.563, 464.176]].
- Dominant support-aware loss terms (primary case): active_infidelity=5.162e-01, active_theta=3.105e-01, worst_block=2.064e-01, active_state_mean=2.758e-02
- Case E vs Case D (support-aware ablation): active mean fid delta=+0.0122, active min fid delta=+0.0035, state min delta=+0.2200.
| x90_y90 | global | E | 0.488204 | 0.478029 | 1.311739 | 1.110721 | 0.982154 | 0.673377 | 6.3929e-09/9.9400e-09 | 2.5448e-03/2.5448e-03 | 0.605233/0.577660 | 0.298665 | 0.729764/0.478029 | 2.331041 |
| x90_y90 | support | E | 0.475801 | 0.466182 | 1.139775 | 1.110721 | 0.417985 | 0.124512 | 7.5482e-09/1.1719e-08 | 3.6255e-05/3.6255e-05 | 0.920543/0.798552 | 0.242688 | 0.636438/0.466182 | 2.293919 |
- Delta (support-aware - global objective): active mean fid=-0.0124, active min fid=-0.0118, state min=+0.2209, support-state leak mean=+1.155e-09.
- Active-boundary crosstalk (Case D diagnostic): global boundary spill=1.500e-01, support boundary spill=1.539e-01, global max active->inactive=5.670e+03, support max active->inactive=1.056e+02.
- Restricted active->inactive crosstalk matrices (rows=inactive, cols=active): global=[[0.047, 0.15], [474.714, 1032.787], [2606.771, 5669.135], [2606.928, 5670.078]], support=[[0.048, 0.154], [15.648, 17.367], [91.068, 104.708], [91.183, 105.556]].
- Dominant support-aware loss terms (primary case): active_infidelity=5.242e-01, active_phase_axis=4.318e-01, active_theta=2.598e-01, worst_block=2.112e-01
- Case E vs Case D (support-aware ablation): active mean fid delta=-0.0011, active min fid delta=-0.0027, state min delta=-0.0950.
| x180_identity | global | E | 0.483557 | 0.000000 | 1.424155 | 0.000000 | 0.617290 | 0.617290 | 1.8927e-09/3.2437e-09 | 3.3211e-04/3.3211e-04 | 0.492433/0.072722 | 0.155412 | 0.723767/0.000000 | 2.149229 |
| x180_identity | support | E | 0.499802 | 0.000000 | 0.145202 | 0.000000 | 0.346782 | 0.346781 | 9.3302e-10/3.5803e-09 | 1.3779e-15/1.3779e-15 | 0.965443/0.912593 | 0.114382 | 0.614027/0.000000 | 2.439620 |
- Delta (support-aware - global objective): active mean fid=+0.0162, active min fid=+0.0000, state min=+0.8399, support-state leak mean=-9.597e-10.
- Active-boundary crosstalk (Case D diagnostic): global boundary spill=1.395e-01, support boundary spill=0.000e+00, global max active->inactive=7.864e+06, support max active->inactive=5.314e+00.
- Restricted active->inactive crosstalk matrices (rows=inactive, cols=active): global=[[0.047, 0.139], [95.331, 1435059.276], [527.735, 7864198.613], [527.857, 7864331.075]], support=[[0.003, 0.0], [1.143, 0.97], [1.559, 5.314], [1.567, 5.314]].
- Dominant support-aware loss terms (primary case): worst_block=7.683e-01, active_infidelity=5.002e-01, active_pre_z=3.006e-02, active_post_z=3.006e-02
- Case E vs Case D (support-aware ablation): active mean fid delta=-0.0000, active min fid delta=+0.0000, state min delta=-0.0000.

## Active Support: S0123
- Levels: [0, 1, 2, 3]

| Target | Objective | Case | active weighted mean fid | active min fid | active theta RMS | active phase RMS | active pre-Z RMS | active post-Z RMS | support-leak mean/max | spectral-leak mean/max | state mean/min | phase-super RMS | global mean/min fid | global phase-sensitive RMS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| x90_x90 | global | E | 0.730138 | 0.453842 | 0.929687 | 0.000000 | 0.699207 | 0.395238 | 2.5991e-07/9.8618e-07 | 2.7506e-08/2.7506e-08 | 0.556237/0.319274 | 0.926015 | 0.741670/0.453842 | 2.374853 |
| x90_x90 | support | E | 0.714601 | 0.494242 | 1.093642 | 0.000000 | 0.196297 | 0.196297 | 8.5686e-08/5.0840e-07 | 2.0291e-07/2.0291e-07 | 0.864686/0.403873 | 0.135095 | 0.734936/0.494242 | 2.394651 |
- Delta (support-aware - global objective): active mean fid=-0.0155, active min fid=+0.0404, state min=+0.0846, support-state leak mean=-1.742e-07.
- Active-boundary crosstalk (Case D diagnostic): global boundary spill=5.480e+00, support boundary spill=5.484e+00, global max active->inactive=1.181e+03, support max active->inactive=5.631e+03.
- Restricted active->inactive crosstalk matrices (rows=inactive, cols=active): global=[[511.811, 25.536, 1181.206, 5.48], [511.933, 26.371, 1174.012, 5.48]], support=[[81.613, 73.582, 5631.165, 5.484], [81.728, 74.428, 5624.029, 5.431]].
- Dominant support-aware loss terms (primary case): active_infidelity=2.854e-01, active_theta=2.392e-01, worst_block=1.888e-01, active_state_min=1.442e-01
- Case E vs Case D (support-aware ablation): active mean fid delta=-0.0092, active min fid delta=+0.0115, state min delta=-0.1971.
| x90_y90 | global | E | 0.725305 | 0.478029 | 0.968178 | 0.785398 | 0.708218 | 0.495962 | 1.6776e-07/6.7460e-07 | 5.5991e-07/5.5991e-07 | 0.633482/0.478969 | 0.784954 | 0.729764/0.478029 | 2.331041 |
| x90_y90 | support | E | 0.719475 | 0.489031 | 1.021204 | 0.785398 | 0.185638 | 0.187846 | 2.2298e-07/7.9490e-07 | 6.1033e-06/6.1033e-06 | 0.808645/0.539166 | 0.185011 | 0.728882/0.489031 | 2.343223 |
- Delta (support-aware - global objective): active mean fid=-0.0058, active min fid=+0.0110, state min=+0.0602, support-state leak mean=+5.522e-08.
- Active-boundary crosstalk (Case D diagnostic): global boundary spill=5.480e+00, support boundary spill=5.483e+00, global max active->inactive=3.058e+07, support max active->inactive=1.190e+03.
- Restricted active->inactive crosstalk matrices (rows=inactive, cols=active): global=[[2606.771, 5669.135, 30577129.804, 5.48], [2606.928, 5670.078, 30577628.605, 5.48]], support=[[95.461, 93.315, 1189.52, 5.483], [95.577, 94.162, 1182.326, 5.446]].
- Dominant support-aware loss terms (primary case): active_infidelity=2.805e-01, active_phase_axis=2.159e-01, active_theta=2.086e-01, worst_block=1.928e-01
- Case E vs Case D (support-aware ablation): active mean fid delta=-0.0036, active min fid delta=+0.0054, state min delta=-0.0563.
| x180_identity | global | E | 0.739419 | 0.000000 | 1.011718 | 0.000000 | 0.439192 | 0.439192 | 9.1785e-08/5.2028e-07 | 3.1302e-09/3.1302e-09 | 0.617653/0.072722 | 0.773015 | 0.723767/0.000000 | 2.149229 |
| x180_identity | support | E | 0.748193 | 0.000000 | 1.006341 | 0.000000 | 0.464269 | 0.464269 | 1.1688e-07/6.3719e-07 | 5.6688e-08/5.6688e-08 | 0.870142/0.444604 | 0.675883 | 0.756585/0.000000 | 2.216598 |
- Delta (support-aware - global objective): active mean fid=+0.0088, active min fid=+0.0000, state min=+0.3719, support-state leak mean=+2.510e-08.
- Active-boundary crosstalk (Case D diagnostic): global boundary spill=5.480e+00, support boundary spill=5.484e+00, global max active->inactive=1.110e+07, support max active->inactive=1.726e+04.
- Restricted active->inactive crosstalk matrices (rows=inactive, cols=active): global=[[527.735, 7864198.613, 11100089.397, 5.48], [527.857, 7864331.075, 11100265.364, 5.48]], support=[[32.575, 17263.471, 3382.349, 5.484], [32.69, 17264.606, 3375.179, 5.435]].
- Dominant support-aware loss terms (primary case): worst_block=7.683e-01, active_infidelity=2.518e-01, active_theta=2.025e-01, phase_superposition=1.599e-01
- Case E vs Case D (support-aware ablation): active mean fid delta=+0.0083, active min fid delta=+0.0000, state min delta=-0.1478.

## Main Question
- Support-aware optimization can improve the experimentally occupied subspace metrics even when global equal-weight metrics move less.
- The largest practical gains typically come from worst-block protection and active-ensemble fidelity terms; leakage terms are most impactful near support boundaries.