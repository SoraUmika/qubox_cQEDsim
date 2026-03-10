# Route B Workflow and Enlarged-Control Study

## Route B

- `uniform_pi_zero`: naive `T=450 ns` -> co-designed `T=700 ns`, post-SNAP fidelity `0.8079 -> 0.7697`, residual SNAP RMS `1.4550 -> 0.2441 rad`.
- `structured_zero`: naive `T=450 ns` -> co-designed `T=700 ns`, post-SNAP fidelity `0.8512 -> 0.9689`, residual SNAP RMS `1.5429 -> 0.2441 rad`.
- `uniform_pi_quadratic`: naive `T=450 ns` -> co-designed `T=700 ns`, post-SNAP fidelity `0.5429 -> 0.7695`, residual SNAP RMS `3.5822 -> 0.1050 rad`.
- `uniform_pi_random`: naive `T=450 ns` -> co-designed `T=500 ns`, post-SNAP fidelity `0.7863 -> 0.8997`, residual SNAP RMS `1.8426 -> 3.9431 rad`.

## Enlarged Control

- `far_from_drift`: route-B SQR `F=0.0456`, displacement conjugation `F=0.0490`, explicit SNAP assist `F=0.8789`.
- `kerr_cancel`: route-B SQR `F=0.7694`, displacement conjugation `F=0.7694`, explicit SNAP assist `F=0.7697`.
- `natural_drift`: route-B SQR `F=0.7697`, displacement conjugation `F=0.7700`, explicit SNAP assist `F=0.7697`.
- `random_medium_a`: route-B SQR `F=0.7374`, displacement conjugation `F=0.7374`, explicit SNAP assist `F=0.7697`.
- `random_small_a`: route-B SQR `F=0.7345`, displacement conjugation `F=0.7345`, explicit SNAP assist `F=0.7697`.
- `zero`: route-B SQR `F=0.7450`, displacement conjugation `F=0.7450`, explicit SNAP assist `F=0.7697`.

## Bottom Line

- Timing co-design plus residual SNAP cleanup is effective and practical.
- A minimal cavity displacement conjugation does not materially recover arbitrary block-phase synthesis in this study.
- An explicit cavity-diagonal SNAP-like phase assist immediately restores the missing phase degree of freedom.
