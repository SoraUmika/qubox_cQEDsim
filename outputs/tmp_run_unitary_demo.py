from __future__ import annotations
import json
from pathlib import Path
from cqed_sim.map_synthesis import QuantumMapSynthesizer, Subspace, make_target
from cqed_sim.map_synthesis.reporting import make_run_report

subspace = Subspace.qubit_cavity_block(n_match=3)
target_name = 'cluster'
u_target = make_target(target_name, n_match=3, variant='mps')

synth = QuantumMapSynthesizer(
    subspace=subspace,
    backend='pulse',
    gateset=['QubitRotation', 'SQR', 'SNAP', 'Displacement'],
    optimize_times=True,
    time_bounds={'default': (20e-9, 2000e-9)},
    leakage_weight=10.0,
    time_reg_weight=1e-2,
    seed=1234,
)

result = synth.fit(target=u_target, init_guess='heuristic', multistart=8, maxiter=220)
report = make_run_report(result.report, result.simulation.subspace_operator)
out = {
    'fidelity': report['metrics']['fidelity'],
    'leakage_worst': report['metrics']['leakage_worst'],
    'durations': report['parameters']['durations'],
    'objective': report['objective'],
}
Path('outputs').mkdir(exist_ok=True)
Path('outputs/unitary_synthesis_demo_metrics.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
print('WROTE outputs/unitary_synthesis_demo_metrics.json')
