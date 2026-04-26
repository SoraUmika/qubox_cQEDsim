from __future__ import annotations

import os
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import qutip as qt

import cqed_sim.calibration.sqr as sqr_module
import cqed_sim.calibration.targeted_subspace_multitone as targeted_module
import cqed_sim.floquet.core as floquet_core
import cqed_sim.optimal_control.readout_emptying_eval as readout_emptying_eval
import cqed_sim.sim.runner as sim_runner
import cqed_sim.solvers.master_equation as master_equation_module
import cqed_sim.solvers.trajectories as trajectories_module
import cqed_sim.tomo.protocol as tomo_protocol
import cqed_sim.unitary_synthesis.sequence as synthesis_sequence
from cqed_sim.calibration import ConditionedMultitoneRunConfig
from cqed_sim.core import DispersiveTransmonCavityModel, FrameSpec
from cqed_sim.floquet import FloquetConfig, FloquetProblem, solve_floquet
from cqed_sim.measurement.stochastic import ContinuousReadoutSpec, _stochastic_solver_options
from cqed_sim.models import IQPulse, MultilevelCQEDModel, ReadoutFrame, TransmonCosineSpec, diagonalize_transmon
from cqed_sim.pulses import Pulse
from cqed_sim.rl_control.configs import HybridSystemConfig, ReducedDispersiveModelConfig
from cqed_sim.rl_control.runtime import HamiltonianModelFactory, OpenSystemEngine
from cqed_sim.sequence import SequenceCompiler
from cqed_sim.sim import SimulationConfig, simulate_sequence
from cqed_sim.solvers import MasterEquationConfig, solve_master_equation
from cqed_sim.solvers.options import build_qutip_solver_options


def _two_mode_model() -> DispersiveTransmonCavityModel:
    return DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=0.0,
        alpha=-0.25,
        chi=-0.01,
        kerr=0.0,
        n_cav=2,
        n_tr=2,
    )


def test_shared_qutip_option_builder_validates_nsteps_and_conflicts() -> None:
    assert build_qutip_solver_options(nsteps=7, solver_options={"method": "bdf"}) == {
        "nsteps": 7,
        "method": "bdf",
    }
    assert build_qutip_solver_options(solver_options={"nsteps": "8"})["nsteps"] == 8

    with pytest.raises(ValueError, match="positive integer"):
        build_qutip_solver_options(nsteps=0)
    with pytest.raises(ValueError, match="set both"):
        build_qutip_solver_options(nsteps=7, solver_options={"nsteps": 8})


def test_master_equation_config_passes_nsteps_to_mesolve(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, dict[str, object]] = {}

    def fake_mesolve(hamiltonian, rho0, tlist, *, c_ops, e_ops, options):
        captured["options"] = dict(options)
        return SimpleNamespace(expect=[np.zeros(len(tlist)) for _ in e_ops], states=[rho0], final_state=rho0)

    monkeypatch.setattr(master_equation_module.qt, "mesolve", fake_mesolve)

    solve_master_equation(
        0.0 * qt.sigmax(),
        qt.basis(2, 0),
        tlist=np.linspace(0.0, 0.1, 3),
        config=MasterEquationConfig(nsteps=1234, solver_options={"method": "bdf"}),
    )

    assert captured["options"]["nsteps"] == 1234
    assert captured["options"]["method"] == "bdf"


def test_simulation_config_passes_nsteps_to_sesolve_and_mesolve(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _two_mode_model()
    compiled = SequenceCompiler(dt=0.05).compile([], t_end=0.1)
    captured: dict[str, dict[str, object]] = {}

    def fake_sesolve(hamiltonian, psi0, tlist, *, e_ops, options):
        captured["sesolve"] = dict(options)
        return SimpleNamespace(expect=[np.zeros(len(tlist)) for _ in e_ops], states=[psi0], final_state=psi0)

    def fake_mesolve(hamiltonian, rho0, tlist, *, c_ops, e_ops, options):
        captured["mesolve"] = dict(options)
        return SimpleNamespace(expect=[np.zeros(len(tlist)) for _ in e_ops], states=[rho0], final_state=rho0)

    monkeypatch.setattr(sim_runner.qt, "sesolve", fake_sesolve)
    monkeypatch.setattr(sim_runner.qt, "mesolve", fake_mesolve)
    config = SimulationConfig(nsteps=4321, solver_options={"method": "bdf"})

    simulate_sequence(model, compiled, model.basis_state(0, 0), {}, config, e_ops={})
    simulate_sequence(model, compiled, model.basis_state(0, 0).proj(), {}, config, e_ops={})

    assert captured["sesolve"]["nsteps"] == 4321
    assert captured["mesolve"]["nsteps"] == 4321
    assert captured["sesolve"]["method"] == "bdf"
    assert captured["mesolve"]["method"] == "bdf"


def test_unitary_synthesis_runtime_config_passes_options_to_propagator(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _two_mode_model()
    compiled = SequenceCompiler(dt=0.05).compile([], t_end=0.1)
    config = synthesis_sequence._runtime_simulation_config(
        {"nsteps": 91, "solver_options": {"method": "bdf"}}
    )
    captured: dict[str, dict[str, object]] = {}

    def fake_propagator(hamiltonian, times, *, options, tlist):
        captured["options"] = dict(options)
        return [qt.qeye(int(np.prod(model.subsystem_dims)))]

    monkeypatch.setattr(synthesis_sequence.qt, "propagator", fake_propagator)

    synthesis_sequence._final_unitary_from_compiled(
        model,
        compiled,
        {},
        frame=FrameSpec(),
        config=config,
    )

    assert captured["options"]["nsteps"] == 91
    assert captured["options"]["method"] == "bdf"


def test_floquet_config_passes_options_to_floquet_basis(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, dict[str, object]] = {}

    class FakeFloquetBasis:
        e_quasi = np.asarray([0.0, 0.25], dtype=float)

        def __init__(self, hamiltonian, period, **kwargs):
            captured["kwargs"] = dict(kwargs)

        def mode(self, time):
            return (qt.basis(2, 0), qt.basis(2, 1))

        def state(self, time):
            return self.mode(time)

        def U(self, time):
            return qt.qeye(2)

    monkeypatch.setattr(floquet_core.qt, "FloquetBasis", FakeFloquetBasis)
    problem = FloquetProblem(static_hamiltonian=qt.sigmaz(), period=1.0)

    solve_floquet(problem, FloquetConfig(nsteps=222, solver_options={"method": "bdf"}))

    options = captured["kwargs"]["options"]
    assert options["nsteps"] == 222
    assert options["method"] == "bdf"


def test_continuous_readout_spec_keeps_dt_and_merges_solver_options() -> None:
    options = _stochastic_solver_options(
        ContinuousReadoutSpec(
            max_step=0.05,
            solver_options={"nsteps": 333, "method": "rouchon"},
            progress_bar="",
        )
    )

    assert options["dt"] == 0.05
    assert options["nsteps"] == 333
    assert options["method"] == "rouchon"


def test_conditioned_and_legacy_multitone_solver_options() -> None:
    run_config = ConditionedMultitoneRunConfig(nsteps=444, solver_options={"method": "bdf"})
    assert targeted_module._solver_options(run_config)["nsteps"] == 444
    assert targeted_module._solver_options(replace(run_config, nsteps=None, solver_options={}))["nsteps"] == 100000

    legacy = sqr_module._solver_options(
        {
            "max_step_s": 0.0,
            "qutip_nsteps_sqr_calibration": 555,
            "solver_options": {"method": "bdf"},
        }
    )
    explicit = sqr_module._solver_options(
        {
            "max_step_s": 0.0,
            "qutip_nsteps_sqr_calibration": 555,
            "solver_options": {"nsteps": 556, "method": "bdf"},
        }
    )

    assert legacy["nsteps"] == 555
    assert explicit["nsteps"] == 556
    assert explicit["method"] == "bdf"


def test_trajectory_config_passes_master_equation_config(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, MasterEquationConfig] = {}
    rho = qt.basis(2, 0).proj()

    def fake_solve_master_equation(*args, config, **kwargs):
        captured["config"] = config
        times = np.asarray(kwargs["tlist"], dtype=float)
        return SimpleNamespace(
            states=[rho for _ in times],
            final_state=rho,
            expectations={
                "output_I": np.zeros(times.shape, dtype=float),
                "output_Q": np.zeros(times.shape, dtype=float),
            },
        )

    monkeypatch.setattr(trajectories_module, "solve_master_equation", fake_solve_master_equation)

    trajectories_module.simulate_measurement_trajectories(
        0.0 * qt.sigmax(),
        rho,
        tlist=np.linspace(0.0, 0.1, 3),
        output_operator=qt.destroy(2),
        config=trajectories_module.TrajectoryConfig(
            master_equation_config=MasterEquationConfig(nsteps=678, solver_options={"method": "bdf"})
        ),
    )

    assert captured["config"].nsteps == 678
    assert captured["config"].solver_options["method"] == "bdf"
    assert captured["config"].store_states is True


def test_rl_runtime_and_tomography_wrappers_preserve_solver_options(monkeypatch: pytest.MonkeyPatch) -> None:
    system = HybridSystemConfig(
        reduced_model=ReducedDispersiveModelConfig(
            omega_c=0.0,
            omega_q=0.0,
            alpha=-0.25,
            chi=-0.01,
            n_cav=2,
            n_tr=2,
        ),
        nsteps=789,
        solver_options={"method": "bdf"},
    )
    bundle = HamiltonianModelFactory.build(system)
    runtime_config = OpenSystemEngine(bundle)._simulation_config()
    assert runtime_config.nsteps == 789
    assert runtime_config.solver_options["method"] == "bdf"

    model = _two_mode_model()
    sim_config = SimulationConfig(nsteps=790, solver_options={"method": "bdf"})
    captured_configs: list[SimulationConfig] = []

    def fake_simulate_sequence(model, compiled, initial_state, drive_ops, config, noise=None, e_ops=None):
        captured_configs.append(config)
        return SimpleNamespace(final_state=initial_state)

    monkeypatch.setattr(tomo_protocol, "simulate_sequence", fake_simulate_sequence)
    tomo_protocol.run_all_xy(
        model,
        tomo_protocol.QubitPulseCal.nominal(),
        dt_ns=0.5,
        simulation_config=sim_config,
    )

    assert captured_configs
    assert all(config.nsteps == 790 for config in captured_configs)
    assert all(config.solver_options["method"] == "bdf" for config in captured_configs)


def test_readout_emptying_lindblad_validation_uses_simulation_config(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, SimulationConfig] = {}

    class FakeReadoutModel:
        subsystem_dims = (2,)

        def basis_state(self, level, *unused):
            return qt.basis(2, min(int(level), 1))

    class FakeSession:
        def run(self, initial_state):
            times = np.asarray([0.0, 0.1], dtype=float)
            return SimpleNamespace(
                expectations={
                    "a_r": np.zeros(times.shape, dtype=np.complex128),
                    "n_r": np.zeros(times.shape, dtype=float),
                    "P_g": np.ones(times.shape, dtype=float),
                    "P_e": np.zeros(times.shape, dtype=float),
                }
            )

    def fake_prepare_simulation(model, compiled, drive_ops, *, config, noise=None, e_ops=None):
        captured["config"] = config
        return FakeSession()

    monkeypatch.setattr(readout_emptying_eval, "prepare_simulation", fake_prepare_simulation)
    monkeypatch.setattr(
        readout_emptying_eval,
        "_default_lindblad_observables",
        lambda model: {"a_r": qt.qeye(2), "n_r": qt.qeye(2), "P_g": qt.qeye(2), "P_e": qt.qeye(2)},
    )
    monkeypatch.setattr(
        readout_emptying_eval,
        "export_readout_emptying_to_pulse",
        lambda result, channel, carrier: Pulse(channel, 0.0, 0.1, lambda t: np.ones_like(t), carrier=carrier),
    )
    result = SimpleNamespace(
        segment_edges_s=np.asarray([0.0, 0.1], dtype=float),
        spec=SimpleNamespace(tau=0.1, kappa=0.5),
    )
    sim_config = SimulationConfig(nsteps=801, solver_options={"method": "bdf"})

    readout_emptying_eval._lindblad_validation(
        result,
        readout_model=FakeReadoutModel(),
        frame=FrameSpec(omega_r_frame=0.0),
        noise=None,
        compiler_dt_s=0.1,
        max_step_s=None,
        simulation_config=sim_config,
        drive_frequency=1.25,
    )

    assert captured["config"].nsteps == 801
    assert captured["config"].solver_options["method"] == "bdf"
    assert captured["config"].frame.omega_r_frame == 1.25


def test_compact_real_solves_accept_nsteps_configs() -> None:
    master = solve_master_equation(
        0.1 * qt.sigmax(),
        qt.basis(2, 0),
        tlist=np.linspace(0.0, 0.1, 4),
        config=MasterEquationConfig(nsteps=50, solver_options={"method": "adams"}),
    )
    assert abs(master.final_state.tr() - 1.0) < 1.0e-10

    model = _two_mode_model()
    compiled = SequenceCompiler(dt=0.05).compile([], t_end=0.1)
    runtime = simulate_sequence(
        model,
        compiled,
        model.basis_state(0, 0),
        {},
        SimulationConfig(nsteps=50, solver_options={"method": "adams"}),
        e_ops={},
    )
    assert np.isclose(runtime.final_state.norm(), 1.0)


@pytest.mark.slow
def test_gated_large_cutoff_full_cosine_master_equation_uses_package_nsteps() -> None:
    if os.environ.get("CQED_SIM_RUN_LARGE_CUTOFF_FULL_COSINE") != "1":
        pytest.skip("Set CQED_SIM_RUN_LARGE_CUTOFF_FULL_COSINE=1 to run the Nq=9, Nr=16 stability check.")

    duration = 0.32
    dt = 0.08
    samples = np.full(max(1, int(round(duration / dt))), 0.004 + 0.001j, dtype=np.complex128)
    spectrum = diagonalize_transmon(TransmonCosineSpec(EJ=34.0, EC=0.55, n_cut=15, levels=9))
    model = MultilevelCQEDModel.from_transmon_spectrum(
        spectrum,
        resonator_frequency=0.0,
        resonator_levels=16,
        coupling_strength=0.012,
        rotating_frame=ReadoutFrame(resonator_frequency=0.0),
        counter_rotating=False,
    )
    pulse = IQPulse(samples=samples, dt=dt, drive_frequency=0.0)
    hdata = model.build_hamiltonian(pulse)
    result = solve_master_equation(
        hdata,
        model.basis_state(0, 0),
        c_ops=[],
        e_ops={"n_r": model.operators()["n_r"]},
        config=MasterEquationConfig(nsteps=200000, atol=1.0e-8, rtol=1.0e-7),
    )

    assert np.isfinite(float(np.real(result.final_state.tr())))
    assert np.all(np.isfinite(result.expectations["n_r"]))
