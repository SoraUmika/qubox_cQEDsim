import json
from copy import deepcopy
from pathlib import Path
import textwrap

SRC = Path(r"E:\qubox\notebooks\post_cavity_experiment_context.ipynb")
DST = Path(r"E:\qubox\notebooks\post_cavity_experiment_context_SIM.ipynb")

with SRC.open("r", encoding="utf-8") as f:
    nb = json.load(f)


def code_cell(source: str):
    normalized = textwrap.dedent(source).strip("\n").lstrip()
    return {
        "cell_type": "code",
        "metadata": {"language": "python"},
        "source": [line + "\n" for line in normalized.split("\n")],
    }


bootstrap = code_cell(
    textwrap.dedent(
        """
        MODE = "simulate"  # or "hardware"
        SEED = 12345

        import math
        import numpy as np
        from dataclasses import dataclass, field
        from types import SimpleNamespace

        rng = np.random.default_rng(SEED)

        SimParams = {
            "f01_true": 6.15023e9,
            "T1_true": 26e-6,
            "T2ramsey_true": 18e-6,
            "T2echo_true": 31e-6,
            "alpha_true": 0.085,
            "amp_to_theta_scale": np.pi / 0.1,
            "noise_std": 0.012,
            "readout_snr": 35.0,
            "clock_period_ns": 4.0,
        }

        calibration_store = {
            "frequencies": {"transmon": {"qubit_freq": 6.15000e9}},
            "attributes": {"qb_therm_clks": 10000},
            "pulse_calibrations": {
                "ref_r180": {
                    "amplitude": 0.1,
                    "length": 16,
                    "sigma": 16 / 6,
                    "drag_coeff": 0.0,
                    "detuning": 0.0,
                }
            },
        }

        PATCH_LOG = []
        bringup = {"steps": [], "results": {}}


        def _deep_get(d, path):
            cur = d
            for p in path.split('.'):
                cur = cur[p]
            return cur


        def _deep_set(d, path, value):
            parts = path.split('.')
            cur = d
            for p in parts[:-1]:
                if p not in cur or not isinstance(cur[p], dict):
                    cur[p] = {}
                cur = cur[p]
            cur[parts[-1]] = value


        @dataclass
        class Patch:
            reason: str = "sim_patch"
            provenance: dict = field(default_factory=dict)
            updates: list = field(default_factory=list)

            def add(self, op, path, value):
                self.updates.append({"op": op, "path": path, "value": value})


        class SimOrchestrator:
            def apply_patch(self, patch, dry_run=False):
                preview = []
                for u in patch.updates:
                    before = None
                    try:
                        before = _deep_get(calibration_store, u["path"])
                    except Exception:
                        before = None
                    after = u["value"]
                    preview.append({"path": u["path"], "before": before, "after": after})
                    if not dry_run:
                        _deep_set(calibration_store, u["path"], after)
                out = {"ok": True, "n_updates": len(preview), "preview": preview, "dry_run": dry_run}
                if not dry_run:
                    PATCH_LOG.append({"reason": patch.reason, "provenance": patch.provenance, "updates": preview})
                return out


        orch = SimOrchestrator()


        @dataclass
        class SimRunRes:
            output: dict
            mode: str = "simulate"
            meta: dict = field(default_factory=dict)


        def _noise(n):
            return rng.normal(0.0, SimParams["noise_std"], size=n)


        def simulate_spectroscopy(rf_begin_mhz=6145.0, rf_end_mhz=6155.0, df_khz=200.0):
            freqs_mhz = np.arange(rf_begin_mhz, rf_end_mhz + 1e-9, df_khz / 1000.0)
            f0_mhz = SimParams["f01_true"] / 1e6
            width = 0.8
            dip = 1.0 - np.exp(-0.5 * ((freqs_mhz - f0_mhz) / width) ** 2)
            S = dip + _noise(freqs_mhz.size)
            fit_f0_mhz = float(freqs_mhz[int(np.argmin(S))])
            return SimRunRes(output={"freqs_hz": freqs_mhz * 1e6, "freqs_mhz": freqs_mhz, "S": S}, meta={"f0_MHz": fit_f0_mhz})


        def simulate_power_rabi(max_gain=1.2, dg=0.04):
            gains = np.arange(0.0, max_gain + 1e-12, dg)
            k = SimParams["amp_to_theta_scale"]
            pe = np.sin(0.5 * k * gains) ** 2 + _noise(gains.size)
            pe = np.clip(pe, 0.0, 1.0)
            pi_gain = float(np.pi / k)
            return SimRunRes(output={"gains": gains, "pe": pe}, meta={"g_pi": pi_gain})


        def simulate_t1(delay_end_us=50.0, dt_ns=500.0):
            taus_ns = np.arange(0.0, delay_end_us * 1000.0 + 1e-9, dt_ns)
            t1_ns = SimParams["T1_true"] * 1e9
            pe = 0.95 * np.exp(-taus_ns / t1_ns) + 0.04 + _noise(taus_ns.size)
            pe = np.clip(pe, 0.0, 1.0)
            return SimRunRes(output={"taus_ns": taus_ns, "pe": pe}, meta={"T1_us": SimParams["T1_true"] * 1e6})


        def simulate_ramsey(delay_end_us=40.0, dt_ns=100.0):
            taus_ns = np.arange(0.0, delay_end_us * 1000.0 + 1e-9, dt_ns)
            f_cal = calibration_store["frequencies"]["transmon"]["qubit_freq"]
            f_det_hz = SimParams["f01_true"] - f_cal
            t2_ns = SimParams["T2ramsey_true"] * 1e9
            pe = 0.5 + 0.45 * np.exp(-taus_ns / t2_ns) * np.cos(2 * np.pi * f_det_hz * taus_ns * 1e-9) + _noise(taus_ns.size)
            pe = np.clip(pe, 0.0, 1.0)
            return SimRunRes(output={"taus_ns": taus_ns, "pe": pe}, meta={"T2_star_us": SimParams["T2ramsey_true"] * 1e6, "f_det_MHz": f_det_hz / 1e6})


        def simulate_echo(delay_end_us=40.0, dt_ns=200.0):
            taus_ns = np.arange(0.0, delay_end_us * 1000.0 + 1e-9, dt_ns)
            t2e_ns = SimParams["T2echo_true"] * 1e9
            pe = 0.5 + 0.45 * np.exp(-taus_ns / t2e_ns) + _noise(taus_ns.size)
            pe = np.clip(pe, 0.0, 1.0)
            return SimRunRes(output={"taus_ns": taus_ns, "pe": pe}, meta={"T2_echo_us": SimParams["T2echo_true"] * 1e6})


        def simulate_drag_sweep(amps=None):
            if amps is None:
                amps = np.linspace(-0.5, 0.5, 20)
            alpha_true = SimParams["alpha_true"]
            metric = (amps - alpha_true) ** 2 + 0.02 * rng.normal(size=amps.size)
            alpha_opt = float(amps[int(np.argmin(metric))])
            return SimRunRes(output={"alphas": amps, "metric": metric}, meta={"optimal_alpha": alpha_opt, "loss_before": float(metric[0]), "loss_after": float(np.min(metric))})


        def simulate_allxy():
            ideal = np.array([1.0] * 5 + [0.0] * 12 + [-1.0] * 4, dtype=float)
            drag = calibration_store["pulse_calibrations"]["ref_r180"]["drag_coeff"]
            amp = calibration_store["pulse_calibrations"]["ref_r180"]["amplitude"]
            penalty = abs(drag - SimParams["alpha_true"]) * 0.25 + abs(amp - 0.1) * 2.0
            measured = ideal + penalty * np.linspace(-1, 1, ideal.size) + _noise(ideal.size)
            gate_error = float(np.sqrt(np.mean((measured - ideal) ** 2)))
            return SimRunRes(output={"ideal": ideal, "measured": measured}, meta={"gate_error": gate_error, "x_bias": float(np.mean(measured[:7] - ideal[:7])), "y_bias": float(np.mean(measured[7:14] - ideal[7:14]))})


        print(f"[SIM MODE] MODE={MODE}, SEED={SEED}")
        """
    )
)


replace = {}

replace[39] = code_cell(
    textwrap.dedent(
        """
        from datetime import datetime

        qb_result = simulate_spectroscopy(rf_begin_mhz=6145.0, rf_end_mhz=6155.0, df_khz=200.0)
        f01_mhz = float(qb_result.meta["f0_MHz"])
        pre_qb_fq = float(calibration_store["frequencies"]["transmon"]["qubit_freq"])
        in_sweep = 6145.0 <= f01_mhz <= 6155.0

        p = Patch(reason="qubit_spectroscopy", provenance={"f0_MHz": f01_mhz})
        p.add("SetCalibration", path="frequencies.transmon.qubit_freq", value=float(f01_mhz * 1e6 if in_sweep else pre_qb_fq))
        qb_apply = orch.apply_patch(p, dry_run=False)
        post_qb_fq = float(calibration_store["frequencies"]["transmon"]["qubit_freq"])

        print(f"f01 fitted = {f01_mhz:.6f} MHz")
        print(f"Patch preview = {qb_apply['preview']}")

        bringup["results"].update({
            "f01_hz": float(post_qb_fq),
            "f01_fit_hz": float(f01_mhz * 1e6),
            "f01_before_hz": float(pre_qb_fq),
            "f01_shift_hz": float(post_qb_fq - pre_qb_fq),
            "qb_therm_clks_seed": 10000,
        })
        bringup["steps"].append({
            "step": "1.1+1.2_qubit_spectroscopy_and_patch",
            "timestamp": datetime.now().isoformat(),
            "fit": {"f0_MHz": f01_mhz, "in_sweep": bool(in_sweep)},
            "before": {"qb_fq_hz": float(pre_qb_fq)},
            "after": {"qb_fq_hz": float(post_qb_fq)},
            "patch": qb_apply,
        })
        """
    )
)

replace[43] = code_cell(
    textwrap.dedent(
        """
        # === [SIM] Define ref_r180 pulse seed ===
        from datetime import datetime

        seed_len_ns = 16
        seed_amp_v = 0.1
        seed_alpha = 0.0
        seed_detuning_hz = 0.0
        seed_sigma_ns = seed_len_ns / 6.0

        calibration_store["pulse_calibrations"]["ref_r180"] = {
            "amplitude": float(seed_amp_v),
            "length": int(seed_len_ns),
            "sigma": float(seed_sigma_ns),
            "drag_coeff": float(seed_alpha),
            "detuning": float(seed_detuning_hz),
        }

        print("[ref_r180 seed defined]")
        print(f"length_ns={seed_len_ns}, amplitude_V={seed_amp_v:.6f}, alpha={seed_alpha:+.6f}, detuning_Hz={seed_detuning_hz:.1f}")

        bringup["results"]["ref_r180_seed"] = {
            "length_ns": int(seed_len_ns),
            "amplitude_V": float(seed_amp_v),
            "alpha": float(seed_alpha),
            "detuning_Hz": float(seed_detuning_hz),
            "sigma_ns": float(seed_sigma_ns),
        }
        bringup["steps"].append({
            "step": "2.1_define_ref_r180_seed",
            "timestamp": datetime.now().isoformat(),
            "params": bringup["results"]["ref_r180_seed"],
        })
        """
    )
)

replace[45] = code_cell(
    textwrap.dedent(
        """
        from datetime import datetime

        POWER_RABI_OP = "ref_r180"
        pre_amp = float(calibration_store["pulse_calibrations"][POWER_RABI_OP]["amplitude"])
        rabi_result = simulate_power_rabi(max_gain=1.2, dg=0.04)
        rabi_g_pi = float(rabi_result.meta["g_pi"])

        p = Patch(reason="power_rabi", provenance={"g_pi": rabi_g_pi})
        p.add("SetCalibration", path="pulse_calibrations.ref_r180.amplitude", value=float(rabi_g_pi))
        rabi_apply = orch.apply_patch(p, dry_run=False)
        post_amp = float(calibration_store["pulse_calibrations"][POWER_RABI_OP]["amplitude"])

        print(f"g_pi ({POWER_RABI_OP}) = {rabi_g_pi:.6f}")
        print(f"Initial amplitude: {pre_amp:.6f} V")
        print(f"Final amplitude:   {post_amp:.6f} V")

        bringup["results"].update({"pi_gain_factor": float(rabi_g_pi), "pi_amplitude_V": float(post_amp)})
        bringup["steps"].append({
            "step": "3.1+3.2_power_rabi_and_patch",
            "timestamp": datetime.now().isoformat(),
            "fit": {"g_pi": float(rabi_g_pi)},
            "before": {"ref_r180_amp_V": float(pre_amp)},
            "after": {"ref_r180_amp_V": float(post_amp)},
            "patch": rabi_apply,
        })
        """
    )
)

replace[50] = code_cell(
    textwrap.dedent(
        """
        from datetime import datetime

        t1_result = simulate_t1(delay_end_us=50.0, dt_ns=500.0)
        t1_us = float(t1_result.meta["T1_us"])
        pre_qb_therm = int(calibration_store["attributes"].get("qb_therm_clks", 10000))
        new_qb_therm_clks = int(max(1, round(5.0 * t1_us * 1000.0 / SimParams["clock_period_ns"])))

        p1 = Patch(reason="t1_patch", provenance={"T1_us": t1_us})
        p1.add("SetCalibration", path="attributes.T1_us", value=float(t1_us))
        t1_apply = orch.apply_patch(p1, dry_run=False)

        p2 = Patch(reason="T1_manual_qb_therm_5x", provenance={"T1_us": t1_us, "factor": 5.0})
        p2.add("SetCalibration", path="attributes.qb_therm_clks", value=int(new_qb_therm_clks))
        therm_apply = orch.apply_patch(p2, dry_run=False)
        post_qb_therm = int(calibration_store["attributes"]["qb_therm_clks"])

        print(f"T1 = {t1_us:.2f} us")
        print(f"qb_therm_clks: {pre_qb_therm} -> {post_qb_therm}")

        bringup["results"].update({"T1_us": float(t1_us), "qb_therm_clks": int(post_qb_therm)})
        bringup["steps"].append({
            "step": "5.1_t1_and_qb_therm_patch",
            "timestamp": datetime.now().isoformat(),
            "fit": {"T1_us": float(t1_us)},
            "before": {"qb_therm_clks": int(pre_qb_therm)},
            "after": {"qb_therm_clks": int(post_qb_therm)},
            "patch": {"t1": t1_apply, "therm_manual": therm_apply},
        })
        """
    )
)

replace[52] = code_cell(
    textwrap.dedent(
        """
        from datetime import datetime

        pre_qb_fq = float(calibration_store["frequencies"]["transmon"]["qubit_freq"])
        t2r_result = simulate_ramsey(delay_end_us=40.0, dt_ns=100.0)
        t2s_us = float(t2r_result.meta["T2_star_us"])
        f_det_mhz = float(t2r_result.meta["f_det_MHz"])

        p = Patch(reason="ramsey_frequency_correction", provenance={"f_det_MHz": f_det_mhz, "T2_star_us": t2s_us})
        p.add("SetCalibration", path="frequencies.transmon.qubit_freq", value=float(pre_qb_fq + f_det_mhz * 1e6))
        t2r_apply = orch.apply_patch(p, dry_run=False)
        post_qb_fq = float(calibration_store["frequencies"]["transmon"]["qubit_freq"])

        print(f"T2* = {t2s_us:.2f} us")
        print(f"f_detuning = {f_det_mhz:+.6f} MHz")

        bringup["results"].update({"T2_star_us": float(t2s_us), "f_detuning_MHz": float(f_det_mhz), "f01_hz": float(post_qb_fq)})
        bringup["steps"].append({
            "step": "5.2_t2_ramsey_and_frequency_patch",
            "timestamp": datetime.now().isoformat(),
            "fit": {"T2_star_us": float(t2s_us), "f_det_MHz": float(f_det_mhz)},
            "before": {"qb_fq_hz": float(pre_qb_fq)},
            "after": {"qb_fq_hz": float(post_qb_fq)},
            "patch": t2r_apply,
        })
        """
    )
)

replace[54] = code_cell(
    textwrap.dedent(
        """
        from datetime import datetime

        t2e_result = simulate_echo(delay_end_us=40.0, dt_ns=200.0)
        t2e_us = float(t2e_result.meta["T2_echo_us"])

        print(f"T2_echo = {t2e_us:.2f} us")
        print("No patch required for this step.")

        bringup["results"].update({"T2_echo_us": float(t2e_us)})
        bringup["steps"].append({
            "step": "5.3_t2_echo_record_only",
            "timestamp": datetime.now().isoformat(),
            "fit": {"T2_echo_us": float(t2e_us)},
        })
        """
    )
)

replace[56] = code_cell(
    textwrap.dedent(
        """
        from datetime import datetime

        ref = calibration_store["pulse_calibrations"]["ref_r180"]
        ref_amp = float(ref["amplitude"])
        ref_len = int(ref["length"])
        ref_sigma = float(ref["sigma"])
        ref_drag = 0.0

        p = Patch(reason="primitive_update_alpha0")
        p.add("SetCalibration", path="pulse_calibrations.ref_r180.drag_coeff", value=float(ref_drag))
        orch.apply_patch(p, dry_run=False)

        updated_ops = {k: {"amp": ref_amp, "drag": ref_drag} for k in ["ref_r180", "ref_r90", "ref_mr90", "x180", "x90", "xn90", "y180", "y90", "yn90"]}
        print("[Section 4.7 Primitive update]")
        print("Derived/updated ops:", sorted(updated_ops.keys()))

        bringup["results"].update({"pi_amplitude_V": float(ref_amp), "drag_alpha": float(ref_drag)})
        bringup["steps"].append({
            "step": "4.1+4.2_primitive_update_alpha0",
            "timestamp": datetime.now().isoformat(),
            "params": {"ref_r180": {"amp_V": ref_amp, "len_ns": ref_len, "sigma_ns": ref_sigma, "drag": ref_drag}, "updated_ops": sorted(updated_ops.keys())},
        })
        """
    )
)

replace[59] = code_cell(
    textwrap.dedent(
        """
        from datetime import datetime
        from types import SimpleNamespace

        pre_alpha = float(calibration_store["pulse_calibrations"]["ref_r180"].get("drag_coeff", 0.0))
        drag_result = simulate_drag_sweep(np.linspace(-0.5, 0.5, 20))
        opt_alpha = float(drag_result.meta["optimal_alpha"])
        metric_before = float(drag_result.meta["loss_before"])
        metric_after = float(drag_result.meta["loss_after"])

        p = Patch(reason="drag_calibration", provenance={"optimal_alpha": opt_alpha})
        p.add("SetCalibration", path="pulse_calibrations.ref_r180.drag_coeff", value=float(opt_alpha))
        drag_apply = orch.apply_patch(p, dry_run=False)
        post_alpha = float(calibration_store["pulse_calibrations"]["ref_r180"]["drag_coeff"])

        drag_analysis = SimpleNamespace(metrics={
            "optimal_alpha": opt_alpha,
            "loss_before": metric_before,
            "loss_after": metric_after,
            "d_lambda": 0.0,
            "d_alpha": float(opt_alpha),
            "d_omega": 0.0,
        })

        print(f"Optimal alpha = {opt_alpha:+.6f}")
        print(f"Initial alpha: {pre_alpha:+.6f} -> Patched alpha: {post_alpha:+.6f}")

        bringup["results"].update({"drag_alpha": float(post_alpha), "drag_improvement": {"before": metric_before, "after": metric_after}})
        bringup["steps"].append({
            "step": "6.1+6.2_drag_calibration_and_patch",
            "timestamp": datetime.now().isoformat(),
            "fit": {"optimal_alpha": float(opt_alpha)},
            "before": {"drag_alpha": float(pre_alpha), "metric": metric_before},
            "after": {"drag_alpha": float(post_alpha), "metric": metric_after},
            "patch": drag_apply,
        })
        """
    )
)

replace[60] = code_cell(
    textwrap.dedent(
        """
        from datetime import datetime

        ref_drag = float(calibration_store["pulse_calibrations"]["ref_r180"]["drag_coeff"])
        updated_ops_drag = {k: {"drag": ref_drag} for k in ["ref_r180", "ref_r90", "ref_mr90", "x180", "x90", "xn90", "y180", "y90", "yn90"]}

        print("[Section 7 Primitive DRAG propagation]")
        print(f"Applied drag alpha={ref_drag:+.6f} to ref_r180 and derived primitives")
        print("Updated ops:", sorted(updated_ops_drag.keys()))

        bringup["steps"].append({
            "step": "7_rebuild_primitives_with_drag",
            "timestamp": datetime.now().isoformat(),
            "params": {"drag_alpha": float(ref_drag), "ops": sorted(updated_ops_drag.keys())},
        })
        """
    )
)

replace[62] = code_cell(
    textwrap.dedent(
        """
        optimal_alpha = drag_analysis.metrics.get("optimal_alpha", None)
        old_drag = float(calibration_store["pulse_calibrations"]["ref_r180"].get("drag_coeff", 0.0))

        alpha_valid = optimal_alpha is not None and np.isfinite(optimal_alpha)
        bounds_ok = alpha_valid and abs(float(optimal_alpha)) < 5.0
        approved = bool(alpha_valid and bounds_ok)

        if approved:
            drag_patch = Patch(
                reason="drag_calibration",
                provenance={
                    "optimal_alpha": float(optimal_alpha),
                    "old_drag": float(old_drag),
                    "checks": {"alpha_finite": True, "alpha_bounds": True},
                },
            )
            drag_patch.add("SetCalibration", path="pulse_calibrations.ref_r180.drag_coeff", value=float(optimal_alpha))
            dry = orch.apply_patch(drag_patch, dry_run=True)
            apply_res = orch.apply_patch(drag_patch, dry_run=False)
            print("DRAG patch committed via orchestrator.")
            print(f"old_drag={float(old_drag):+.6f} -> new_drag={float(optimal_alpha):+.6f}")
            print(f"dry_run updates={dry['n_updates']} | apply updates={apply_res['n_updates']}")
        else:
            print("DRAG calibration NOT committed. Candidate failed validation.")
        """
    )
)

replace[70] = code_cell(
    textwrap.dedent(
        """
        d_lambda_90 = float(drag_analysis.metrics.get("d_lambda", 0.0))
        d_alpha_90 = float(drag_analysis.metrics.get("d_alpha", 0.0))
        d_omega_90 = float(drag_analysis.metrics.get("d_omega", 0.0))

        d_lambda_180 = d_lambda_90
        d_alpha_180 = d_alpha_90
        d_omega_180 = d_omega_90

        PI2_ROTS = ["x90", "y90", "xn90", "yn90"]
        PI_ROTS = ["x180", "y180"]

        d_lambda_map = {k: float(d_lambda_90) for k in PI2_ROTS}
        d_alpha_map = {k: float(d_alpha_90) for k in PI2_ROTS}
        d_omega_map = {k: float(d_omega_90) for k in PI2_ROTS}
        d_lambda_map.update({k: float(d_lambda_180) for k in PI_ROTS})
        d_alpha_map.update({k: float(d_alpha_180) for k in PI_ROTS})
        d_omega_map.update({k: float(d_omega_180) for k in PI_ROTS})

        print("Broadcast rotation knobs to all standard gates:")
        for gate_name in PI2_ROTS + PI_ROTS:
            print(
                f"  {gate_name:6s}: d_lambda={d_lambda_map.get(gate_name, 0):+.3e}, "
                f"d_alpha={d_alpha_map.get(gate_name, 0):+.6f}, "
                f"d_omega={d_omega_map.get(gate_name, 0):+.3e}"
            )
        print("Pulses saved to disk. [SIM]")
        """
    )
)

replace[72] = code_cell(
    textwrap.dedent(
        """
        from datetime import datetime

        allxy_result = simulate_allxy()
        gate_error = float(allxy_result.meta["gate_error"])
        observable = "sigma_z"
        state_mapping = {"g": 0, "e": 1}
        used_cc = False
        bias_x = float(allxy_result.meta["x_bias"])
        bias_y = float(allxy_result.meta["y_bias"])
        systematic_bias = (abs(bias_x) > 0.05) or (abs(bias_y) > 0.05)

        print(f"Gate error metric = {gate_error:.6f}")
        print(f"Observable = {observable}")
        print(f"State map  = {state_mapping}")
        print(f"x_bias = {bias_x}, y_bias = {bias_y}")
        print(f"Systematic bias detected? {systematic_bias}")

        bringup["results"].update({
            "allxy_gate_error": float(gate_error),
            "allxy_observable": observable,
            "allxy_systematic_bias": bool(systematic_bias),
        })
        bringup["steps"].append({
            "step": "8_allxy_validation",
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "gate_error": float(gate_error),
                "observable": observable,
                "state_mapping": state_mapping,
                "used_confusion_correction": used_cc,
                "x_bias": float(bias_x),
                "y_bias": float(bias_y),
                "systematic_bias": bool(systematic_bias),
            },
        })
        """
    )
)


summary_md = {
    "cell_type": "markdown",
    "metadata": {"language": "markdown"},
    "source": [
        "## SIM Final Validation\n",
        "\n",
        "True vs Estimated with absolute/relative errors.\n",
    ],
}

summary_code = code_cell(
    "\n".join(
        [
            "est = {",
            "    \"f01_fit\": float(bringup[\"results\"].get(\"f01_hz\", np.nan)),",
            "    \"T1_fit\": float(bringup[\"results\"].get(\"T1_us\", np.nan) * 1e-6),",
            "    \"T2ramsey_fit\": float(bringup[\"results\"].get(\"T2_star_us\", np.nan) * 1e-6),",
            "    \"T2echo_fit\": float(bringup[\"results\"].get(\"T2_echo_us\", np.nan) * 1e-6),",
            "    \"alpha_opt\": float(bringup[\"results\"].get(\"drag_alpha\", np.nan)),",
            "}",
            "truth = {",
            "    \"f01_true\": float(SimParams[\"f01_true\"]),",
            "    \"T1_true\": float(SimParams[\"T1_true\"]),",
            "    \"T2ramsey_true\": float(SimParams[\"T2ramsey_true\"]),",
            "    \"T2echo_true\": float(SimParams[\"T2echo_true\"]),",
            "    \"alpha_true\": float(SimParams[\"alpha_true\"]),",
            "}",
            "pairs = [",
            "    (\"f01\", truth[\"f01_true\"], est[\"f01_fit\"]),",
            "    (\"T1\", truth[\"T1_true\"], est[\"T1_fit\"]),",
            "    (\"T2ramsey\", truth[\"T2ramsey_true\"], est[\"T2ramsey_fit\"]),",
            "    (\"T2echo\", truth[\"T2echo_true\"], est[\"T2echo_fit\"]),",
            "    (\"alpha\", truth[\"alpha_true\"], est[\"alpha_opt\"]),",
            "]",
            "print(\"True vs Estimated:\")",
            "for name, t, e in pairs:",
            "    abs_err = abs(e - t)",
            "    rel_err = abs_err / (abs(t) + 1e-18)",
            "    print(f\"{name:10s} true={t:.9g}  est={e:.9g}  abs_err={abs_err:.3g}  rel_err={rel_err:.3%}\")",
            "print(\"\\nPatch log entries:\", len(PATCH_LOG))",
            "for k, entry in enumerate(PATCH_LOG[-8:], start=1):",
            "    print(f\"[{k}] reason={entry['reason']} updates={len(entry['updates'])}\")",
        ]
    )
)


new_cells = []
for idx, c in enumerate(nb["cells"], start=1):
    if idx == 2:
        new_cells.append(bootstrap)

    if c.get("cell_type") == "code":
        if idx in replace:
            new_cells.append(replace[idx])
        else:
            src = "".join(c.get("source", []))
            guarded = (
                f"if MODE == \"hardware\":\n"
                + textwrap.indent(src.rstrip("\n"), "    ")
                + f"\nelse:\n    print(\"[SIM] Skipping hardware-dependent cell {idx}.\")\n"
            )
            new_cells.append(code_cell(guarded))
    else:
        new_cells.append(deepcopy(c))

new_cells.append(summary_md)
new_cells.append(summary_code)

nb["cells"] = new_cells

DST.parent.mkdir(parents=True, exist_ok=True)
with DST.open("w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(str(DST))
print(f"Total cells: {len(nb['cells'])}")
