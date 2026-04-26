from __future__ import annotations

from typing import Any, Mapping


_MISSING = object()


def _validate_nsteps(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("QuTiP solver option 'nsteps' must be a positive integer.")
    nsteps = int(value)
    if nsteps <= 0:
        raise ValueError("QuTiP solver option 'nsteps' must be a positive integer.")
    return nsteps


def _values_match(left: Any, right: Any) -> bool:
    try:
        return bool(left == right)
    except Exception:
        return False


def merge_qutip_solver_options(
    base_options: Mapping[str, Any] | None = None,
    solver_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge project-level QuTiP options with caller-provided escape hatches."""

    merged = dict(base_options or {})
    if "nsteps" in merged:
        merged["nsteps"] = _validate_nsteps(merged["nsteps"])

    for key, value in dict(solver_options or {}).items():
        normalized = _validate_nsteps(value) if key == "nsteps" else value
        if key in merged:
            current = merged[key]
            current_normalized = _validate_nsteps(current) if key == "nsteps" else current
            if not _values_match(current_normalized, normalized):
                raise ValueError(
                    f"QuTiP solver option '{key}' is set both by a config field "
                    "and by solver_options with different values."
                )
            merged[key] = current_normalized
            continue
        merged[key] = normalized
    return merged


def build_qutip_solver_options(
    *,
    atol: float | object = _MISSING,
    rtol: float | object = _MISSING,
    max_step: float | None | object = _MISSING,
    nsteps: int | None | object = _MISSING,
    store_states: bool | None | object = _MISSING,
    store_final_state: bool | object = _MISSING,
    progress_bar: str | bool | object = _MISSING,
    solver_options: Mapping[str, Any] | None = None,
    extra_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build QuTiP solver options from common cqed_sim config fields."""

    base: dict[str, Any] = {}
    if atol is not _MISSING and atol is not None:
        base["atol"] = float(atol)
    if rtol is not _MISSING and rtol is not None:
        base["rtol"] = float(rtol)
    if max_step is not _MISSING and max_step is not None:
        base["max_step"] = float(max_step)
    if nsteps is not _MISSING and nsteps is not None:
        base["nsteps"] = _validate_nsteps(nsteps)
    if store_states is not _MISSING:
        base["store_states"] = store_states
    if store_final_state is not _MISSING:
        base["store_final_state"] = bool(store_final_state)
    if progress_bar is not _MISSING:
        base["progress_bar"] = progress_bar
    if extra_options:
        base = merge_qutip_solver_options(base, extra_options)
    return merge_qutip_solver_options(base, solver_options)


__all__ = ["build_qutip_solver_options", "merge_qutip_solver_options"]
