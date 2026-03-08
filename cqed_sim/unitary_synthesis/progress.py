from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
import pandas as pd

PROGRESS_SCHEMA_VERSION = 1


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


@dataclass(frozen=True)
class ProgressEvent:
    run_id: str
    iteration: int
    timestamp: float
    objective_total: float
    objective_terms: dict[str, Any]
    metrics: dict[str, Any]
    best_so_far: dict[str, Any]
    params_summary: dict[str, Any]
    backend: str
    solver_stats: dict[str, Any]
    progress_schema_version: int = PROGRESS_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))


class ProgressReporter:
    def on_start(self, meta: dict[str, Any]) -> None:
        return None

    def on_event(self, event: dict[str, Any]) -> None:
        return None

    def on_end(self, summary: dict[str, Any]) -> None:
        return None


class NullReporter(ProgressReporter):
    pass


class CompositeReporter(ProgressReporter):
    def __init__(self, reporters: Iterable[ProgressReporter] | None = None):
        self.reporters = [r for r in list(reporters or []) if r is not None]

    def on_start(self, meta: dict[str, Any]) -> None:
        for reporter in self.reporters:
            reporter.on_start(meta)

    def on_event(self, event: dict[str, Any]) -> None:
        for reporter in self.reporters:
            reporter.on_event(event)

    def on_end(self, summary: dict[str, Any]) -> None:
        for reporter in self.reporters:
            reporter.on_end(summary)


class HistoryReporter(ProgressReporter):
    def __init__(self) -> None:
        self._starts: list[dict[str, Any]] = []
        self._events: list[dict[str, Any]] = []
        self._ends: list[dict[str, Any]] = []

    def on_start(self, meta: dict[str, Any]) -> None:
        self._starts.append(_jsonify(meta))

    def on_event(self, event: dict[str, Any]) -> None:
        self._events.append(_jsonify(event))

    def on_end(self, summary: dict[str, Any]) -> None:
        self._ends.append(_jsonify(summary))

    @property
    def events(self) -> list[dict[str, Any]]:
        return list(self._events)

    @property
    def starts(self) -> list[dict[str, Any]]:
        return list(self._starts)

    @property
    def ends(self) -> list[dict[str, Any]]:
        return list(self._ends)

    @property
    def history_by_run(self) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for event in self._events:
            grouped.setdefault(str(event["run_id"]), []).append(event)
        for run_id in grouped:
            grouped[run_id] = sorted(grouped[run_id], key=lambda row: int(row["iteration"]))
        return grouped

    def to_dataframe(self) -> pd.DataFrame:
        if not self._events:
            return pd.DataFrame()
        return pd.json_normalize(self._events, sep=".")


class JupyterLiveReporter(HistoryReporter):
    def __init__(self, what: str = "objective_total", fidelity_what: str = "metrics.fidelity_subspace", print_every: int = 10):
        super().__init__()
        self.what = what
        self.fidelity_what = fidelity_what
        self.print_every = max(1, int(print_every))
        self._display_ready = True
        self._last_render_count = 0
        try:
            from IPython.display import clear_output, display  # type: ignore

            self._clear_output = clear_output
            self._display = display
        except Exception:
            self._display_ready = False
            self._clear_output = None
            self._display = None

    def on_event(self, event: dict[str, Any]) -> None:
        super().on_event(event)
        if int(event["iteration"]) % self.print_every == 0:
            print(
                f"[{event['run_id']}] iter {event['iteration']} | "
                f"best fidelity={float(event['metrics'].get('fidelity_subspace', 0.0)):.4f} | "
                f"leakage_worst={float(event['metrics'].get('leakage_worst', 0.0)):.3e} | "
                f"cost={float(event['objective_total']):.6g}"
            )
        if self._display_ready:
            self._render()

    def on_end(self, summary: dict[str, Any]) -> None:
        super().on_end(summary)
        if self._display_ready:
            self._render(force=True)

    def _render(self, force: bool = False) -> None:
        if not self._display_ready:
            return
        if not force and self._last_render_count == len(self._events):
            return
        self._last_render_count = len(self._events)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        plot_history(self.history_by_run, what=self.what, ax=axes[0], title="Objective")
        plot_history(self.history_by_run, what=self.fidelity_what, ax=axes[1], title="Subspace Fidelity")
        fig.tight_layout()
        assert self._clear_output is not None
        assert self._display is not None
        self._clear_output(wait=True)
        self._display(fig)
        plt.close(fig)


def history_to_dataframe(history: list[dict[str, Any]] | dict[str, list[dict[str, Any]]]) -> pd.DataFrame:
    if isinstance(history, dict):
        rows: list[dict[str, Any]] = []
        for run_rows in history.values():
            rows.extend(run_rows)
    else:
        rows = list(history)
    if not rows:
        return pd.DataFrame()
    return pd.json_normalize(rows, sep=".")


def plot_history(
    history: list[dict[str, Any]] | dict[str, list[dict[str, Any]]],
    what: str = "objective_total",
    group_by: str = "run_id",
    ax: Any | None = None,
    title: str | None = None,
) -> Any:
    df = history_to_dataframe(history)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    if df.empty or what not in df.columns:
        ax.set_title(title or what)
        ax.set_xlabel("iteration")
        ax.set_ylabel(what)
        return ax

    if group_by not in df.columns:
        df[group_by] = "run_000"

    for run_id, run_df in df.sort_values(["timestamp", "iteration"]).groupby(group_by):
        ax.plot(run_df["iteration"], run_df[what], alpha=0.35, linewidth=1.0, label=str(run_id))

    ordered = df.sort_values("timestamp")
    if "objective" in what or "cost" in what:
        best = ordered[what].cummin()
    else:
        best = ordered[what].cummax()
    ax.plot(ordered["iteration"], best, color="black", linewidth=2.0, label="best")
    ax.set_xlabel("iteration")
    ax.set_ylabel(what)
    ax.set_title(title or what)
    return ax


def save_history_json(history: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonify(history), indent=2), encoding="utf-8")
    return path


def save_history_csv(history: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    history_to_dataframe(history).to_csv(path, index=False)
    return path
