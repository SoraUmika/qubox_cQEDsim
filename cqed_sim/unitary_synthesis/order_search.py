"""Gate-order search for unitary synthesis.

This module provides :class:`GateOrderOptimizer`, which wraps
:class:`~cqed_sim.unitary_synthesis.UnitarySynthesizer` and tries multiple gate
orderings drawn from a user-supplied pool.  The ordering that achieves the
lowest final objective value is returned.

Typical usage::

    from cqed_sim.unitary_synthesis import (
        GateOrderConfig,
        GateOrderOptimizer,
        make_gate_from_matrix,
    )
    import numpy as np

    pool = [
        make_gate_from_matrix("H", np.array([[1,1],[1,-1]])/np.sqrt(2), duration=50e-9),
        make_gate_from_matrix("T", np.diag([1, np.exp(1j*np.pi/4)]), duration=50e-9),
        make_gate_from_matrix("S", np.diag([1, 1j]), duration=50e-9),
    ]

    optimizer = GateOrderOptimizer(
        gate_pool=pool,
        order_config=GateOrderConfig(
            search_strategy="random",
            n_random_trials=15,
            max_sequence_length=4,
        ),
        synthesizer_kwargs=dict(subspace=my_subspace),
    )
    result = optimizer.search(target=my_target)
    print("Best infidelity:", result.best_result.objective)
    print("Best sequence:", [g.name for g in result.best_ordering])
"""
from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .sequence import GateBase


@dataclass(frozen=True)
class GateOrderConfig:
    """Configuration for gate-order search.

    Args:
        max_sequence_length: Maximum number of gates in any candidate sequence
            (default 6).
        min_sequence_length: Minimum number of gates (default 1).
        allow_repetitions: If ``True`` (default), the same gate from the pool
            may appear more than once in a candidate sequence.
        search_strategy: How to generate candidate orderings:

            * ``"random"`` (default) — sample *n_random_trials* random
              orderings.
            * ``"exhaustive"`` — enumerate all permutations up to
              *max_sequence_length* (may be expensive for large pools).
            * ``"greedy"`` — iteratively append the gate from *gate_pool*
              that most improves the objective.

        n_random_trials: Number of random orderings to try when
            ``search_strategy="random"`` (default 20).
        seed: Global RNG seed for random and greedy strategies (default 0).
        early_stop_infidelity: Stop searching once any result reaches this
            infidelity (default 1e-6).  Set to 0 to disable early stopping.
    """

    max_sequence_length: int = 6
    min_sequence_length: int = 1
    allow_repetitions: bool = True
    search_strategy: str = "random"
    n_random_trials: int = 20
    seed: int = 0
    early_stop_infidelity: float = 1.0e-6

    def __post_init__(self) -> None:
        if int(self.max_sequence_length) < 1:
            raise ValueError("max_sequence_length must be at least 1.")
        if int(self.min_sequence_length) < 1:
            raise ValueError("min_sequence_length must be at least 1.")
        if int(self.min_sequence_length) > int(self.max_sequence_length):
            raise ValueError("min_sequence_length must not exceed max_sequence_length.")
        if str(self.search_strategy) not in {"exhaustive", "random", "greedy"}:
            raise ValueError("search_strategy must be 'exhaustive', 'random', or 'greedy'.")
        if int(self.n_random_trials) < 1:
            raise ValueError("n_random_trials must be at least 1.")


@dataclass
class GateOrderSearchResult:
    """Result of a :class:`GateOrderOptimizer` run.

    Attributes:
        best_result: The synthesis result with the lowest objective value.
        best_ordering: The gate list (deep copies) that produced
            ``best_result``.
        all_results: All ``(ordering, SynthesisResult)`` pairs tried,
            sorted by objective (ascending).
        n_orderings_tried: Total number of orderings evaluated (including
            failed ones).
        order_config: The :class:`GateOrderConfig` used.
    """

    best_result: Any  # SynthesisResult
    best_ordering: list[GateBase]
    all_results: list[tuple[list[GateBase], Any]] = field(default_factory=list)
    n_orderings_tried: int = 0
    order_config: GateOrderConfig = field(default_factory=GateOrderConfig)


def _build_index_lists(
    n: int,
    config: GateOrderConfig,
    rng: np.random.Generator,
) -> list[list[int]]:
    """Return lists of gate-pool indices for each candidate ordering."""
    min_len = int(config.min_sequence_length)
    max_len = int(config.max_sequence_length)

    if config.search_strategy == "exhaustive":
        orderings: list[list[int]] = []
        for length in range(min_len, max_len + 1):
            if config.allow_repetitions:
                for combo in itertools.product(range(n), repeat=length):
                    orderings.append(list(combo))
            else:
                for perm in itertools.permutations(range(n), r=min(length, n)):
                    orderings.append(list(perm))
        return orderings

    if config.search_strategy == "random":
        seen: set[tuple[int, ...]] = set()
        orderings = []
        max_attempts = int(config.n_random_trials) * 20
        attempt = 0
        while len(orderings) < int(config.n_random_trials) and attempt < max_attempts:
            attempt += 1
            length = int(rng.integers(min_len, max_len + 1))
            if config.allow_repetitions:
                seq = tuple(int(x) for x in rng.integers(0, n, size=length))
            else:
                actual_len = min(length, n)
                seq = tuple(int(x) for x in rng.choice(n, size=actual_len, replace=False))
            if seq not in seen:
                seen.add(seq)
                orderings.append(list(seq))
        return orderings

    # Greedy is handled separately in GateOrderOptimizer._greedy_search.
    return []


class GateOrderOptimizer:
    """Search over gate orderings in addition to gate parameters.

    For each candidate ordering drawn from *gate_pool*, a fresh
    :class:`~cqed_sim.unitary_synthesis.UnitarySynthesizer` is instantiated
    with that ordering as the ``primitives`` argument, and synthesis is run to
    convergence.  The ordering that achieves the lowest final objective is
    returned.

    Args:
        gate_pool: List of :class:`~cqed_sim.unitary_synthesis.GateBase`
            instances available for sequencing.
        order_config: :class:`GateOrderConfig` controlling the search strategy
            and sequence length constraints.
        synthesizer_kwargs: Keyword arguments forwarded verbatim to each
            :class:`~cqed_sim.unitary_synthesis.UnitarySynthesizer` constructor
            call.  May include ``target``, ``subspace``, ``backend``,
            ``model``, etc.

    Note:
        Gate instances in *gate_pool* are deep-copied for each candidate
        ordering, so modifying the pool after construction has no effect.
    """

    def __init__(
        self,
        gate_pool: list[GateBase],
        *,
        order_config: GateOrderConfig | None = None,
        synthesizer_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if not gate_pool:
            raise ValueError("gate_pool must contain at least one gate.")
        self.gate_pool = list(gate_pool)
        self.order_config = order_config if order_config is not None else GateOrderConfig()
        self.synthesizer_kwargs = dict(synthesizer_kwargs) if synthesizer_kwargs is not None else {}

    def search(
        self,
        target: Any | None = None,
    ) -> GateOrderSearchResult:
        """Run the gate-order search.

        Args:
            target: Synthesis target.  If ``None``, the target must be
                provided via ``synthesizer_kwargs["target"]``.

        Returns:
            A :class:`GateOrderSearchResult` with the best ordering and all
            individual results sorted by objective (ascending).

        Raises:
            ValueError: If no target is available.
            RuntimeError: If all orderings fail to converge.
        """
        if target is None and "target" not in self.synthesizer_kwargs:
            raise ValueError(
                "GateOrderOptimizer.search() requires a target. "
                "Pass target= directly or include 'target' in synthesizer_kwargs."
            )

        rng = np.random.default_rng(self.order_config.seed)
        kwargs = dict(self.synthesizer_kwargs)
        if target is not None:
            kwargs["target"] = target

        if self.order_config.search_strategy == "greedy":
            return self._greedy_search(kwargs, rng)

        n = len(self.gate_pool)
        index_lists = _build_index_lists(n, self.order_config, rng)
        if not index_lists:
            raise RuntimeError(
                "No orderings were generated. "
                "Check gate_pool size and GateOrderConfig constraints."
            )

        all_results: list[tuple[list[GateBase], Any]] = []
        n_tried = 0
        for indices in index_lists:
            ordering = [copy.deepcopy(self.gate_pool[i]) for i in indices]
            result = self._run_one(ordering, kwargs)
            n_tried += 1
            if result is not None:
                all_results.append((ordering, result))
                if float(result.objective) <= float(self.order_config.early_stop_infidelity):
                    break

        if not all_results:
            raise RuntimeError(
                "All orderings failed to produce a valid synthesis result. "
                "Check that synthesizer_kwargs are correct and the gate pool is compatible with the target."
            )

        all_results.sort(key=lambda x: float(x[1].objective))
        best_ordering, best_result = all_results[0]

        return GateOrderSearchResult(
            best_result=best_result,
            best_ordering=best_ordering,
            all_results=all_results,
            n_orderings_tried=n_tried,
            order_config=self.order_config,
        )

    def _run_one(
        self,
        ordering: list[GateBase],
        kwargs: dict[str, Any],
    ) -> Any:
        """Run synthesis for one ordering; return None on failure."""
        # Import here to avoid circular imports at module load time.
        from .optim import UnitarySynthesizer  # noqa: PLC0415

        try:
            synth = UnitarySynthesizer(primitives=ordering, **kwargs)
            return synth.fit()
        except Exception:
            return None

    def _greedy_search(
        self,
        kwargs: dict[str, Any],
        rng: np.random.Generator,  # noqa: ARG002
    ) -> GateOrderSearchResult:
        """Greedily extend the sequence by appending the best gate at each step."""
        max_len = int(self.order_config.max_sequence_length)
        min_len = int(self.order_config.min_sequence_length)

        current: list[GateBase] = []
        all_results: list[tuple[list[GateBase], Any]] = []
        best_result: Any = None
        best_ordering: list[GateBase] = []
        n_tried = 0

        for _step in range(max_len):
            step_best_result: Any = None
            step_best_gate: GateBase | None = None
            step_best_ordering: list[GateBase] = []

            for gate in self.gate_pool:
                candidate = current + [copy.deepcopy(gate)]
                result = self._run_one(candidate, kwargs)
                n_tried += 1
                if result is None:
                    continue
                all_results.append((list(candidate), result))
                if step_best_result is None or float(result.objective) < float(step_best_result.objective):
                    step_best_result = result
                    step_best_gate = copy.deepcopy(gate)
                    step_best_ordering = list(candidate)

            if step_best_gate is None:
                break  # No gate improved anything; stop growing.

            current.append(step_best_gate)

            if len(current) >= min_len:
                if best_result is None or float(step_best_result.objective) < float(best_result.objective):  # type: ignore[union-attr]
                    best_result = step_best_result
                    best_ordering = list(step_best_ordering)

            if (
                step_best_result is not None
                and float(step_best_result.objective) <= float(self.order_config.early_stop_infidelity)
            ):
                break  # Early exit: already good enough.

        if best_result is None:
            raise RuntimeError(
                "Greedy search produced no valid result. "
                "Ensure the gate pool and target are compatible."
            )

        all_results.sort(key=lambda x: float(x[1].objective))
        return GateOrderSearchResult(
            best_result=best_result,
            best_ordering=best_ordering,
            all_results=all_results,
            n_orderings_tried=n_tried,
            order_config=self.order_config,
        )


__all__ = [
    "GateOrderConfig",
    "GateOrderOptimizer",
    "GateOrderSearchResult",
]
