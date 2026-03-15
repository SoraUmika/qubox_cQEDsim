from __future__ import annotations

"""Generic holographic quantum algorithms built around bond-space channels.

The package follows the architecture summarized in
`paper_summary/holographic_quantum_algorithms.pdf`:

- a holographic channel / MPS layer
- a purified step embedding
- explicit observable schedules and burn-in controls
- Monte Carlo and exact branch estimators
- diagnostics plus future-facing holographic optimization scaffolding

The implementation is intentionally generic in physical and bond Hilbert-space
dimensions. It lives inside `cqed_sim`, but it is not hardcoded to cQED models.
"""

from .api import *  # noqa: F401,F403
from .api import __all__
