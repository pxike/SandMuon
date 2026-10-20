"""
Self-registering optimizer registry with auto-discovery.

How to add a new optimizer
--------------------------
1. Create ``odl/optim/my_optimizer.py``.
2. Define your class/wrapper.
3. Decorate a factory function::

       from odl.optim.registry import register_optimizer

       @register_optimizer("my_optimizer")
       def _create(model, *, lr, betas, weight_decay, device_type):
           return MyOptimizer(model, lr=lr, ...)

That's it.  ``create_optimizer(model, optimizer_type="my_optimizer")`` will
find it automatically — no edits to factory.py or __init__.py needed.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Callable

# ── Registry ──────────────────────────────────────────────────────────────

_REGISTRY: dict[str, Callable[..., Any]] = {}
_discovered = False


def register_optimizer(name: str):
    """Decorator that registers a factory function under *name*."""
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        _REGISTRY[name] = fn
        return fn
    return deco


def _discover_all() -> None:
    """Import every module in ``odl.optim`` so their ``@register_optimizer``
    decorators fire.  Called once, lazily, on first ``create_optimizer``."""
    global _discovered
    if _discovered:
        return
    _discovered = True

    import odl.optim as _pkg

    for info in pkgutil.iter_modules(_pkg.__path__, _pkg.__name__ + "."):
        try:
            importlib.import_module(info.name)
        except Exception:
            pass  # skip broken / optional-dep modules silently


def create_optimizer(
    model,
    lr: float = 3e-4,
    betas: tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0.1,
    optimizer_type: str = "dense",
    device_type: str = "cuda",
):
    """Build an optimizer by name.  Auto-discovers all registered optimizers."""
    _discover_all()
    fn = _REGISTRY.get(optimizer_type.lower())
    if fn is None:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ValueError(
            f"Unknown optimizer_type '{optimizer_type}'. "
            f"Available: {available}"
        )
    return fn(model, lr=lr, betas=betas, weight_decay=weight_decay,
              device_type=device_type)
