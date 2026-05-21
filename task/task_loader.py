"""Task registry and auto-loader for GRPO-MA.

Every file under ``task/`` that is not in ``_SKIP_MODULES`` is imported once
when the registry is first accessed. A task module participates in the
registry by exposing:

* ``TASK_CONFIG``: a ``dict`` with at least ``task_type`` and ``grpo_template``.
* ``register(accuracy_registry, format_registry, answer_registry=None)``:
  populates the registries with task-specific reward / processing callables.
* (Optional) ``compute_metrics(predictions, references, **kwargs)``: returns a
  ``dict`` of evaluation metrics; consumed by ``scripts/eval/eval.py``.

The registry is intentionally tiny: no plugin discovery, no entry points, no
external dependencies. New tasks become available by dropping a file into
``task/``.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Any, Callable, Dict, Optional

_SKIP_MODULES = {"__init__", "task_loader", "processing_utils", "TEMPLATE"}


class TaskRegistry:
    """Collects task configurations and callables discovered under ``task/``."""

    def __init__(self) -> None:
        self.accuracy_registry: Dict[str, Callable] = {}
        self.format_registry: Dict[str, Callable] = {}
        self.answer_registry: Dict[str, Callable] = {}
        self.metrics_registry: Dict[str, Callable] = {}
        self.config_registry: Dict[str, Dict[str, Any]] = {}
        self.modules: Dict[str, Any] = {}

    def discover(self) -> "TaskRegistry":
        pkg_dir = Path(__file__).parent
        for mod in pkgutil.iter_modules([str(pkg_dir)]):
            if mod.ispkg or mod.name.startswith("_") or mod.name in _SKIP_MODULES:
                continue
            module = importlib.import_module(f"task.{mod.name}")
            self.modules[mod.name] = module

            config = getattr(module, "TASK_CONFIG", None)
            if isinstance(config, dict) and "task_type" in config:
                key = _normalize(config["task_type"])
                self.config_registry[key] = config

            register = getattr(module, "register", None)
            if callable(register):
                register(self.accuracy_registry, self.format_registry, self.answer_registry)

            compute_metrics = getattr(module, "compute_metrics", None)
            if callable(compute_metrics) and isinstance(config, dict):
                self.metrics_registry[_normalize(config["task_type"])] = compute_metrics

        return self

    def get_accuracy_reward(self, task_type: str) -> Optional[Callable]:
        return self.accuracy_registry.get(_normalize(task_type))

    def get_format_reward(self, task_type: str) -> Optional[Callable]:
        return self.format_registry.get(_normalize(task_type))

    def get_answer_processor(self, task_type: str) -> Optional[Callable]:
        return self.answer_registry.get(_normalize(task_type))

    def get_metrics_fn(self, task_type: str) -> Optional[Callable]:
        return self.metrics_registry.get(_normalize(task_type))

    def get_task_config(self, task_type: str) -> Dict[str, Any]:
        return self.config_registry.get(_normalize(task_type), {})

    def list_registered_tasks(self):
        return sorted(self.config_registry.keys())

    def get_summary(self) -> str:
        if not self.config_registry:
            return "Registered tasks: (none)"
        lines = ["Registered tasks:"]
        for name in self.list_registered_tasks():
            cfg = self.config_registry[name]
            desc = cfg.get("description", "")
            lines.append(f"  - {name}: {desc}")
        return "\n".join(lines)


def _normalize(task_type: str) -> str:
    return task_type.lower().strip()


_REGISTRY: Optional[TaskRegistry] = None


def get_task_registry() -> TaskRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = TaskRegistry().discover()
    return _REGISTRY


def reload_tasks() -> TaskRegistry:
    global _REGISTRY
    _REGISTRY = TaskRegistry().discover()
    return _REGISTRY
