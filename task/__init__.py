"""Tasks package - modular task definitions for GRPO-MA.

Each module under ``task/`` is auto-discovered by :mod:`task.task_loader`. A
task module exposes ``TASK_CONFIG``, ``register(...)``, and optionally
``compute_metrics`` (consumed by ``scripts/eval/eval.py``). See
``task/TEMPLATE.py`` for a worked example.
"""

import os

from .task_loader import TaskRegistry, get_task_registry, reload_tasks

# Auto-load all tasks when the package is imported
_registry = get_task_registry()

__all__ = [
    'TaskRegistry',
    'get_task_registry',
    'reload_tasks',
    'get_accuracy_reward',
    'get_format_reward',
    'get_answer_processor',
    'get_task_config',
    'list_registered_tasks',
    'print_summary',
]


def get_accuracy_reward(task_type: str):
    return _registry.get_accuracy_reward(task_type)


def get_format_reward(task_type: str):
    return _registry.get_format_reward(task_type)


def get_answer_processor(task_type: str):
    return _registry.get_answer_processor(task_type)


def get_task_config(task_type: str):
    return _registry.get_task_config(task_type)


def list_registered_tasks():
    return _registry.list_registered_tasks()


def print_summary():
    print(_registry.get_summary())


# Print a registry summary at import time only when GRPO_MA_VERBOSE is set.
if os.environ.get("GRPO_MA_VERBOSE", "0").lower() not in ("", "0", "false", "no"):
    print("\n" + _registry.get_summary() + "\n")
