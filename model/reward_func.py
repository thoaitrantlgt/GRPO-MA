"""
Consolidated reward function module - Refactored to use modular task structure

This module now acts as a central router to task-specific reward functions
defined in individual task modules under model/tasks/.
"""

from typing import Callable, Dict

from task import get_task_registry
from model.task_configs import normalize_task_type

# Optional scipy import for .mat files
try:
    from scipy.io import loadmat
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

    def loadmat(filename):
        raise ImportError("scipy is not available")


# Get the global task registry
_task_registry = get_task_registry()

# Legacy registries for backward compatibility
ACCURACY_REWARD_REGISTRY: Dict[str, Callable] = _task_registry.accuracy_registry
FORMAT_REWARD_REGISTRY: Dict[str, Callable] = _task_registry.format_registry


def register_accuracy_reward(task_type: str):
    """Decorator to register accuracy reward functions (for backward compatibility)"""

    def decorator(func: Callable):
        ACCURACY_REWARD_REGISTRY[task_type] = func
        return func

    return decorator


def register_format_reward(task_type: str):
    """Decorator to register format reward functions (for backward compatibility)"""

    def decorator(func: Callable):
        FORMAT_REWARD_REGISTRY[task_type] = func
        return func

    return decorator


def accuracy_reward(completions, solution, question_types, **kwargs):
    """Main accuracy reward function - routes to specific implementations"""
    rewards = []
    debug_metrics = {}  # Collect debug metrics
    kwargs_values = list(kwargs.values())

    for completion, sol, question_type, *other_values in zip(completions, solution, question_types, *kwargs_values):
        other_kwargs = dict(zip(kwargs.keys(), other_values))

        reward_func = ACCURACY_REWARD_REGISTRY.get(normalize_task_type(question_type))
        if reward_func:
            try:
                reward = reward_func(completion, sol, **other_kwargs)
                # Handle case where function returns (reward, metrics)
                if isinstance(reward, tuple):
                    reward, metrics = reward
                    debug_metrics[question_type] = metrics
            except Exception as e:
                print(f"Error in accuracy_reward for {question_type}: {e}")
                reward = 0.0
        else:
            # Unsupported task type
            print(f"Warning: No accuracy reward function for task type '{question_type}'")
            reward = 0.0

        rewards.append(reward)

    # Return rewards and debug metrics
    return rewards, debug_metrics


def format_reward(completions, solution, question_types, **kwargs):
    """Main format reward function - routes to specific implementations"""
    rewards = []
    kwargs_values = list(kwargs.values())

    for completion, sol, question_type, *other_values in zip(completions, solution, question_types, *kwargs_values):
        other_kwargs = dict(zip(kwargs.keys(), other_values))

        reward_func = FORMAT_REWARD_REGISTRY.get(normalize_task_type(question_type))
        if reward_func:
            try:
                reward = reward_func(completion, sol, **other_kwargs)
            except Exception as e:
                print(f"Error in format_reward for {question_type}: {e}")
                reward = 0.0
        else:
            # Unsupported task type
            print(f"Warning: No format reward function for task type '{question_type}'")
            reward = 0.0

        rewards.append(reward)
    return rewards


# ============= Utility Functions =============


def list_registered_functions():
    """List all registered reward functions"""
    return {"accuracy": list(ACCURACY_REWARD_REGISTRY.keys()), "format": list(FORMAT_REWARD_REGISTRY.keys())}


def get_supported_task_types():
    """Get all supported task types"""
    accuracy_types = set(ACCURACY_REWARD_REGISTRY.keys())
    format_types = set(FORMAT_REWARD_REGISTRY.keys())
    return {"accuracy": list(accuracy_types), "format": list(format_types), "all": list(accuracy_types | format_types)}
