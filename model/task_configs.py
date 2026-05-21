"""
Task type configuration file - Refactored to use modular task structure

This module now loads configurations from individual task modules
defined in model/tasks/.
"""

from typing import Any, Dict, List
from task import get_task_registry

# Get the global task registry
_task_registry = get_task_registry()


def normalize_task_type(task_type: str) -> str:
    """Normalize task type string"""
    return task_type.lower().strip()


def get_task_config(task_type: str) -> Dict[str, Any]:
    """Get configuration for a specific task type"""
    normalized_type = normalize_task_type(task_type)
    return _task_registry.get_task_config(normalized_type)


def get_question_template(task_type: str) -> str:
    """Get question template for a specific task type"""
    config = get_task_config(task_type)
    return config.get('grpo_template', 'N/A')


def get_all_task_configs() -> Dict[str, Dict[str, Any]]:
    """Get all task configurations"""
    return _task_registry.config_registry


def list_task_types() -> List[str]:
    """List all available task types"""
    return list(_task_registry.config_registry.keys())


def get_task_description(task_type: str) -> str:
    """Get description for a task type"""
    config = get_task_config(task_type)
    return config.get('description', 'No description available')


def get_task_format(task_type: str) -> Dict[str, str]:
    """Get input/output format for a task type"""
    config = get_task_config(task_type)
    return {'input_format': config.get('input_format', 'N/A'), 'output_format': config.get('output_format', 'N/A')}


def get_task_metrics(task_type: str) -> List[str]:
    """Get evaluation metrics for a task type"""
    config = get_task_config(task_type)
    return config.get('evaluation_metrics', [])


def get_format_requirements(task_type: str) -> Dict[str, Any]:
    """Get format requirements for a task type"""
    config = get_task_config(task_type)
    return config.get('format_requirements', {})


def print_task_summary(task_type: str = None):
    """Print summary of task configuration(s)"""
    if task_type:
        # Print single task
        config = get_task_config(task_type)
        if config:
            print(f"\n{'=' * 60}")
            print(f"Task Type: {task_type}")
            print(f"{'=' * 60}")
            print(f"Description: {config.get('description', 'N/A')}")
            print(f"Input Format: {config.get('input_format', 'N/A')}")
            print(f"Output Format: {config.get('output_format', 'N/A')}")
            print(f"Metrics: {', '.join(config.get('evaluation_metrics', []))}")
            print(f"{'=' * 60}\n")
        else:
            print(f"Task type '{task_type}' not found")
    else:
        # Print all tasks
        print(_task_registry.get_summary())


# Legacy compatibility - create a TASK_CONFIGS dict for backward compatibility
TASK_CONFIGS = _task_registry.config_registry
