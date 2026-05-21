"""GRPO-MA modeling module: VLM wrappers, monkey patches, and reward routing."""
from .qwen_module import Qwen2VLModule
from .reward_func import (
    ACCURACY_REWARD_REGISTRY,
    FORMAT_REWARD_REGISTRY,
    accuracy_reward,
    format_reward,
    get_supported_task_types,
    list_registered_functions,
    register_accuracy_reward,
    register_format_reward,
)
from .vlm_module import VLMBaseModule

# Import task configs if available
try:
    from .task_configs import (
        TASK_TYPE_CONFIGS,
        get_question_template,
        get_task_config,
        list_all_task_types,
        normalize_task_type,
    )
except ImportError:
    pass

# Import extensibility tools if needed
try:
    from .extensibility_guide import (
        CommonEvaluators,
        CommonFormatCheckers,
        TaskTypeRegistry,
        create_simple_accuracy_reward,
        create_simple_format_reward,
        task_registry,
    )

    EXTENSIBILITY_AVAILABLE = True
except ImportError:
    EXTENSIBILITY_AVAILABLE = False

__all__ = [
    "accuracy_reward",
    "format_reward",
    "register_accuracy_reward",
    "register_format_reward",
    "list_registered_functions",
    "get_supported_task_types",
    "Qwen2VLModule",
    "VLMBaseModule",
]
