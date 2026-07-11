from __future__ import annotations

import ast
from typing import Any, Callable, Dict, List, Sequence

from core.experiment.spec import Formatter, RewardFunc, TransitionFunc


def build_experiment_dataset() -> List[Dict[str, object]]:
    """
    默认 CHE 实验任务集。

    说明：
    - 这里先保留为默认内置任务集
    - 后续如果你想让用户切换不同任务集，可以继续增加：
      - build_python_api_dataset()
      - build_debug_dataset()
      - build_data_cleaning_dataset()
      等等
    """
    return [
        {
            "id": "task-1",
            "prompt": (
                "Implement a Python function `sum_list(nums)` that returns the total sum of all "
                "numbers in the input list. Also define a helper function `add(a, b)` and use it "
                "inside `sum_list`."
            ),
            "entry_function": "sum_list",
            "required_helpers": ["add"],
            "reward_keywords": [
                "def sum_list",
                "def add",
                "for",
                "return",
            ],
        },
        {
            "id": "task-2",
            "prompt": (
                "Implement a Python function `bubble_sort(arr)` that sorts a list of integers in "
                "ascending order using the bubble sort algorithm and returns the sorted list. "
                "Do not use Python built-in sorting functions. Also define a helper function "
                "`swap(arr, i, j)` and use it during sorting."
            ),
            "entry_function": "bubble_sort",
            "required_helpers": ["swap"],
            "reward_keywords": [
                "def bubble_sort",
                "def swap",
                "for",
                "range",
                "if",
                "return",
            ],
        },
        {
            "id": "task-3",
            "prompt": (
                "Implement a Python function `calc_prime_sum(limit)` that returns the sum of all "
                "prime numbers from 2 to the given limit. Also define a helper function "
                "`is_prime(n)` to check whether a number is prime, and use it inside "
                "`calc_prime_sum`."
            ),
            "entry_function": "calc_prime_sum",
            "required_helpers": ["is_prime"],
            "reward_keywords": [
                "def calc_prime_sum",
                "def is_prime",
                "for",
                "if",
                "return",
            ],
        },
    ]


def build_reward_function() -> RewardFunc:
    """
    默认联合奖励函数。

    设计目标：
    - 奖励保持可解释
    - 不依赖执行环境
    - 可作为默认模块直接复用
    - 后续用户有专属奖励需求时，可在 experiment 层自行替换
    """

    def reward_func(*completion_args, batch_items=None):
        item = (batch_items or [{}])[0]
        helper_code = completion_args[0][0] if len(completion_args) > 0 and completion_args[0] else ""
        main_code = completion_args[1][0] if len(completion_args) > 1 and completion_args[1] else ""

        required_helpers = [str(name) for name in item.get("required_helpers", [])]
        entry_function = str(item.get("entry_function", ""))
        reward_keywords = [str(keyword) for keyword in item.get("reward_keywords", [])]

        score = 0.0
        max_score = 10.0 + len(required_helpers) * 1.5 + len(reward_keywords) * 0.3

        helper_tree = _safe_parse(helper_code)
        main_tree = _safe_parse(main_code)
        joint_tree = _safe_parse(f"{helper_code}\n\n{main_code}".strip())

        if helper_tree is not None:
            score += 1.5
        if main_tree is not None:
            score += 1.5
        if joint_tree is not None:
            score += 1.0

        helper_defs = _top_level_function_names(helper_tree)
        main_defs = _top_level_function_names(main_tree)
        main_calls = _called_function_names(main_tree)
        joint_calls = _called_function_names(joint_tree)

        for helper_name in required_helpers:
            if helper_name in helper_defs:
                score += 1.0
            if helper_name in main_calls or helper_name in joint_calls:
                score += 0.75

        if entry_function and entry_function in main_defs:
            score += 1.25

        if "return" in main_code:
            score += 0.4

        if any(token in main_code for token in ["for ", "sorted(", "dict", "set("]):
            score += 0.5

        if any(token in helper_code for token in ["def ", "return", "if ", "for "]):
            score += 0.5

        joined = f"{helper_code}\n{main_code}"
        for keyword in reward_keywords:
            if keyword in joined:
                score += 0.3

        if main_tree is not None and required_helpers and any(name in main_calls for name in required_helpers):
            score += 0.8

        if helper_tree is not None and len(helper_defs) >= len(required_helpers):
            score += 0.6

        normalized = score / max(max_score, 1.0)
        return float(max(0.0, min(normalized, 1.0)))

    return reward_func


def build_transition_function() -> TransitionFunc:
    """
    默认多轮协作 transition 函数。
    """

    def transition_fn(prompt, completions, prompt_hist, response_hist, item):
        helper_code = completions[0] if len(completions) > 0 else ""
        main_code = completions[1] if len(completions) > 1 else ""
        required_helpers = ", ".join(str(name) for name in item.get("required_helpers", []))
        entry_function = str(item.get("entry_function", ""))

        return [
            "\n".join(
                [
                    f"Original task:\n{item.get('prompt', prompt)}",
                    f"Main draft from Agent B:\n{main_code}",
                    f"Required helper functions: {required_helpers}",
                    "Revise your helper-only code so it contains valid Python helper functions with clear names and useful control flow.",
                ]
            ),
            "\n".join(
                [
                    f"Original task:\n{item.get('prompt', prompt)}",
                    f"Helper draft from Agent A:\n{helper_code}",
                    f"Your API must define `{entry_function}` and call these helpers when appropriate: {required_helpers}.",
                    "Revise your main-only code so it stays executable, calls helper functions explicitly, and returns the final result.",
                ]
            ),
        ]

    return transition_fn


def build_formatters() -> Sequence[Formatter]:
    """
    默认 CHE 双 agent 首轮 prompt 构造器。
    """

    def formatter_agent_0(item: Dict[str, object]) -> str:
        helper_names = ", ".join(str(name) for name in item.get("required_helpers", []))
        return "\n".join(
            [
                "You are Agent A, a helper-function specialist.",
                "Write only helper functions in valid Python.",
                "Do not write the final public API function.",
                f"You should strongly prefer these helper names: {helper_names}.",
                f"Task:\n{item['prompt']}",
            ]
        )

    def formatter_agent_1(item: Dict[str, object]) -> str:
        helper_names = ", ".join(str(name) for name in item.get("required_helpers", []))
        return "\n".join(
            [
                "You are Agent B, a main-function integrator.",
                "Write only the public API function in valid Python.",
                "Do not redefine helper functions unless absolutely necessary.",
                f"Your function must call helper functions such as: {helper_names}.",
                f"Task:\n{item['prompt']}",
            ]
        )

    return [formatter_agent_0, formatter_agent_1]


def build_default_che_components() -> Dict[str, Any]:
    """
    返回一整套默认 CHE 实验组件，方便 experiment 层一次性取用。

    用法示例：
        components = build_default_che_components()
        dataset = components["dataset"]
        reward_func = components["reward_func"]
    """
    return {
        "dataset": build_experiment_dataset(),
        "reward_func": build_reward_function(),
        "transition_fn": build_transition_function(),
        "formatters": build_formatters(),
    }


def build_task_structured_reward_function() -> RewardFunc:
    """
    当前默认奖励函数的语义别名。

    这样做的意义：
    - 保留更清晰的业务语义名
    - 后续你如果再引入别的奖励函数，不会只能叫 build_reward_function
    """
    return build_reward_function()


def build_python_api_dataset() -> List[Dict[str, object]]:
    """
    当前默认任务集的语义别名。

    后续如果增加别的数据集，这个名字会更稳定。
    """
    return build_experiment_dataset()


def _safe_parse(code: str) -> ast.AST | None:
    try:
        return ast.parse(str(code or ""))
    except SyntaxError:
        return None


def _top_level_function_names(tree: ast.AST | None) -> set[str]:
    if tree is None:
        return set()
    return {
        str(node.name)
        for node in getattr(tree, "body", [])
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _called_function_names(tree: ast.AST | None) -> set[str]:
    if tree is None:
        return set()

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(str(node.func.id))
            elif isinstance(node.func, ast.Attribute):
                names.add(str(node.func.attr))
    return names