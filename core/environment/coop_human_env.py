from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from uuid import uuid4

Formatter = Callable[[Dict[str, Any]], str]
RewardFunc = Callable[..., Union[float, Sequence[float]]]
TransitionFunc = Callable[
    [
        str,
        Sequence[str],
        Sequence[Sequence[str]],
        Sequence[Sequence[str]],
        Dict[str, Any],
    ],
    Sequence[str],
]


@dataclass(slots=True)
class CoopHumanEnvState:
    """CoopHumanEval 单条样本的 episode 状态。"""

    item: Dict[str, Any]
    episode_id: str = field(default_factory=lambda: uuid4().hex)


class CoopHumanEnv:
    """
    CoopHumanEval 专用环境。

    约定：
    - `formatters[i](item)` 构造第 i 个 agent 的首轮 prompt
    - `reward_func(*[[c] for c in completions], batch_items=[item])` 计算联合奖励
    - `transition_fn(prompt, completions, prompt_hist, response_hist, item)`
      返回下一轮每个 agent 的 prompt
    """

    def __init__(
        self,
        *,
        formatters: Sequence[Formatter],
        reward_func: RewardFunc,
        transition_fn: Optional[TransitionFunc] = None,
        num_turns: int = 2,
        reward_processor: Optional[Callable[[float], float]] = None,
    ) -> None:
        if num_turns < 1:
            raise ValueError("num_turns must be >= 1.")
        if not formatters:
            raise ValueError("formatters must not be empty.")
        if reward_func is None or not callable(reward_func):
            raise ValueError("reward_func must be a callable.")
        if num_turns > 1 and transition_fn is None:
            raise ValueError("Multi-turn CHE requires transition_fn.")

        self.formatters = list(formatters)
        self.reward_func = reward_func
        self.transition_fn = transition_fn
        self.num_agents = len(formatters)
        self.num_turns = num_turns
        self.reward_processor = reward_processor or (lambda x: x)

    def reset(self, item: Dict[str, Any]) -> CoopHumanEnvState:
        if not isinstance(item, dict):
            raise TypeError("item must be a dict.")
        return CoopHumanEnvState(item=dict(item))

    def build_prompts(
        self,
        *,
        state: CoopHumanEnvState,
        external_prompts: Optional[Sequence[str]] = None,
    ) -> List[str]:
        if external_prompts is not None:
            prompts = list(external_prompts)
        else:
            prompts = [formatter(state.item) for formatter in self.formatters]
        self._validate_prompts(prompts)
        return prompts

    def compute_reward(
        self,
        *,
        state: CoopHumanEnvState,
        agent_completions: Sequence[str],
    ) -> float:
        completion_args = [[completion] for completion in agent_completions]
        raw = self.reward_func(*completion_args, batch_items=[state.item])
        if isinstance(raw, (list, tuple)):
            value = float(raw[0] if raw else 0.0)
        else:
            value = float(raw)
        return float(self.reward_processor(value))

    def transition(
        self,
        *,
        state: CoopHumanEnvState,
        agent_completions: Sequence[str],
        prompt_history_per_agent: Sequence[Sequence[str]],
        response_history_per_agent: Sequence[Sequence[str]],
    ) -> List[str]:
        if self.transition_fn is None:
            raise ValueError("transition_fn is required for multi-turn CHE.")

        prompts = list(
            self.transition_fn(
                state.item.get("prompt", ""),
                list(agent_completions),
                [list(values) for values in prompt_history_per_agent],
                [list(values) for values in response_history_per_agent],
                state.item,
            )
        )
        self._validate_prompts(prompts)
        return prompts

    def should_terminate(
        self,
        *,
        turn_idx: int,
        rewards: Sequence[float],
        early_termination_threshold: Optional[float],
    ) -> bool:
        if turn_idx >= self.num_turns - 1:
            return True
        if early_termination_threshold is None or not rewards:
            return False
        mean_reward = sum(rewards) / len(rewards)
        return mean_reward > float(early_termination_threshold)

    def _validate_prompts(self, prompts: Sequence[str]) -> None:
        if len(prompts) != self.num_agents:
            raise ValueError(
                f"Expected {self.num_agents} prompts, got {len(prompts)}."
            )
        if any(not isinstance(prompt, str) for prompt in prompts):
            raise TypeError("All prompts must be strings.")
