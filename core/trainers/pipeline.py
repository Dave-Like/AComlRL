from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence

from core.common.types import RolloutBatch
from core.config.config import BaseMultiAgentConfig
from core.environment.che_rollout import CHEEpisodeRollout


class PipelineManager:
    """
    第一阶段的数据流水线控制层。

    当前只承担两类职责：
    1. 屏蔽 rollout 组件的直接调用细节；
    2. 为 trainer 提供串行的单样本 / 多样本采样入口。

    这一版故意保持轻量：
    - 不管理训练生命周期；
    - 不做更新调度；
    - 不做复杂并发；
    - 只负责把输入样本组织成 `RolloutBatch` 列表。
    """

    def __init__(
        self,
        *,
        rollout: CHEEpisodeRollout,
        config: BaseMultiAgentConfig | None = None,
        default_num_generations: Optional[int] = None,
        default_generation_kwargs: Optional[Dict[str, Any]] = None,
        default_max_turns: Optional[int] = None,
    ) -> None:
        self.rollout = rollout
        self.config = config or BaseMultiAgentConfig()
        self.default_num_generations = (
            self.config.num_generations
            if default_num_generations is None
            else default_num_generations
        )
        self.default_generation_kwargs = dict(default_generation_kwargs or {})
        self.default_max_turns = (
            self.config.max_turns if default_max_turns is None else default_max_turns
        )

    def collect_one(
        self,
        *,
        item: Dict[str, Any],
        num_generations: Optional[int] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        max_turns: Optional[int] = None,
    ) -> RolloutBatch:
        return self.rollout.rollout_episode(
            item=item,
            num_generations=self._resolve_num_generations(num_generations),
            generation_kwargs=self._resolve_generation_kwargs(generation_kwargs),
            max_turns=self._resolve_max_turns(max_turns),
        )

    def collect_batch(
        self,
        *,
        items: Sequence[Dict[str, Any]],
        num_generations: Optional[int] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        max_turns: Optional[int] = None,
    ) -> List[RolloutBatch]:
        resolved_num_generations = self._resolve_num_generations(num_generations)
        resolved_generation_kwargs = self._resolve_generation_kwargs(generation_kwargs)
        resolved_max_turns = self._resolve_max_turns(max_turns)

        return [
            self.rollout.rollout_episode(
                item=item,
                num_generations=resolved_num_generations,
                generation_kwargs=dict(resolved_generation_kwargs),
                max_turns=resolved_max_turns,
            )
            for item in items
        ]

    @staticmethod
    def _normalize_generation_kwargs(
        generation_kwargs: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return dict(generation_kwargs or {})

    def _resolve_generation_kwargs(
        self,
        generation_kwargs: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        resolved = dict(self.default_generation_kwargs)
        resolved.update(self._normalize_generation_kwargs(generation_kwargs))
        return resolved

    def _resolve_num_generations(
        self,
        num_generations: Optional[int],
    ) -> Optional[int]:
        return (
            self.default_num_generations
            if num_generations is None
            else num_generations
        )

    def _resolve_max_turns(self, max_turns: Optional[int]) -> Optional[int]:
        return self.default_max_turns if max_turns is None else max_turns

    def describe_defaults(self) -> Dict[str, Any]:
        return {
            "config": asdict(self.config),
            "default_num_generations": self.default_num_generations,
            "default_generation_kwargs": dict(self.default_generation_kwargs),
            "default_max_turns": self.default_max_turns,
        }
