from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence

from core.common.types import RolloutBatch
from core.config.config import BaseMultiAgentConfig
from core.environment.che_rollout import CHEEpisodeRollout


class PipelineManager:
    """
    第一阶段的数据流水线控制层。

    当前职责：
    1. 屏蔽 rollout 组件的直接调用细节；
    2. 为 trainer 提供单样本 / 多样本采样入口；
    3. 统一管理 rollout 的 execution mode，便于未来扩展。

    当前支持的模式：
    - `serial`：逐样本串行收集 rollout
    - `batched`：当前仍逐样本组织 episode，但底层执行路径允许走 batched generation

    这一版故意保持轻量：
    - 不管理训练生命周期；
    - 不做更新调度；
    - 不实现复杂的多 episode 全局并发；
    - 只负责把输入样本组织成 `RolloutBatch` 列表。
    """

    SUPPORTED_EXECUTION_MODES = {"serial", "batched"}

    def __init__(
        self,
        *,
        rollout: CHEEpisodeRollout,
        config: BaseMultiAgentConfig | None = None,
        default_num_generations: Optional[int] = None,
        default_generation_kwargs: Optional[Dict[str, Any]] = None,
        default_max_turns: Optional[int] = None,
        execution_mode: Optional[str] = None,
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
        self.execution_mode = self._resolve_execution_mode(execution_mode)

    def collect_one(
        self,
        *,
        item: Dict[str, Any],
        num_generations: Optional[int] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        max_turns: Optional[int] = None,
        execution_mode: Optional[str] = None,
    ) -> RolloutBatch:
        mode = self._resolve_execution_mode(execution_mode)
        if mode == "serial":
            return self._collect_one_serial(
                item=item,
                num_generations=num_generations,
                generation_kwargs=generation_kwargs,
                max_turns=max_turns,
            )
        if mode == "batched":
            return self._collect_one_batched(
                item=item,
                num_generations=num_generations,
                generation_kwargs=generation_kwargs,
                max_turns=max_turns,
            )
        raise ValueError(f"Unsupported execution mode: {mode}")

    def collect_batch(
        self,
        *,
        items: Sequence[Dict[str, Any]],
        num_generations: Optional[int] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        max_turns: Optional[int] = None,
        execution_mode: Optional[str] = None,
    ) -> List[RolloutBatch]:
        mode = self._resolve_execution_mode(execution_mode)
        if mode == "serial":
            return self._collect_batch_serial(
                items=items,
                num_generations=num_generations,
                generation_kwargs=generation_kwargs,
                max_turns=max_turns,
            )
        if mode == "batched":
            return self._collect_batch_batched(
                items=items,
                num_generations=num_generations,
                generation_kwargs=generation_kwargs,
                max_turns=max_turns,
            )
        raise ValueError(f"Unsupported execution mode: {mode}")

    def _collect_one_serial(
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

    def _collect_one_batched(
        self,
        *,
        item: Dict[str, Any],
        num_generations: Optional[int] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        max_turns: Optional[int] = None,
    ) -> RolloutBatch:
        """
        当前 batched 模式的最小实现仍然按单个 episode 调用 rollout，
        但底层 executor / generation engine 可走 batched generation 路径。
        """
        return self.rollout.rollout_episode(
            item=item,
            num_generations=self._resolve_num_generations(num_generations),
            generation_kwargs=self._resolve_generation_kwargs(generation_kwargs),
            max_turns=self._resolve_max_turns(max_turns),
        )

    def _collect_batch_serial(
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

    def _collect_batch_batched(
        self,
        *,
        items: Sequence[Dict[str, Any]],
        num_generations: Optional[int] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        max_turns: Optional[int] = None,
    ) -> List[RolloutBatch]:
        """
        当前 batched 模式仍保留逐样本 episode 组织方式，
        但显式表达执行语义：底层 rollout 执行应优先利用 batch-friendly 路径。

        这样做的目的：
        1. 不破坏现有 rollout / trainer 协议；
        2. 先吃到 generation 层 batch 化带来的收益；
        3. 为未来升级为多 episode 同步调度器保留稳定入口。
        """
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

    def _resolve_execution_mode(self, execution_mode: Optional[str]) -> str:
        raw_mode = execution_mode
        if raw_mode is None:
            raw_mode = getattr(self.config, "execution_mode", "serial")

        mode = str(raw_mode or "serial").strip().lower()
        if mode not in self.SUPPORTED_EXECUTION_MODES:
            raise ValueError(
                f"execution_mode must be one of: "
                f"{', '.join(sorted(self.SUPPORTED_EXECUTION_MODES))}."
            )
        return mode

    def set_execution_mode(self, execution_mode: str) -> None:
        self.execution_mode = self._resolve_execution_mode(execution_mode)

    def describe_defaults(self) -> Dict[str, Any]:
        return {
            "config": asdict(self.config),
            "default_num_generations": self.default_num_generations,
            "default_generation_kwargs": dict(self.default_generation_kwargs),
            "default_max_turns": self.default_max_turns,
            "execution_mode": self.execution_mode,
            "supported_execution_modes": sorted(self.SUPPORTED_EXECUTION_MODES),
        }