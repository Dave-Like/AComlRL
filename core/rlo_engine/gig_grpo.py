from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Sequence

from core.common.types import EngineUpdateResult, UpdateBatch
from core.config.config import GIG_GRPOConfig
from core.rlo_engine.base_engine import BaseRLEngine


class GIGGRPOEngine(BaseRLEngine):
    """
    GIG-GRPO 算法引擎骨架。

    当前版本已具备 group-aware 中间训练样本构造能力，但尚未实现
    更高层 group-in-group 统计与真实优化更新。
    """

    def __init__(
        self,
        config: GIG_GRPOConfig | None = None,
        **overrides: Any,
    ) -> None:
        resolved_config = config or GIG_GRPOConfig()
        if overrides:
            resolved_config = GIG_GRPOConfig(
                **{**asdict(resolved_config), **overrides}
            )

        super().__init__(algorithm_name=resolved_config.algorithm_name)
        self.config = resolved_config

    def update(
        self,
        update_batches: Sequence[UpdateBatch],
    ) -> EngineUpdateResult:
        self.validate_update_batches(update_batches)
        metrics = self._build_placeholder_metrics(update_batches)
        metadata = {
            "engine_class": self.__class__.__name__,
            "status": "skeleton",
            "config": asdict(self.config),
        }
        return self.build_update_result(
            updated=False,
            update_batches=update_batches,
            metrics=metrics,
            metadata=metadata,
        )

    def _build_placeholder_metrics(
        self,
        update_batches: Sequence[UpdateBatch],
    ) -> Dict[str, float]:
        train_samples_by_agent = self.build_train_samples(update_batches)
        total_samples = sum(len(batch.samples) for batch in update_batches)
        total_train_samples = sum(
            len(samples) for samples in train_samples_by_agent.values()
        )
        num_agents = len(update_batches)
        mean_samples_per_agent = (
            total_samples / num_agents if num_agents > 0 else 0.0
        )
        mean_train_samples_per_agent = (
            total_train_samples / num_agents if num_agents > 0 else 0.0
        )
        mean_return = (
            sum(
                sample.return_
                for samples in train_samples_by_agent.values()
                for sample in samples
            )
            / total_train_samples
            if total_train_samples > 0
            else 0.0
        )
        mean_advantage = (
            sum(
                sample.normalized_advantage
                for samples in train_samples_by_agent.values()
                for sample in samples
            )
            / total_train_samples
            if total_train_samples > 0
            else 0.0
        )
        mean_group_std = (
            sum(
                sample.group_std_return
                for samples in train_samples_by_agent.values()
                for sample in samples
            )
            / total_train_samples
            if total_train_samples > 0
            else 0.0
        )
        return {
            "num_agents": float(num_agents),
            "total_samples": float(total_samples),
            "total_train_samples": float(total_train_samples),
            "mean_samples_per_agent": float(mean_samples_per_agent),
            "mean_train_samples_per_agent": float(mean_train_samples_per_agent),
            "mean_return": float(mean_return),
            "mean_advantage": float(mean_advantage),
            "mean_group_std": float(mean_group_std),
        }
