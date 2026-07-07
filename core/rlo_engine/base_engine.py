from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence

from core.common.types import (
    EngineTrainSample,
    EngineUpdateResult,
    FlatBranchSample,
    UpdateBatch,
)


class BaseRLEngine(ABC):
    """
    多智能体大模型强化学习算法引擎的抽象基类。

    第一阶段统一两层公共约定：
    1. `trainer -> engine` 输入协议固定为 `Sequence[UpdateBatch]`；
    2. `engine -> trainer/logger` 输出协议固定为 `EngineUpdateResult`。

    当前版本额外提供两层辅助接口：
    - `FlatBranchSample`: branch 展平样本；
    - `EngineTrainSample`: 已带组统计的训练中间样本。
    """

    def __init__(self, *, algorithm_name: str = "") -> None:
        self.algorithm_name = algorithm_name

    @abstractmethod
    def update(
        self,
        update_batches: Sequence[UpdateBatch],
    ) -> EngineUpdateResult:
        """
        消费 trainer 组装好的更新批，并执行一次算法更新。

        子类必须返回 `EngineUpdateResult`，以保证 trainer / logger
        面向统一协议工作，而不是依赖某个具体算法的私有返回格式。
        """

    def validate_update_batches(
        self,
        update_batches: Sequence[UpdateBatch],
    ) -> None:
        if not update_batches:
            raise ValueError("update_batches must not be empty.")
        for batch in update_batches:
            self._validate_single_update_batch(batch)

    def flatten_update_batches(
        self,
        update_batches: Sequence[UpdateBatch],
    ) -> Dict[int, List[FlatBranchSample]]:
        self.validate_update_batches(update_batches)
        return {
            batch.agent_idx: batch.flatten_branch_samples()
            for batch in update_batches
        }

    def build_train_samples(
        self,
        update_batches: Sequence[UpdateBatch],
    ) -> Dict[int, List[EngineTrainSample]]:
        self.validate_update_batches(update_batches)
        train_samples_by_agent: Dict[int, List[EngineTrainSample]] = {}

        for batch in update_batches:
            train_samples_by_agent[batch.agent_idx] = self._build_train_samples_for_batch(
                batch
            )
        return train_samples_by_agent

    def summarize_update_batches(
        self,
        update_batches: Sequence[UpdateBatch],
    ) -> Dict[str, Any]:
        self.validate_update_batches(update_batches)
        flat_by_agent = self.flatten_update_batches(update_batches)
        return {
            "algorithm_name": self.algorithm_name,
            "num_update_batches": len(update_batches),
            "agent_indices": [batch.agent_idx for batch in update_batches],
            "num_samples_per_batch": [len(batch.samples) for batch in update_batches],
            "total_num_samples": sum(len(batch.samples) for batch in update_batches),
            "num_branches_per_batch": [
                len(flat_by_agent[batch.agent_idx]) for batch in update_batches
            ],
            "total_num_branches": sum(
                len(samples) for samples in flat_by_agent.values()
            ),
        }

    def build_update_result(
        self,
        *,
        updated: bool,
        update_batches: Sequence[UpdateBatch],
        metrics: Dict[str, float] | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> EngineUpdateResult:
        summary = self.summarize_update_batches(update_batches)
        return EngineUpdateResult(
            algorithm_name=self.algorithm_name,
            updated=bool(updated),
            num_update_batches=int(summary["num_update_batches"]),
            num_samples=int(summary["total_num_samples"]),
            metrics=dict(metrics or {}),
            metadata={**summary, **dict(metadata or {})},
        )

    def _build_train_samples_for_batch(
        self,
        update_batch: UpdateBatch,
    ) -> List[EngineTrainSample]:
        flat_samples = update_batch.flatten_branch_samples()
        grouped_returns: Dict[str, List[float]] = {}

        for sample in flat_samples:
            grouped_returns.setdefault(sample.node_id, []).append(float(sample.return_))

        train_samples: List[EngineTrainSample] = []
        for sample in flat_samples:
            returns = grouped_returns.get(sample.node_id, [float(sample.return_)])
            group_mean = sum(returns) / len(returns) if returns else 0.0
            variance = (
                sum((value - group_mean) ** 2 for value in returns) / len(returns)
                if returns
                else 0.0
            )
            group_std = variance ** 0.5
            centered_return = float(sample.return_) - group_mean
            normalized_advantage = (
                centered_return / group_std
                if update_batch.normalize_advantages and group_std > 1e-8
                else centered_return
            )

            agent_prompt = (
                sample.agent_prompts[update_batch.agent_idx]
                if update_batch.agent_idx < len(sample.agent_prompts)
                else ""
            )
            agent_action = (
                sample.actions_per_agent[update_batch.agent_idx]
                if update_batch.agent_idx < len(sample.actions_per_agent)
                else ""
            )
            agent_prompt_history = (
                list(sample.prompt_history_per_agent[update_batch.agent_idx])
                if update_batch.agent_idx < len(sample.prompt_history_per_agent)
                else []
            )
            agent_response_history = (
                list(sample.response_history_per_agent[update_batch.agent_idx])
                if update_batch.agent_idx < len(sample.response_history_per_agent)
                else []
            )
            agent_logprob = (
                float(sample.metadata["logprobs_per_agent"][update_batch.agent_idx])
                if "logprobs_per_agent" in sample.metadata
                and update_batch.agent_idx < len(sample.metadata["logprobs_per_agent"])
                and sample.metadata["logprobs_per_agent"][update_batch.agent_idx]
                is not None
                else None
            )

            train_samples.append(
                EngineTrainSample(
                    agent_idx=update_batch.agent_idx,
                    node_id=sample.node_id,
                    episode_id=sample.episode_id,
                    turn_idx=sample.turn_idx,
                    env_step=sample.env_step,
                    depth=sample.depth,
                    branch_idx=sample.branch_idx,
                    action_text=agent_action,
                    agent_prompt=agent_prompt,
                    agent_prompt_history=agent_prompt_history,
                    agent_response_history=agent_response_history,
                    joint_actions=list(sample.actions_per_agent),
                    reward=float(sample.reward),
                    return_=float(sample.return_),
                    group_mean_return=float(group_mean),
                    group_std_return=float(group_std),
                    centered_return=float(centered_return),
                    normalized_advantage=float(normalized_advantage),
                    logprob=agent_logprob,
                    old_logprob=agent_logprob,
                    terminal=bool(sample.terminal),
                    metadata=dict(sample.metadata),
                )
            )
        return train_samples

    def _validate_single_update_batch(self, update_batch: UpdateBatch) -> None:
        if update_batch.agent_idx < 0:
            raise ValueError("update_batch.agent_idx must be >= 0.")
        if update_batch.discount < 0.0 or update_batch.discount > 1.0:
            raise ValueError("update_batch.discount must be within [0, 1].")
        if not isinstance(update_batch.normalize_advantages, bool):
            raise TypeError("update_batch.normalize_advantages must be a bool.")
        if not isinstance(update_batch.samples, list):
            raise TypeError("update_batch.samples must be a list of NodeSample.")
        if any(sample is None for sample in update_batch.samples):
            raise ValueError("update_batch.samples must not contain None.")

    def get_algorithm_name(self) -> str:
        return self.algorithm_name

    def build_empty_update_result(self) -> EngineUpdateResult:
        return EngineUpdateResult(
            algorithm_name=self.algorithm_name,
            updated=False,
            num_update_batches=0,
            num_samples=0,
            metrics={},
            metadata={},
        )
