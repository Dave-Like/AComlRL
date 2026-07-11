from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Sequence

from core.common.types import EngineTrainSample, EngineUpdateResult, UpdateBatch
from core.config.config import MAGRPOConfig
from core.rlo_engine.base_engine import BaseRLEngine
from core.rlo_engine.magrpo_update import MAGRPOPolicyUpdater


class MAGRPOEngine(BaseRLEngine):
    """MAGRPO 算法引擎。真实 update 逻辑委托给 `MAGRPOPolicyUpdater`。"""

    def __init__(
        self,
        config: MAGRPOConfig | None = None,
        **overrides: Any,
    ) -> None:
        resolved_config = config or MAGRPOConfig()
        if overrides:
            resolved_config = MAGRPOConfig(**{**asdict(resolved_config), **overrides})

        super().__init__(algorithm_name=resolved_config.algorithm_name)
        self.config = resolved_config
        self.advantage_mode = str(self.config.advantage_mode)
        self.advantage_epsilon = float(self.config.advantage_epsilon)
        self.clip_range = float(self.config.clip_range)
        self.kl_coef = float(self.config.kl_coef)
        self.learning_rate = float(self.config.learning_rate)
        self.update_epochs = int(self.config.update_epochs)
        self.max_grad_norm = self.config.max_grad_norm
        self.policy_updater = MAGRPOPolicyUpdater(
            learning_rate=self.learning_rate,
            update_epochs=self.update_epochs,
            clip_range=self.clip_range,
            kl_coef=self.kl_coef,
            max_grad_norm=self.max_grad_norm,
        )

    def attach_policy_components(
        self,
        *,
        policy_models: Sequence[Any] | None = None,
        tokenizers: Sequence[Any] | Any | None = None,
        optimizers: Sequence[Any] | None = None,
    ) -> None:
        self.policy_updater = MAGRPOPolicyUpdater(
            policy_models=policy_models,
            tokenizers=tokenizers,
            optimizers=optimizers,
            learning_rate=self.learning_rate,
            update_epochs=self.update_epochs,
            clip_range=self.clip_range,
            kl_coef=self.kl_coef,
            max_grad_norm=self.max_grad_norm,
        )

    def update(
        self,
        update_batches: Sequence[UpdateBatch],
    ) -> EngineUpdateResult:
        self.validate_update_batches(update_batches)
        train_samples_by_agent = self.build_magrpo_train_samples(update_batches)
        metrics = self._build_magrpo_metrics(update_batches, train_samples_by_agent)

        updated = self.policy_updater.is_ready(train_samples_by_agent)
        status = "policy_skeleton_ready"
        if updated:
            metrics.update(self.policy_updater.run(train_samples_by_agent))
            status = "updated"

        return self.build_update_result(
            updated=updated,
            update_batches=update_batches,
            metrics=metrics,
            metadata={
                "engine_class": self.__class__.__name__,
                "status": status,
                "config": asdict(self.config),
                "advantage_mode": self.advantage_mode,
                "advantage_epsilon": self.advantage_epsilon,
                "clip_range": self.clip_range,
                "kl_coef": self.kl_coef,
                "learning_rate": self.learning_rate,
                "update_epochs": float(self.update_epochs),
            },
        )

    def build_magrpo_train_samples(
        self,
        update_batches: Sequence[UpdateBatch],
    ) -> Dict[int, List[EngineTrainSample]]:
        self.validate_update_batches(update_batches)
        return {
            batch.agent_idx: self._build_magrpo_train_samples_for_batch(batch)
            for batch in update_batches
        }

    def _build_magrpo_train_samples_for_batch(
        self,
        update_batch: UpdateBatch,
    ) -> List[EngineTrainSample]:
        base_samples = self._build_train_samples_for_batch(update_batch)
        adjusted_samples: List[EngineTrainSample] = []

        for sample in base_samples:
            advantage = self._compute_group_relative_advantage(sample)
            logprob = self._resolve_logprob(sample.metadata.get("logprob", sample.logprob))
            old_logprob = self._resolve_logprob(
                sample.metadata.get("old_logprob", sample.old_logprob)
            )
            ref_logprob = self._resolve_logprob(sample.metadata.get("ref_logprob"))
            importance_ratio = self._compute_importance_ratio(logprob, old_logprob)
            clipped_ratio = self._clip_ratio(importance_ratio)
            adjusted_samples.append(
                EngineTrainSample(
                    agent_idx=sample.agent_idx,
                    node_id=sample.node_id,
                    episode_id=sample.episode_id,
                    turn_idx=sample.turn_idx,
                    env_step=sample.env_step,
                    depth=sample.depth,
                    branch_idx=sample.branch_idx,
                    action_text=sample.action_text,
                    agent_prompt=sample.agent_prompt,
                    agent_prompt_history=list(sample.agent_prompt_history),
                    agent_response_history=list(sample.agent_response_history),
                    joint_actions=list(sample.joint_actions),
                    reward=sample.reward,
                    return_=sample.return_,
                    group_mean_return=sample.group_mean_return,
                    group_std_return=sample.group_std_return,
                    centered_return=sample.centered_return,
                    normalized_advantage=advantage,
                    logprob=logprob,
                    old_logprob=old_logprob,
                    ref_logprob=ref_logprob,
                    importance_ratio=importance_ratio,
                    clipped_ratio=clipped_ratio,
                    policy_objective=importance_ratio * advantage,
                    clipped_policy_objective=clipped_ratio * advantage,
                    approx_kl=self._compute_approx_kl(logprob, ref_logprob),
                    terminal=sample.terminal,
                    metadata={
                        **dict(sample.metadata),
                        "advantage_mode": self.advantage_mode,
                        "clip_range": self.clip_range,
                    },
                )
            )
        return adjusted_samples

    def _compute_group_relative_advantage(
        self,
        sample: EngineTrainSample,
    ) -> float:
        if self.advantage_mode == "return":
            return float(sample.return_)
        if self.advantage_mode == "centered":
            return float(sample.centered_return)
        if self.advantage_mode == "zscore":
            if sample.group_std_return <= self.advantage_epsilon:
                return 0.0
            return float(sample.centered_return / sample.group_std_return)
        raise ValueError(
            "Unsupported MAGRPO advantage_mode: "
            f"{self.advantage_mode!r}. Expected one of ['zscore', 'centered', 'return']."
        )

    @staticmethod
    def _resolve_logprob(value: Any) -> float | None:
        if value is None:
            return None
        return float(value)

    def _compute_importance_ratio(
        self,
        logprob: float | None,
        old_logprob: float | None,
    ) -> float:
        if logprob is None or old_logprob is None:
            return 1.0
        return float(pow(2.718281828459045, logprob - old_logprob))

    def _clip_ratio(self, ratio: float) -> float:
        lower = 1.0 - self.clip_range
        upper = 1.0 + self.clip_range
        return float(min(max(ratio, lower), upper))

    def _compute_approx_kl(
        self,
        logprob: float | None,
        ref_logprob: float | None,
    ) -> float:
        if logprob is None or ref_logprob is None:
            return 0.0
        return float(logprob - ref_logprob)

    def _build_magrpo_metrics(
        self,
        update_batches: Sequence[UpdateBatch],
        train_samples_by_agent: Dict[int, List[EngineTrainSample]],
    ) -> Dict[str, float]:
        total_samples = sum(len(batch.samples) for batch in update_batches)
        all_train_samples = [sample for samples in train_samples_by_agent.values() for sample in samples]
        total_train_samples = len(all_train_samples)
        num_agents = len(update_batches)

        def mean(values: List[float], default: float = 0.0) -> float:
            return sum(values) / len(values) if values else default

        return {
            "num_agents": float(num_agents),
            "total_samples": float(total_samples),
            "total_train_samples": float(total_train_samples),
            "mean_samples_per_agent": float(total_samples / num_agents) if num_agents > 0 else 0.0,
            "mean_train_samples_per_agent": float(total_train_samples / num_agents) if num_agents > 0 else 0.0,
            "mean_return": mean([sample.return_ for sample in all_train_samples]),
            "mean_centered_return": mean([sample.centered_return for sample in all_train_samples]),
            "mean_advantage": mean([sample.normalized_advantage for sample in all_train_samples]),
            "mean_group_std": mean([sample.group_std_return for sample in all_train_samples]),
            "mean_ratio": mean([sample.importance_ratio for sample in all_train_samples], 1.0),
            "mean_clipped_ratio": mean([sample.clipped_ratio for sample in all_train_samples], 1.0),
            "mean_policy_objective": mean([sample.policy_objective for sample in all_train_samples]),
            "mean_clipped_policy_objective": mean([sample.clipped_policy_objective for sample in all_train_samples]),
            "mean_approx_kl": mean([sample.approx_kl for sample in all_train_samples]),
            "positive_advantage_ratio": mean([1.0 if sample.normalized_advantage > 0.0 else 0.0 for sample in all_train_samples]),
            "zero_advantage_ratio": mean([1.0 if abs(sample.normalized_advantage) <= self.advantage_epsilon else 0.0 for sample in all_train_samples]),
            "degenerate_group_ratio": mean([1.0 if sample.group_std_return <= self.advantage_epsilon else 0.0 for sample in all_train_samples]),
            "missing_logprob_ratio": mean([1.0 if sample.logprob is None else 0.0 for sample in all_train_samples], 1.0),
        }
