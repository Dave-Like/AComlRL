from __future__ import annotations

import math
from typing import Dict, List, Sequence

from torch.optim import Optimizer
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from core.common.types import EngineTrainSample
from core.rlo_engine.gig_grpo_helper import (
    ContributionAnalyzer,
    stable_mean,
    stable_std,
)
from core.rlo_engine.magrpo_update import MAGRPOPolicyUpdater


class GIGGRPOPolicyUpdater(MAGRPOPolicyUpdater):
    """封装 GIG-GRPO 的真实策略更新逻辑。"""

    def __init__(
        self,
        *,
        policy_models: Sequence[PreTrainedModel] | None = None,
        tokenizers: Sequence[PreTrainedTokenizerBase] | PreTrainedTokenizerBase | None = None,
        learning_rate: float = 1e-5,
        update_epochs: int = 1,
        clip_range: float = 0.2,
        kl_coef: float = 0.0,
        max_grad_norm: float | None = 1.0,
        max_safe_kl: float | None = 2.0,
        optimizers: Sequence[Optimizer] | None = None,
        advantage_mode: str = "zscore",
        advantage_epsilon: float = 1e-8,
        contribution_mode: str = "hybrid",
        task_combination: str = "linear",
        contribution_lambda: float = 1.0,
        contribution_mix_alpha: float = 0.5,
        counterfactual_anchor_coef: float = 0.25,
        no_helper_token: str = "Nohelperutilitycodeavailable",
        outer_advantage_clip: float | None = 5.0,
        inner_advantage_clip: float | None = 3.0,
        combined_advantage_clip: float | None = 5.0,
        inner_scale_mode: str = "match_outer_mean_abs",
        min_inner_scale: float = 0.5,
        max_inner_scale: float = 3.0,
    ) -> None:
        super().__init__(
            policy_models=policy_models,
            tokenizers=tokenizers,
            learning_rate=learning_rate,
            update_epochs=update_epochs,
            clip_range=clip_range,
            kl_coef=kl_coef,
            max_grad_norm=max_grad_norm,
            max_safe_kl=max_safe_kl,
            optimizers=optimizers,
        )
        self.advantage_mode = str(advantage_mode)
        self.advantage_epsilon = float(advantage_epsilon)
        self.contribution_mode = str(contribution_mode).strip().lower()
        self.contribution_lambda = float(contribution_lambda)
        self.contribution_mix_alpha = float(contribution_mix_alpha)
        self.outer_advantage_clip = (
            None if outer_advantage_clip is None else float(outer_advantage_clip)
        )
        self.inner_advantage_clip = (
            None if inner_advantage_clip is None else float(inner_advantage_clip)
        )
        self.combined_advantage_clip = (
            None if combined_advantage_clip is None else float(combined_advantage_clip)
        )
        self.inner_scale_mode = str(inner_scale_mode or "match_outer_mean_abs").strip().lower()
        self.min_inner_scale = float(min_inner_scale)
        self.max_inner_scale = float(max_inner_scale)
        self.contribution_analyzer = ContributionAnalyzer(
            task_combination=task_combination,
            anchor_coef=counterfactual_anchor_coef,
            no_helper_token=no_helper_token,
        )

    def build_gig_train_samples(
        self,
        train_samples_by_agent: Dict[int, List[EngineTrainSample]],
    ) -> Dict[int, List[EngineTrainSample]]:
        branch_scores_by_node = self._compute_branch_contribution_scores(
            train_samples_by_agent
        )
        adjusted: Dict[int, List[EngineTrainSample]] = {}
        for agent_idx, samples in train_samples_by_agent.items():
            adjusted[agent_idx] = self._build_adjusted_samples_for_agent(
                samples=samples,
                branch_scores_by_node=branch_scores_by_node,
            )
        return adjusted

    def run(
        self,
        train_samples_by_agent: Dict[int, List[EngineTrainSample]],
    ) -> Dict[str, float]:
        return super().run(train_samples_by_agent)

    def _compute_branch_contribution_scores(
        self,
        train_samples_by_agent: Dict[int, List[EngineTrainSample]],
    ) -> Dict[str, Dict[int, float]]:
        samples_by_node: Dict[str, Dict[int, List[EngineTrainSample]]] = {}
        for samples in train_samples_by_agent.values():
            for sample in samples:
                samples_by_node.setdefault(sample.node_id, {}).setdefault(
                    sample.branch_idx, []
                ).append(sample)

        branch_scores_by_node: Dict[str, Dict[int, float]] = {}
        for node_id, per_branch_samples in samples_by_node.items():
            raw_scores: Dict[int, float] = {}
            for branch_idx, branch_samples in per_branch_samples.items():
                task_values: List[float] = []
                counterfactual_values: List[float] = []
                for sample in branch_samples:
                    task_score, task_features = self.contribution_analyzer.task_score(
                        sample.action_text
                    )
                    peer_samples = [
                        peer
                        for peer in branch_samples
                        if peer.agent_idx != sample.agent_idx
                    ]
                    cf_score, cf_metrics = (
                        self.contribution_analyzer.counterfactual_score(
                            sample=sample,
                            peer_samples_in_group=peer_samples,
                        )
                    )
                    task_values.append(task_score)
                    counterfactual_values.append(cf_score)
                    sample.metadata.update(task_features)
                    sample.metadata.update(cf_metrics)
                    sample.metadata["task_score"] = float(task_score)
                    sample.metadata["counterfactual_score"] = float(cf_score)
                raw_scores[branch_idx] = self._combine_branch_scores(
                    task_values,
                    counterfactual_values,
                )
            branch_scores_by_node[node_id] = self._normalize_branch_scores(raw_scores)
        return branch_scores_by_node

    def _combine_branch_scores(
        self,
        task_values: Sequence[float],
        counterfactual_values: Sequence[float],
    ) -> float:
        task_score = stable_mean(task_values)
        counterfactual_score = stable_mean(counterfactual_values)
        if self.contribution_mode == "task":
            return float(task_score)
        if self.contribution_mode == "counterfactual":
            return float(counterfactual_score)
        return float(
            self.contribution_mix_alpha * task_score
            + (1.0 - self.contribution_mix_alpha) * counterfactual_score
        )

    def _normalize_branch_scores(
        self,
        raw_scores: Dict[int, float],
    ) -> Dict[int, float]:
        values = list(raw_scores.values())
        mean_value = stable_mean(values)
        std_value = stable_std(values)
        if std_value <= self.advantage_epsilon:
            return {branch_idx: 0.0 for branch_idx in raw_scores}
        return {
            branch_idx: float((score - mean_value) / std_value)
            for branch_idx, score in raw_scores.items()
        }

    def _build_adjusted_samples_for_agent(
        self,
        *,
        samples: Sequence[EngineTrainSample],
        branch_scores_by_node: Dict[str, Dict[int, float]],
    ) -> List[EngineTrainSample]:
        raw_outer_values: List[float] = []
        raw_inner_values: List[float] = []

        for sample in samples:
            raw_outer_values.append(self._compute_outer_advantage(sample))
            raw_inner_values.append(
                float(
                    branch_scores_by_node.get(sample.node_id, {}).get(
                        sample.branch_idx,
                        0.0,
                    )
                )
            )

        inner_scale = self._compute_inner_scale(
            raw_outer_values=raw_outer_values,
            raw_inner_values=raw_inner_values,
        )

        adjusted_samples: List[EngineTrainSample] = []
        combined_advantages: List[float] = []

        for sample in samples:
            raw_outer_advantage = self._compute_outer_advantage(sample)
            raw_inner_advantage = float(
                branch_scores_by_node.get(sample.node_id, {}).get(
                    sample.branch_idx,
                    0.0,
                )
            )

            outer_advantage = self._clip_finite_scalar(
                raw_outer_advantage,
                self.outer_advantage_clip,
            )
            inner_advantage = self._clip_finite_scalar(
                raw_inner_advantage,
                self.inner_advantage_clip,
            )
            scaled_inner_advantage = self._clip_finite_scalar(
                inner_advantage * inner_scale,
                self.inner_advantage_clip,
            )

            raw_combined_advantage = (
                outer_advantage
                + self.contribution_lambda * scaled_inner_advantage
            )
            combined_advantage = self._clip_finite_scalar(
                raw_combined_advantage,
                self.combined_advantage_clip,
            )

            sample.metadata["raw_outer_advantage"] = float(raw_outer_advantage)
            sample.metadata["raw_inner_advantage"] = float(raw_inner_advantage)
            sample.metadata["outer_advantage"] = float(outer_advantage)
            sample.metadata["inner_advantage"] = float(inner_advantage)
            sample.metadata["scaled_inner_advantage"] = float(scaled_inner_advantage)
            sample.metadata["inner_scale"] = float(inner_scale)
            sample.metadata["raw_combined_advantage"] = float(raw_combined_advantage)
            sample.metadata["combined_advantage"] = float(combined_advantage)
            sample.metadata["contribution_mode"] = self.contribution_mode

            adjusted_samples.append(sample)
            combined_advantages.append(float(combined_advantage))

            mean_combined = stable_mean(combined_advantages, 0.0)
            std_combined = float(stable_std(combined_advantages))

            for sample, combined_advantage in zip(adjusted_samples, combined_advantages):
                if not math.isfinite(std_combined) or std_combined <= self.advantage_epsilon:
                    final_advantage = 0.0
                else:
                    final_advantage = (combined_advantage - mean_combined) / std_combined

            final_advantage = self._clip_finite_scalar(
                final_advantage,
                self.combined_advantage_clip,
            )

            sample.normalized_advantage = float(final_advantage)
            sample.policy_objective = (
                sample.importance_ratio * sample.normalized_advantage
            )
            sample.clipped_policy_objective = (
                sample.clipped_ratio * sample.normalized_advantage
            )

            sample.metadata["final_advantage"] = float(final_advantage)
            sample.metadata["combined_advantage_mean"] = float(mean_combined)
            sample.metadata["combined_advantage_std"] = float(std_combined)

        return adjusted_samples

    def _compute_outer_advantage(self, sample: EngineTrainSample) -> float:
        if self.advantage_mode == "return":
            return float(sample.return_)
        if self.advantage_mode == "centered":
            return float(sample.centered_return)
        if self.advantage_mode == "zscore":
            if sample.group_std_return <= self.advantage_epsilon:
                return 0.0
            return float(sample.centered_return / sample.group_std_return)
        raise ValueError(
            "Unsupported GIG-GRPO advantage_mode: "
            f"{self.advantage_mode!r}. Expected one of ['zscore', 'centered', 'return']."
        )

    def _compute_inner_scale(
        self,
        *,
        raw_outer_values: Sequence[float],
        raw_inner_values: Sequence[float],
    ) -> float:
        if self.inner_scale_mode in {"none", "identity"}:
            return 1.0

        mean_abs_outer = self._mean_abs_finite(raw_outer_values)
        mean_abs_inner = self._mean_abs_finite(raw_inner_values)

        if mean_abs_inner <= self.advantage_epsilon:
            return 1.0

        if self.inner_scale_mode == "match_outer_mean_abs":
            ratio = mean_abs_outer / max(mean_abs_inner, self.advantage_epsilon)
            return self._clip_scale(ratio)

        return 1.0

    def _clip_scale(self, value: float) -> float:
        if not math.isfinite(float(value)):
            return 1.0
        lower = min(self.min_inner_scale, self.max_inner_scale)
        upper = max(self.min_inner_scale, self.max_inner_scale)
        return float(min(max(float(value), lower), upper))

    @staticmethod
    def _mean_abs_finite(values: Sequence[float]) -> float:
        finite_values = [
            abs(float(value))
            for value in values
            if math.isfinite(float(value))
        ]
        return stable_mean(finite_values, 0.0)

    @staticmethod
    def _clip_finite_scalar(value: float, clip_value: float | None) -> float:
        if not math.isfinite(float(value)):
            return 0.0
        scalar = float(value)
        if clip_value is None:
            return scalar
        bound = abs(float(clip_value))
        return float(min(max(scalar, -bound), bound))