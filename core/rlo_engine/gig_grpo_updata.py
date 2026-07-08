from __future__ import annotations

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
        optimizers: Sequence[Optimizer] | None = None,
        advantage_mode: str = "zscore",
        advantage_epsilon: float = 1e-8,
        contribution_mode: str = "hybrid",
        task_combination: str = "linear",
        contribution_lambda: float = 1.0,
        contribution_mix_alpha: float = 0.5,
        counterfactual_anchor_coef: float = 0.25,
        no_helper_token: str = "Nohelperutilitycodeavailable",
    ) -> None:
        super().__init__(
            policy_models=policy_models,
            tokenizers=tokenizers,
            learning_rate=learning_rate,
            update_epochs=update_epochs,
            clip_range=clip_range,
            kl_coef=kl_coef,
            max_grad_norm=max_grad_norm,
            optimizers=optimizers,
        )
        self.advantage_mode = str(advantage_mode)
        self.advantage_epsilon = float(advantage_epsilon)
        self.contribution_mode = str(contribution_mode).strip().lower()
        self.contribution_lambda = float(contribution_lambda)
        self.contribution_mix_alpha = float(contribution_mix_alpha)
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
        adjusted_samples: List[EngineTrainSample] = []
        for sample in samples:
            outer_advantage = self._compute_outer_advantage(sample)
            inner_advantage = float(
                branch_scores_by_node.get(sample.node_id, {}).get(
                    sample.branch_idx,
                    0.0,
                )
            )
            combined_advantage = (
                outer_advantage + self.contribution_lambda * inner_advantage
            )
            sample.normalized_advantage = float(combined_advantage)
            sample.policy_objective = (
                sample.importance_ratio * sample.normalized_advantage
            )
            sample.clipped_policy_objective = (
                sample.clipped_ratio * sample.normalized_advantage
            )
            sample.metadata["outer_advantage"] = float(outer_advantage)
            sample.metadata["inner_advantage"] = float(inner_advantage)
            sample.metadata["combined_advantage"] = float(combined_advantage)
            sample.metadata["contribution_mode"] = self.contribution_mode
            adjusted_samples.append(sample)
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
