from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Sequence

from core.common.types import EngineTrainSample, EngineUpdateResult, UpdateBatch
from core.config.config import GIG_GRPOConfig
from core.rlo_engine.base_engine import BaseRLEngine
from core.rlo_engine.gig_grpo_helper import stable_mean
from core.rlo_engine.gig_grpo_updata import GIGGRPOPolicyUpdater


class GIGGRPOEngine(BaseRLEngine):
    """GIG-GRPO 算法引擎。"""

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
        self.advantage_mode = str(self.config.advantage_mode)
        self.advantage_epsilon = float(self.config.advantage_epsilon)
        self.clip_range = float(self.config.clip_range)
        self.kl_coef = float(self.config.kl_coef)
        self.learning_rate = float(self.config.learning_rate)
        self.update_epochs = int(self.config.update_epochs)
        self.max_grad_norm = self.config.max_grad_norm
        self.policy_updater = GIGGRPOPolicyUpdater(
            learning_rate=self.learning_rate,
            update_epochs=self.update_epochs,
            clip_range=self.clip_range,
            kl_coef=self.kl_coef,
            max_grad_norm=self.max_grad_norm,
            max_safe_kl=self.config.max_safe_kl,
            advantage_mode=self.advantage_mode,
            advantage_epsilon=self.advantage_epsilon,
            contribution_mode=self.config.contribution_mode,
            task_combination=self.config.task_combination,
            contribution_lambda=self.config.contribution_lambda,
            contribution_mix_alpha=self.config.contribution_mix_alpha,
            counterfactual_anchor_coef=self.config.counterfactual_anchor_coef,
            no_helper_token=self.config.no_helper_token,
            outer_advantage_clip=self.config.outer_advantage_clip,
            inner_advantage_clip=self.config.inner_advantage_clip,
            combined_advantage_clip=self.config.combined_advantage_clip,
            inner_scale_mode=self.config.inner_scale_mode,
            min_inner_scale=self.config.min_inner_scale,
            max_inner_scale=self.config.max_inner_scale,
        
        )

    def attach_policy_components(
        self,
        *,
        policy_models: Sequence[Any] | None = None,
        tokenizers: Sequence[Any] | Any | None = None,
        optimizers: Sequence[Any] | None = None,
    ) -> None:
        self.policy_updater = GIGGRPOPolicyUpdater(
            policy_models=policy_models,
            tokenizers=tokenizers,
            optimizers=optimizers,
            learning_rate=self.learning_rate,
            update_epochs=self.update_epochs,
            clip_range=self.clip_range,
            kl_coef=self.kl_coef,
            max_grad_norm=self.max_grad_norm,
            max_safe_kl=self.config.max_safe_kl,
            advantage_mode=self.advantage_mode,
            advantage_epsilon=self.advantage_epsilon,
            contribution_mode=self.config.contribution_mode,
            task_combination=self.config.task_combination,
            contribution_lambda=self.config.contribution_lambda,
            contribution_mix_alpha=self.config.contribution_mix_alpha,
            counterfactual_anchor_coef=self.config.counterfactual_anchor_coef,
            no_helper_token=self.config.no_helper_token,
            outer_advantage_clip=self.config.outer_advantage_clip,
            inner_advantage_clip=self.config.inner_advantage_clip,
            combined_advantage_clip=self.config.combined_advantage_clip,
            inner_scale_mode=self.config.inner_scale_mode,
            min_inner_scale=self.config.min_inner_scale,
            max_inner_scale=self.config.max_inner_scale,
        )

    def update(
        self,
        update_batches: Sequence[UpdateBatch],
    ) -> EngineUpdateResult:
        self.validate_update_batches(update_batches)
        train_samples_by_agent = self.build_train_samples(update_batches)
        gig_train_samples_by_agent = self.policy_updater.build_gig_train_samples(
            train_samples_by_agent
        )
        metrics = self._build_metrics(update_batches, gig_train_samples_by_agent)
        advantage_matrices = self._build_advantage_matrices(gig_train_samples_by_agent)

        updated = self.policy_updater.is_ready(gig_train_samples_by_agent)
        status = "policy_skeleton_ready"
        if updated:
            metrics.update(self.policy_updater.run(gig_train_samples_by_agent))
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
                "contribution_mode": self.config.contribution_mode,
                "task_combination": self.config.task_combination,
                "contribution_lambda": float(self.config.contribution_lambda),
                "contribution_mix_alpha": float(self.config.contribution_mix_alpha),
                "advantage_matrices": advantage_matrices,
            },
        )

    def _build_metrics(
        self,
        update_batches: Sequence[UpdateBatch],
        train_samples_by_agent: Dict[int, List[EngineTrainSample]],
    ) -> Dict[str, float]:
        total_samples = sum(len(batch.samples) for batch in update_batches)
        all_train_samples = [
            sample
            for samples in train_samples_by_agent.values()
            for sample in samples
        ]
        total_train_samples = len(all_train_samples)
        num_agents = len(update_batches)
        return {
            "num_agents": float(num_agents),
            "total_samples": float(total_samples),
            "total_train_samples": float(total_train_samples),
            "mean_samples_per_agent": float(total_samples / num_agents) if num_agents > 0 else 0.0,
            "mean_train_samples_per_agent": float(total_train_samples / num_agents) if num_agents > 0 else 0.0,
            "mean_return": stable_mean([sample.return_ for sample in all_train_samples]),
            "mean_centered_return": stable_mean([sample.centered_return for sample in all_train_samples]),
            "mean_advantage": stable_mean([sample.normalized_advantage for sample in all_train_samples]),
            "mean_group_std": stable_mean([sample.group_std_return for sample in all_train_samples]),
            "mean_outer_advantage": stable_mean([float(sample.metadata.get("outer_advantage", 0.0)) for sample in all_train_samples]),
            "mean_inner_advantage": stable_mean([float(sample.metadata.get("inner_advantage", 0.0)) for sample in all_train_samples]),
            "mean_scaled_inner_advantage": stable_mean([float(sample.metadata.get("scaled_inner_advantage", 0.0)) for sample in all_train_samples]),
            "mean_combined_advantage": stable_mean([float(sample.metadata.get("combined_advantage", 0.0)) for sample in all_train_samples]),
            "mean_final_advantage": stable_mean([float(sample.metadata.get("final_advantage", sample.normalized_advantage)) for sample in all_train_samples]),
            "mean_abs_final_advantage": stable_mean([abs(float(sample.metadata.get("final_advantage", sample.normalized_advantage))) for sample in all_train_samples]),
            "mean_abs_outer_advantage": stable_mean([abs(float(sample.metadata.get("outer_advantage", 0.0))) for sample in all_train_samples]),
            "mean_abs_inner_advantage": stable_mean([abs(float(sample.metadata.get("inner_advantage", 0.0))) for sample in all_train_samples]),
            "mean_abs_scaled_inner_advantage": stable_mean([abs(float(sample.metadata.get("scaled_inner_advantage", 0.0))) for sample in all_train_samples]),
            "mean_abs_combined_advantage": stable_mean([abs(float(sample.metadata.get("combined_advantage", 0.0))) for sample in all_train_samples]),
            "mean_inner_scale": stable_mean([float(sample.metadata.get("inner_scale", 1.0)) for sample in all_train_samples], 1.0),
            "inner_outer_scale_ratio": (
                stable_mean([abs(float(sample.metadata.get("scaled_inner_advantage", 0.0))) for sample in all_train_samples])
                / max(
                    stable_mean([abs(float(sample.metadata.get("outer_advantage", 0.0))) for sample in all_train_samples]),
                    self.advantage_epsilon,
                )
            ),
            "mean_task_score": stable_mean([float(sample.metadata.get("task_score", 0.0)) for sample in all_train_samples]),
            "mean_counterfactual_score": stable_mean([float(sample.metadata.get("counterfactual_score", 0.0)) for sample in all_train_samples]),
            "mean_cf_ablation": stable_mean([float(sample.metadata.get("cf_ablation", 0.0)) for sample in all_train_samples]),
            "mean_cf_cross": stable_mean([float(sample.metadata.get("cf_cross", 0.0)) for sample in all_train_samples]),
            "mean_cf_anchor": stable_mean([float(sample.metadata.get("cf_anchor", 0.0)) for sample in all_train_samples]),
            "mean_exists": stable_mean([float(sample.metadata.get("exists", 0.0)) for sample in all_train_samples]),
            "mean_called": stable_mean([float(sample.metadata.get("called", 0.0)) for sample in all_train_samples]),
            "mean_used": stable_mean([float(sample.metadata.get("used", 0.0)) for sample in all_train_samples]),
            "mean_ignored": stable_mean([float(sample.metadata.get("ignored", 0.0)) for sample in all_train_samples]),
            "mean_ratio": stable_mean([sample.importance_ratio for sample in all_train_samples], 1.0),
            "mean_clipped_ratio": stable_mean([sample.clipped_ratio for sample in all_train_samples], 1.0),
            "mean_policy_objective": stable_mean([sample.policy_objective for sample in all_train_samples]),
            "mean_clipped_policy_objective": stable_mean([sample.clipped_policy_objective for sample in all_train_samples]),
            "mean_approx_kl": stable_mean([sample.approx_kl for sample in all_train_samples]),
            "positive_advantage_ratio": stable_mean([1.0 if sample.normalized_advantage > 0.0 else 0.0 for sample in all_train_samples]),
            "zero_advantage_ratio": stable_mean([
                1.0 if abs(sample.normalized_advantage) <= self.advantage_epsilon else 0.0
                for sample in all_train_samples
            ]),
            "degenerate_group_ratio": stable_mean([
                1.0 if sample.group_std_return <= self.advantage_epsilon else 0.0
                for sample in all_train_samples
            ]),
            "missing_logprob_ratio": stable_mean([
                1.0 if sample.logprob is None else 0.0 for sample in all_train_samples
            ], 1.0),
        }

    def _build_advantage_matrices(
        self,
        train_samples_by_agent: Dict[int, List[EngineTrainSample]],
    ) -> Dict[str, Any]:
        grouped: Dict[str, Dict[int, Dict[int, Dict[str, float]]]] = {}

        for agent_idx, samples in train_samples_by_agent.items():
            for sample in samples:
                node_entry = grouped.setdefault(sample.node_id, {})
                branch_entry = node_entry.setdefault(sample.branch_idx, {})
                branch_entry[agent_idx] = {
                    "outer_advantage": float(sample.metadata.get("outer_advantage", 0.0)),
                    "inner_advantage": float(sample.metadata.get("inner_advantage", 0.0)),
                    "scaled_inner_advantage": float(sample.metadata.get("scaled_inner_advantage", 0.0)),
                    "combined_advantage": float(
                        sample.metadata.get("combined_advantage", sample.normalized_advantage)
                    ),
                }

        node_ids = sorted(grouped.keys())

        outer_matrix: List[List[List[float]]] = []
        inner_matrix: List[List[List[float]]] = []
        scaled_inner_matrix: List[List[List[float]]] = []
        combined_matrix: List[List[List[float]]] = []

        for node_id in node_ids:
            per_branch = grouped[node_id]
            branch_ids = sorted(per_branch.keys())

            node_outer: List[List[float]] = []
            node_inner: List[List[float]] = []
            node_scaled_inner: List[List[float]] = []
            node_combined: List[List[float]] = []

            for branch_idx in branch_ids:
                per_agent = per_branch[branch_idx]
                agent_ids = sorted(per_agent.keys())

                node_outer.append([
                    float(per_agent[agent_idx]["outer_advantage"])
                    for agent_idx in agent_ids
                ])
                node_inner.append([
                    float(per_agent[agent_idx]["inner_advantage"])
                    for agent_idx in agent_ids
                ])
                node_scaled_inner.append([
                    float(per_agent[agent_idx]["scaled_inner_advantage"])
                    for agent_idx in agent_ids
                ])
                node_combined.append([
                    float(per_agent[agent_idx]["combined_advantage"])
                    for agent_idx in agent_ids
                ])

            outer_matrix.append(node_outer)
            inner_matrix.append(node_inner)
            scaled_inner_matrix.append(node_scaled_inner)
            combined_matrix.append(node_combined)

        return {
            "node_ids": node_ids,
            "outer_advantage_matrix": outer_matrix,
            "inner_advantage_matrix": inner_matrix,
            "scaled_inner_advantage_matrix": scaled_inner_matrix,
            "combined_advantage_matrix": combined_matrix,
        }