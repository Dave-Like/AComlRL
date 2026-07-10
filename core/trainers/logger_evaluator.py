from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from core.common.types import EvalRecord, EvaluatorOutput, RolloutBatch
from core.environment.che_rollout import CHEEpisodeRollout


class LoggerEvaluator:
    """
    轻量评估器与日志聚合网关。

    当前版本适配 group rollout：每个 turn 会为每个 branch、每个 agent
    生成一条 `EvalRecord`。
    """

    def __init__(
        self,
        *,
        rollout: Optional[CHEEpisodeRollout] = None,
        default_num_generations: Optional[int] = None,
        default_generation_kwargs: Optional[Dict[str, object]] = None,
        default_max_turns: Optional[int] = None,
    ) -> None:
        self.rollout = rollout
        self.default_num_generations = default_num_generations
        self.default_generation_kwargs = dict(default_generation_kwargs or {})
        self.default_max_turns = default_max_turns
        self.last_output: Optional[EvaluatorOutput] = None

    def evaluate(
        self,
        items: Sequence[Dict[str, object]],
        *,
        num_generations: Optional[int] = None,
        generation_kwargs: Optional[Dict[str, object]] = None,
        max_turns: Optional[int] = None,
    ) -> EvaluatorOutput:
        if self.rollout is None:
            raise ValueError("rollout is required to evaluate raw items.")

        rollout_batches = [
            self.rollout.rollout_episode(
                item=item,
                num_generations=self._resolve_num_generations(num_generations),
                generation_kwargs=self._resolve_generation_kwargs(generation_kwargs),
                max_turns=self._resolve_max_turns(max_turns),
            )
            for item in items
        ]
        return self.evaluate_rollout_batches(rollout_batches)

    def evaluate_rollout_batches(
        self,
        rollout_batches: Sequence[RolloutBatch],
    ) -> EvaluatorOutput:
        records: List[EvalRecord] = []
        for rollout_batch in rollout_batches:
            records.extend(self._build_records_for_rollout_batch(rollout_batch))

        output = EvaluatorOutput(
            records=records,
            aggregated_metrics=self._aggregate_records(records, rollout_batches),
            metadata={
                "num_rollout_batches": len(rollout_batches),
                "num_records": len(records),
            },
        )
        self.last_output = output
        return output

    def evaluate_one_rollout_batch(
        self,
        rollout_batch: RolloutBatch,
    ) -> EvaluatorOutput:
        return self.evaluate_rollout_batches([rollout_batch])

    def get_last_output(self) -> Optional[EvaluatorOutput]:
        return self.last_output

    def _build_records_for_rollout_batch(
        self,
        rollout_batch: RolloutBatch,
    ) -> List[EvalRecord]:
        records: List[EvalRecord] = []
        for node in rollout_batch.nodes:
            for branch_idx, branch_prompts in enumerate(
                node.prompts_per_branch_per_agent
            ):
                reward = self._resolve_branch_reward(node.joint_rewards, branch_idx)
                return_ = self._resolve_branch_return(node.joint_returns, branch_idx)
                branch_actions = self._resolve_branch_actions(
                    node.actions_per_branch_per_agent,
                    branch_idx,
                )
                for agent_idx, prompt in enumerate(branch_prompts):
                    completion = (
                        branch_actions[agent_idx]
                        if agent_idx < len(branch_actions)
                        else ""
                    )
                    records.append(
                        EvalRecord(
                            sample_id=rollout_batch.sample_id,
                            episode_id=rollout_batch.episode_id,
                            agent_idx=agent_idx,
                            turn_idx=node.turn_idx,
                            prompt=prompt,
                            completion=completion,
                            reward=reward,
                            return_=return_,
                            metrics={
                                "branch_idx": float(branch_idx),
                                "num_branches": float(node.num_branches),
                            },
                            metadata={
                                "node_id": node.node_id,
                                "terminal": node.terminal,
                                "branch_idx": branch_idx,
                            },
                        )
                    )
        return records

    def _aggregate_records(
        self,
        records: Sequence[EvalRecord],
        rollout_batches: Sequence[RolloutBatch],
    ) -> Dict[str, float]:
        rewards = [record.reward for record in records if record.reward is not None]
        returns = [record.return_ for record in records if record.return_ is not None]
        completion_lengths = [
            float(len(record.completion))
            for record in records
            if record.completion is not None
        ]

        aggregated_metrics: Dict[str, float] = {
            "num_rollout_batches": float(len(rollout_batches)),
            "num_records": float(len(records)),
            "num_episodes": float(len({record.episode_id for record in records})),
        }
        aggregated_metrics["mean_reward"] = (
            sum(rewards) / len(rewards) if rewards else 0.0
        )
        aggregated_metrics["mean_return"] = (
            sum(returns) / len(returns) if returns else 0.0
        )
        aggregated_metrics["mean_completion_length"] = (
            sum(completion_lengths) / len(completion_lengths)
            if completion_lengths
            else 0.0
        )
        return aggregated_metrics

    @staticmethod
    def _resolve_branch_reward(
        joint_rewards: Sequence[float],
        branch_idx: int,
    ) -> Optional[float]:
        if branch_idx < 0 or branch_idx >= len(joint_rewards):
            return None
        return float(joint_rewards[branch_idx])

    @staticmethod
    def _resolve_branch_return(
        joint_returns: Sequence[float],
        branch_idx: int,
    ) -> Optional[float]:
        if branch_idx < 0 or branch_idx >= len(joint_returns):
            return None
        return float(joint_returns[branch_idx])

    @staticmethod
    def _resolve_branch_actions(
        actions_per_branch_per_agent: Sequence[Sequence[str]],
        branch_idx: int,
    ) -> List[str]:
        if branch_idx < 0 or branch_idx >= len(actions_per_branch_per_agent):
            return []
        return list(actions_per_branch_per_agent[branch_idx])

    def _resolve_generation_kwargs(
        self,
        generation_kwargs: Optional[Dict[str, object]],
    ) -> Dict[str, object]:
        resolved = dict(self.default_generation_kwargs)
        resolved.update(dict(generation_kwargs or {}))
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
