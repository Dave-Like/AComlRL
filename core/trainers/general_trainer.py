from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Iterator, List, Optional, Sequence

from core.common.types import EngineUpdateResult, RolloutBatch, UpdateBatch
from core.config.config import BaseMultiAgentConfig
from core.trainers.pipeline import PipelineManager


class GeneralMultiAgentTrainer:
    """
    第一阶段的通用多智能体 trainer 骨架。

    当前版本面向 group rollout：每个 `NodeSample` 内部都保留 G 条并行分支。
    trainer 负责收集 rollout、组织更新批，并把 branch 统计信息一并传给 engine。
    """

    def __init__(
        self,
        *,
        pipeline: PipelineManager,
        engine: Optional[Any] = None,
        evaluator: Optional[Any] = None,
        train_dataset: Optional[Sequence[Dict[str, Any]]] = None,
        eval_dataset: Optional[Sequence[Dict[str, Any]]] = None,
        config: BaseMultiAgentConfig | None = None,
        batch_size: Optional[int] = None,
        algorithm_name: Optional[str] = None,
        discount: Optional[float] = None,
        normalize_advantages: Optional[bool] = None,
    ) -> None:
        resolved_config = config or BaseMultiAgentConfig()
        config_overrides = {
            key: value
            for key, value in {
                "batch_size": batch_size,
                "algorithm_name": algorithm_name,
                "discount": discount,
                "normalize_advantages": normalize_advantages,
            }.items()
            if value is not None
        }
        if config_overrides:
            resolved_config = BaseMultiAgentConfig(
                **{**asdict(resolved_config), **config_overrides}
            )

        if resolved_config.batch_size < 1:
            raise ValueError("batch_size must be >= 1.")

        self.pipeline = pipeline
        self.engine = engine
        self.evaluator = evaluator
        self.train_dataset = list(train_dataset or [])
        self.eval_dataset = list(eval_dataset or [])
        self.config = resolved_config
        self.batch_size = int(self.config.batch_size)
        self.algorithm_name = str(self.config.algorithm_name)
        self.discount = float(self.config.discount)
        self.normalize_advantages = bool(self.config.normalize_advantages)

        self.global_step = 0
        self.epoch_idx = 0
        self.last_rollout_batches: List[RolloutBatch] = []
        self.last_update_batches: List[UpdateBatch] = []
        self.rollout_history: List[List[RolloutBatch]] = []
        self.last_update_result: Optional[EngineUpdateResult] = None

    def set_train_dataset(self, dataset: Sequence[Dict[str, Any]]) -> None:
        self.train_dataset = list(dataset)

    def set_eval_dataset(self, dataset: Sequence[Dict[str, Any]]) -> None:
        self.eval_dataset = list(dataset)

    def collect_rollouts(
        self,
        *,
        items: Optional[Sequence[Dict[str, Any]]] = None,
        num_generations: Optional[int] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        max_turns: Optional[int] = None,
    ) -> List[RolloutBatch]:
        source_items = list(self.train_dataset if items is None else items)
        if not source_items:
            self.last_rollout_batches = []
            return []

        collected: List[RolloutBatch] = []
        resolved_num_generations = self.config.num_generations if num_generations is None else num_generations
        resolved_max_turns = self.config.max_turns if max_turns is None else max_turns
        for batch_items in self._iter_batches(source_items):
            rollout_batches = self.pipeline.collect_batch(
                items=batch_items,
                num_generations=resolved_num_generations,
                generation_kwargs=generation_kwargs,
                max_turns=resolved_max_turns,
            )
            collected.extend(rollout_batches)
            self.global_step += len(rollout_batches)

        self.last_rollout_batches = collected
        self.rollout_history.append(list(collected))
        return collected

    def build_update_batches(
        self,
        rollout_batches: Sequence[RolloutBatch],
    ) -> List[UpdateBatch]:
        if not rollout_batches:
            self.last_update_batches = []
            return []

        num_agents = max((batch.num_agents for batch in rollout_batches), default=0)
        all_samples = [node for batch in rollout_batches for node in batch.nodes]
        total_num_branches = sum(
            node.num_branches for batch in rollout_batches for node in batch.nodes
        )
        update_batches = [
            UpdateBatch(
                agent_idx=agent_idx,
                algorithm_name=self.algorithm_name,
                samples=list(all_samples),
                discount=self.discount,
                normalize_advantages=self.normalize_advantages,
                metadata={
                    "num_rollout_batches": len(rollout_batches),
                    "num_samples": len(all_samples),
                    "total_num_branches": total_num_branches,
                    "trainer_config": asdict(self.config),
                },
            )
            for agent_idx in range(num_agents)
        ]
        self.last_update_batches = update_batches
        return update_batches

    def train_epoch(
        self,
        *,
        items: Optional[Sequence[Dict[str, Any]]] = None,
        num_generations: Optional[int] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        max_turns: Optional[int] = None,
        run_update: bool = False,
    ) -> Dict[str, Any]:
        rollout_batches = self.collect_rollouts(
            items=items,
            num_generations=num_generations,
            generation_kwargs=generation_kwargs,
            max_turns=max_turns,
        )
        self.epoch_idx += 1

        update_output = None
        if run_update:
            update_output = self.update_agents(rollout_batches)

        return {
            "epoch_idx": self.epoch_idx,
            "num_rollout_batches": len(rollout_batches),
            "num_nodes": sum(len(batch.nodes) for batch in rollout_batches),
            "num_branch_steps": sum(
                node.num_branches for batch in rollout_batches for node in batch.nodes
            ),
            "num_update_batches": len(self.last_update_batches),
            "update_output": update_output,
        }

    def update_agents(
        self,
        rollout_batches: Sequence[RolloutBatch],
    ) -> EngineUpdateResult:
        update_batches = self.build_update_batches(rollout_batches)
        if self.engine is None:
            result = EngineUpdateResult(
                algorithm_name=self.algorithm_name,
                updated=False,
                num_update_batches=len(update_batches),
                num_samples=sum(len(batch.samples) for batch in update_batches),
                metrics={},
                metadata={"reason": "engine_not_configured"},
            )
            self.last_update_result = result
            return result

        if not hasattr(self.engine, "update"):
            raise AttributeError("engine must define an 'update' method.")

        result = self.engine.update(update_batches)
        self.last_update_result = result
        return result

    def evaluate(
        self,
        *,
        items: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Any:
        if self.evaluator is None:
            return None
        eval_items = list(self.eval_dataset if items is None else items)
        if not hasattr(self.evaluator, "evaluate"):
            raise AttributeError("evaluator must define an 'evaluate' method.")
        return self.evaluator.evaluate(eval_items)

    def get_last_rollout_batches(self) -> List[RolloutBatch]:
        return list(self.last_rollout_batches)

    def get_last_update_batches(self) -> List[UpdateBatch]:
        return list(self.last_update_batches)

    def get_last_update_result(self) -> Optional[EngineUpdateResult]:
        return self.last_update_result

    def _iter_batches(
        self,
        items: Sequence[Dict[str, Any]],
    ) -> Iterator[List[Dict[str, Any]]]:
        for start_idx in range(0, len(items), self.batch_size):
            yield list(items[start_idx : start_idx + self.batch_size])
