from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional, Sequence

from core.config.config import GIG_GRPOConfig
from core.rlo_engine.gig_grpo import GIGGRPOEngine
from core.trainers.general_trainer import GeneralMultiAgentTrainer
from core.trainers.pipeline import PipelineManager


class GIGGRPOTrainer(GeneralMultiAgentTrainer):
    """面向 GIG-GRPO 的算法专属 trainer 封装。"""

    def __init__(
        self,
        *,
        pipeline: PipelineManager,
        config: GIG_GRPOConfig | None = None,
        engine: GIGGRPOEngine | None = None,
        evaluator: Optional[Any] = None,
        train_dataset: Optional[Sequence[Dict[str, Any]]] = None,
        eval_dataset: Optional[Sequence[Dict[str, Any]]] = None,
        batch_size: Optional[int] = None,
        discount: Optional[float] = None,
        normalize_advantages: Optional[bool] = None,
        **config_overrides: Any,
    ) -> None:
        resolved_config = config or GIG_GRPOConfig()
        explicit_overrides = {
            key: value
            for key, value in {
                "batch_size": batch_size,
                "discount": discount,
                "normalize_advantages": normalize_advantages,
            }.items()
            if value is not None
        }
        merged_overrides = {**explicit_overrides, **config_overrides}
        if merged_overrides:
            resolved_config = GIG_GRPOConfig(
                **{**asdict(resolved_config), **merged_overrides}
            )

        resolved_engine = engine or GIGGRPOEngine(config=resolved_config)
        super().__init__(
            pipeline=pipeline,
            engine=resolved_engine,
            evaluator=evaluator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            config=resolved_config,
        )
        self.config = resolved_config
        self.engine = resolved_engine
