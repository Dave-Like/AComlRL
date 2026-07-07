from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from core.config.config import BaseMultiAgentConfig, GIG_GRPOConfig, MAGRPOConfig
from core.environment.che_executor import CHEExecutor
from core.environment.che_handler import CoopHumanEnvHandler
from core.environment.che_rollout import (
    CHEEpisodeRollout,
    GIGGRPOEpisodeRollout,
    MAGRPOEpisodeRollout,
)
from core.environment.coop_human_env import CoopHumanEnv
from core.environment.generation_engine import LLMGenerationEngine
from core.rlo_engine.gig_grpo import GIGGRPOEngine
from core.rlo_engine.magrpo import MAGRPOEngine
from core.trainers.gig_grpo_trainer import GIGGRPOTrainer
from core.trainers.magrpo_trainer import MAGRPOTrainer
from core.trainers.pipeline import PipelineManager


@dataclass(slots=True)
class AlgorithmStack:
    """统一组装后的训练栈对象。"""

    config: BaseMultiAgentConfig
    generation_engine: LLMGenerationEngine
    executor: CHEExecutor
    handler: CoopHumanEnvHandler
    rollout: CHEEpisodeRollout
    pipeline: PipelineManager
    engine: Any
    trainer: Any


def build_magrpo_stack(
    *,
    config: MAGRPOConfig,
    agents: Sequence[PreTrainedModel],
    tokenizers: Sequence[PreTrainedTokenizerBase] | PreTrainedTokenizerBase,
    env: CoopHumanEnv,
    train_dataset: Optional[Sequence[Dict[str, Any]]] = None,
    eval_dataset: Optional[Sequence[Dict[str, Any]]] = None,
    evaluator: Optional[Any] = None,
    generation_kwargs: Optional[Dict[str, Any]] = None,
    sample_id_key: str = "id",
    branch_selection: str = "max_reward",
) -> AlgorithmStack:
    generation_engine = LLMGenerationEngine(
        agents=agents,
        tokenizers=tokenizers,
        config=config,
    )
    executor = CHEExecutor(
        env=env,
        generation_engine=generation_engine,
        config=config,
    )
    handler = CoopHumanEnvHandler(
        executor=executor,
        branch_selection=branch_selection,
    )
    rollout = MAGRPOEpisodeRollout(
        handler=handler,
        config=config,
        sample_id_key=sample_id_key,
    )
    pipeline = PipelineManager(
        rollout=rollout,
        config=config,
        default_generation_kwargs=generation_kwargs,
    )
    engine = MAGRPOEngine(config=config)
    engine.attach_policy_components(
        policy_models=agents,
        tokenizers=tokenizers,
    )
    trainer = MAGRPOTrainer(
        pipeline=pipeline,
        config=config,
        engine=engine,
        evaluator=evaluator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    return AlgorithmStack(
        config=config,
        generation_engine=generation_engine,
        executor=executor,
        handler=handler,
        rollout=rollout,
        pipeline=pipeline,
        engine=engine,
        trainer=trainer,
    )


def build_gig_grpo_stack(
    *,
    config: GIG_GRPOConfig,
    agents: Sequence[PreTrainedModel],
    tokenizers: Sequence[PreTrainedTokenizerBase] | PreTrainedTokenizerBase,
    env: CoopHumanEnv,
    train_dataset: Optional[Sequence[Dict[str, Any]]] = None,
    eval_dataset: Optional[Sequence[Dict[str, Any]]] = None,
    evaluator: Optional[Any] = None,
    generation_kwargs: Optional[Dict[str, Any]] = None,
    sample_id_key: str = "id",
    branch_selection: str = "max_reward",
) -> AlgorithmStack:
    generation_engine = LLMGenerationEngine(
        agents=agents,
        tokenizers=tokenizers,
        config=config,
    )
    executor = CHEExecutor(
        env=env,
        generation_engine=generation_engine,
        config=config,
    )
    handler = CoopHumanEnvHandler(
        executor=executor,
        branch_selection=branch_selection,
    )
    rollout = GIGGRPOEpisodeRollout(
        handler=handler,
        config=config,
        sample_id_key=sample_id_key,
    )
    pipeline = PipelineManager(
        rollout=rollout,
        config=config,
        default_generation_kwargs=generation_kwargs,
    )
    engine = GIGGRPOEngine(config=config)
    trainer = GIGGRPOTrainer(
        pipeline=pipeline,
        config=config,
        engine=engine,
        evaluator=evaluator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    return AlgorithmStack(
        config=config,
        generation_engine=generation_engine,
        executor=executor,
        handler=handler,
        rollout=rollout,
        pipeline=pipeline,
        engine=engine,
        trainer=trainer,
    )
