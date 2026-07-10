from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

from core.common.types import EngineUpdateResult, RolloutBatch
from core.environment.coop_human_env import CoopHumanEnv
from core.plot.plot_tool import plot_training_curves, summarize_rewards
from core.trainers.stack_builder import AlgorithmStack, build_algorithm_stack

from core.experiment.spec import ExperimentSpec, MetricSeriesSpec


def run_experiment(spec: ExperimentSpec) -> Dict[str, Any]:
    """
    统一实验入口。

    设计目标：
    - experiment 层只负责组装 spec
    - runner 层负责标准执行流程
    - 算法差异尽量通过 stack_builder / hooks / updater 能力探测处理
    """

    _validate_spec(spec)

    output_path = spec.output_path()
    output_path.mkdir(parents=True, exist_ok=True)

    logger = ExperimentLogger(
        output_dir=output_path,
        filename=spec.io.log_filename,
        enable_console=spec.io.enable_console_log,
        enable_file=spec.io.enable_file_log,
        overwrite=spec.io.overwrite_log,
    )

    if spec.hooks.before_run is not None:
        spec.hooks.before_run(spec=spec, output_path=output_path, logger=logger)

    logger.log(f"Starting experiment: {spec.name}")
    logger.log(
        {
            "experiment": spec.to_metadata_dict(),
            "config": _safe_asdict(spec.config),
        }
    )

    logger.log("Stage: building env")
    env = CoopHumanEnv(
        formatters=spec.formatters,
        reward_func=spec.reward_func,
        transition_fn=spec.transition_fn,
        num_turns=spec.num_turns,
        reward_processor=spec.reward_processor,
    )
    logger.log("Stage: env built")

    logger.log("Stage: building models")
    agents, tokenizers = _build_models(spec)
    logger.log("Stage: models built")

    logger.log("Stage: building stack")
    stack = _build_stack(
        spec=spec,
        agents=agents,
        tokenizers=tokenizers,
        env=env,
    )
    logger.log("Stage: stack built")

    metric_series_specs = spec.resolve_metric_series()
    metric_history = {series.name: [] for series in metric_series_specs}
    round_records: List[Dict[str, Any]] = []

    for round_idx in range(1, spec.runtime.rounds + 1):
        if spec.hooks.before_round is not None:
            spec.hooks.before_round(
                spec=spec,
                round_idx=round_idx,
                stack=stack,
                logger=logger,
            )

        summary, update_result = run_experiment_round(stack=stack, spec=spec)
        metrics = dict(update_result.metrics)

        if spec.hooks.round_metrics_transform is not None:
            metrics = dict(spec.hooks.round_metrics_transform(metrics, spec))

        for series in metric_series_specs:
            metric_history[series.name].append(_resolve_metric_value(metrics, series))

        record = {
            "round": round_idx,
            **summary,
            "updated": update_result.updated,
        }
        for key in spec.round_record_metric_keys:
            record[key] = metrics.get(key)

        round_records.append(record)
        logger.log(record)

        if spec.hooks.after_round is not None:
            spec.hooks.after_round(
                spec=spec,
                round_idx=round_idx,
                stack=stack,
                logger=logger,
                summary=summary,
                update_result=update_result,
                record=record,
            )

    if spec.io.save_plots:
        _plot_metric_series(
            output_dir=output_path,
            metric_series_specs=metric_series_specs,
            metric_history=metric_history,
            plot_window=spec.runtime.plot_window,
            show=spec.io.plot_show,
        )

    if spec.runtime.save_adapters:
        _save_adapters(
            output_dir=output_path / spec.io.adapter_subdir,
            models=agents,
            tokenizers=tokenizers,
        )

    summary = build_experiment_summary(
        spec=spec,
        output_path=output_path,
        metric_series_specs=metric_series_specs,
        metric_history=metric_history,
        round_records=round_records,
        log_path=logger.log_path,
    )

    if spec.hooks.summary_transform is not None:
        summary = dict(spec.hooks.summary_transform(summary, spec))

    logger.log("Experiment finished.")
    if "reward_summary" in summary:
        logger.log(summary["reward_summary"])
    else:
        logger.log(
            {
                "available_summaries": [
                    key for key in summary.keys() if key.endswith("_summary")
                ]
            }
        )

    if spec.hooks.after_run is not None:
        spec.hooks.after_run(
            spec=spec,
            output_path=output_path,
            logger=logger,
            summary=summary,
            stack=stack,
        )

    return summary


def run_experiment_round(
    *,
    stack: AlgorithmStack,
    spec: ExperimentSpec,
) -> tuple[Dict[str, Any], EngineUpdateResult]:
    rollout_batches: List[RolloutBatch] = stack.trainer.collect_rollouts()
    stack.trainer.epoch_idx += 1
    update_batches = stack.trainer.build_update_batches(rollout_batches)

    train_samples_by_agent = stack.engine.build_train_samples(update_batches)

    train_sample_transform = spec.hooks.train_sample_transform
    if train_sample_transform is None:
        train_sample_transform = _resolve_default_train_sample_transform(spec.algorithm_name)
    if train_sample_transform is not None:
        train_sample_transform(train_samples_by_agent, spec)

    policy_train_data = _build_policy_train_data(
        stack=stack,
        train_samples_by_agent=train_samples_by_agent,
    )
    metrics = _build_engine_metrics(
        stack=stack,
        update_batches=update_batches,
        policy_train_data=policy_train_data,
    )

    updated = _is_policy_update_ready(
        stack=stack,
        policy_train_data=policy_train_data,
    )
    status = "policy_skeleton_ready"

    if updated:
        update_metrics = _run_policy_update(
            stack=stack,
            policy_train_data=policy_train_data,
        )
        if isinstance(update_metrics, dict):
            metrics.update(update_metrics)
        status = "updated"

    result = stack.engine.build_update_result(
        updated=updated,
        update_batches=update_batches,
        metrics=metrics,
        metadata={
            "engine_class": stack.engine.__class__.__name__,
            "status": status,
            "config": _safe_asdict(stack.engine.config),
        },
    )
    stack.trainer.last_update_result = result

    summary = {
        "epoch_idx": stack.trainer.epoch_idx,
        "num_rollout_batches": len(rollout_batches),
        "num_nodes": sum(len(batch.nodes) for batch in rollout_batches),
        "num_branch_steps": sum(
            node.num_branches
            for batch in rollout_batches
            for node in batch.nodes
        ),
        "num_update_batches": len(update_batches),
    }
    return summary, result


def build_experiment_summary(
    *,
    spec: ExperimentSpec,
    output_path: Path,
    metric_series_specs: Sequence[MetricSeriesSpec],
    metric_history: Dict[str, List[float]],
    round_records: Sequence[Dict[str, Any]],
    log_path: Path,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "experiment_name": spec.name,
        "algorithm_name": spec.algorithm_name,
        "config": _safe_asdict(spec.config),
        "spec_metadata": spec.to_metadata_dict(),
        "round_records": list(round_records),
        "plot_dir": str(output_path),
        "log_file": str(log_path),
    }

    for series in metric_series_specs:
        if series.enabled_for_summary:
            summary[f"{series.name}_summary"] = summarize_rewards(
                metric_history.get(series.name, []),
                window_size=spec.runtime.plot_window,
            )

    if "reward_summary" not in summary and metric_series_specs:
        first_series_name = metric_series_specs[0].name
        summary["reward_summary"] = summarize_rewards(
            metric_history.get(first_series_name, []),
            window_size=spec.runtime.plot_window,
        )

    summary["metric_history"] = {
        name: list(values) for name, values in metric_history.items()
    }
    return summary


class ExperimentLogger:
    """统一日志输出器，同时支持终端与文件。"""

    def __init__(
        self,
        *,
        output_dir: Path,
        filename: str,
        enable_console: bool = True,
        enable_file: bool = True,
        overwrite: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / filename
        self.enable_console = bool(enable_console)
        self.enable_file = bool(enable_file)

        if self.enable_file:
            mode = "w" if overwrite else "a"
            with self.log_path.open(mode, encoding="utf-8") as f:
                if overwrite:
                    f.write("")

    def log(self, message: Any) -> None:
        text = self._serialize(message)

        if self.enable_console:
            print(text)

        if self.enable_file:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(text + "\n")

    @staticmethod
    def _serialize(message: Any) -> str:
        if isinstance(message, (dict, list, tuple)):
            return json.dumps(message, ensure_ascii=False, indent=2)
        return str(message)


def inject_reference_logprob_proxy(
    train_samples_by_agent: Dict[int, List[Any]],
    spec: ExperimentSpec,
    *,
    base_offset: float = 0.05,
) -> None:
    """
    GIG-GRPO 的默认参考 logprob 占位逻辑。

    说明：
    - 这是算法兼容层逻辑
    - 后续如果引入正式 reference policy，可直接替换
    """
    for agent_idx, samples in train_samples_by_agent.items():
        for sample in samples:
            old_logprob = getattr(sample, "old_logprob", None)
            if old_logprob is None:
                continue

            branch_idx = int(getattr(sample, "branch_idx", 0))
            turn_idx = int(getattr(sample, "turn_idx", 0))
            offset = (
                base_offset
                + 0.01 * branch_idx
                + 0.005 * turn_idx
                + 0.0025 * int(agent_idx)
            )

            ref_logprob = float(old_logprob - offset)
            sample.ref_logprob = ref_logprob

            metadata = getattr(sample, "metadata", None)
            if isinstance(metadata, dict):
                metadata["ref_logprob"] = ref_logprob


def _build_models(spec: ExperimentSpec) -> tuple[Sequence[Any], Sequence[Any]]:
    if spec.model_builder is None:
        raise ValueError(
            "ExperimentSpec.model_builder is required. "
            "Please provide a model builder that returns `(agents, tokenizers)`."
        )

    models, tokenizers = spec.model_builder(spec)
    agents = list(models)
    tokenizer_list = list(tokenizers)

    if len(agents) != spec.config.num_agents:
        raise ValueError(
            f"Number of agents ({len(agents)}) does not match config.num_agents ({spec.config.num_agents})."
        )
    if len(tokenizer_list) != spec.config.num_agents:
        raise ValueError(
            f"Number of tokenizers ({len(tokenizer_list)}) does not match config.num_agents ({spec.config.num_agents})."
        )
    return agents, tokenizer_list


def _build_stack(
    *,
    spec: ExperimentSpec,
    agents: Sequence[Any],
    tokenizers: Sequence[Any],
    env: CoopHumanEnv,
) -> AlgorithmStack:
    if spec.stack_builder is not None:
        return spec.stack_builder(
            config=spec.config,
            agents=agents,
            tokenizers=tokenizers,
            env=env,
            train_dataset=spec.dataset,
            eval_dataset=spec.eval_dataset,
            evaluator=spec.evaluator,
            generation_kwargs=spec.runtime.generation_kwargs,
            sample_id_key=spec.runtime.sample_id_key,
            branch_selection=spec.runtime.branch_selection,
        )

    return build_algorithm_stack(
        algorithm_name=spec.algorithm_name,
        config=spec.config,
        agents=agents,
        tokenizers=tokenizers,
        env=env,
        train_dataset=spec.dataset,
        eval_dataset=spec.eval_dataset,
        evaluator=spec.evaluator,
        generation_kwargs=spec.runtime.generation_kwargs,
        sample_id_key=spec.runtime.sample_id_key,
        branch_selection=spec.runtime.branch_selection,
    )




def _resolve_default_train_sample_transform(algorithm_name: str):
    algorithm_name = str(algorithm_name).strip().lower()
    if algorithm_name == "gig_grpo":
        return inject_reference_logprob_proxy
    return None


def _build_policy_train_data(
    *,
    stack: AlgorithmStack,
    train_samples_by_agent: Dict[int, List[Any]],
) -> Any:
    """
    构建 policy updater 所需训练数据。

    策略：
    - 如果 updater 提供专用 builder，就优先使用
    - 否则直接回退为 train_samples_by_agent
    """
    policy_updater = getattr(stack.engine, "policy_updater", None)
    if policy_updater is None:
        return train_samples_by_agent

    build_candidates = [
        "build_train_data",
        "build_policy_train_data",
        "build_gig_train_samples",
        "build_magrpo_train_samples",
    ]

    for method_name in build_candidates:
        method = getattr(policy_updater, method_name, None)
        if callable(method):
            return method(train_samples_by_agent)

    return train_samples_by_agent


def _build_engine_metrics(
    *,
    stack: AlgorithmStack,
    update_batches: Sequence[Any],
    policy_train_data: Any,
) -> Dict[str, Any]:
    """
    构建指标。

    优先使用 engine 自身能力；若签名不兼容，则回退到更宽松的默认结果。
    """
    build_metrics = getattr(stack.engine, "_build_metrics", None)
    if not callable(build_metrics):
        return {}

    try:
        metrics = build_metrics(update_batches, policy_train_data)
        if isinstance(metrics, dict):
            return dict(metrics)
        return {}
    except TypeError:
        try:
            metrics = build_metrics(update_batches)
            if isinstance(metrics, dict):
                return dict(metrics)
            return {}
        except TypeError:
            return {}


def _is_policy_update_ready(
    *,
    stack: AlgorithmStack,
    policy_train_data: Any,
) -> bool:
    policy_updater = getattr(stack.engine, "policy_updater", None)
    if policy_updater is None:
        return False

    is_ready = getattr(policy_updater, "is_ready", None)
    if callable(is_ready):
        return bool(is_ready(policy_train_data))

    return bool(policy_train_data)


def _run_policy_update(
    *,
    stack: AlgorithmStack,
    policy_train_data: Any,
) -> Dict[str, Any]:
    policy_updater = getattr(stack.engine, "policy_updater", None)
    if policy_updater is None:
        return {}

    run_method = getattr(policy_updater, "run", None)
    if not callable(run_method):
        return {}

    result = run_method(policy_train_data)
    if isinstance(result, dict):
        return dict(result)
    return {}


def _plot_metric_series(
    *,
    output_dir: Path,
    metric_series_specs: Sequence[MetricSeriesSpec],
    metric_history: Dict[str, List[float]],
    plot_window: int,
    show: bool,
) -> None:
    step_values = list(range(1, max((len(values) for values in metric_history.values()), default=0) + 1))

    for series in metric_series_specs:
        if not series.enabled_for_plot:
            continue
        if not series.filename:
            continue

        values = metric_history.get(series.name, [])
        if not values:
            continue

        plot_training_curves(
            values,
            window_size=plot_window,
            title_prefix=series.title_prefix or series.name,
            save_path=output_dir / series.filename,
            show=show,
            x_values=step_values[: len(values)],
            x_label="Step",
        )


def _save_adapters(
    *,
    output_dir: Path,
    models: Sequence[Any],
    tokenizers: Sequence[Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, (model, tokenizer) in enumerate(zip(models, tokenizers)):
        adapter_dir = output_dir / f"agent_{idx}_lora"
        adapter_dir.mkdir(parents=True, exist_ok=True)

        if not hasattr(model, "save_pretrained"):
            raise TypeError(f"Model at index {idx} does not support `save_pretrained`.")
        if not hasattr(tokenizer, "save_pretrained"):
            raise TypeError(f"Tokenizer at index {idx} does not support `save_pretrained`.")

        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))


def _resolve_metric_value(metrics: Dict[str, Any], series: MetricSeriesSpec) -> float:
    keys = (series.metric_key, *series.fallback_keys)
    for key in keys:
        if key in metrics and metrics[key] is not None:
            return _coerce_finite_float(metrics[key], default=series.default_value)
    return float(series.default_value)


def _coerce_finite_float(value: Any, *, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(result):
        return float(default)
    return result


def _safe_asdict(value: Any) -> Any:
    try:
        return asdict(value)
    except TypeError:
        return value


def _validate_spec(spec: ExperimentSpec) -> None:
    if not isinstance(spec.name, str) or not spec.name.strip():
        raise ValueError("ExperimentSpec.name must be a non-empty string.")

    if not isinstance(spec.algorithm_name, str) or not spec.algorithm_name.strip():
        raise ValueError("ExperimentSpec.algorithm_name must be a non-empty string.")

    if spec.config is None:
        raise ValueError("ExperimentSpec.config must not be None.")

    if not spec.dataset:
        raise ValueError("ExperimentSpec.dataset must not be empty.")

    if not spec.formatters:
        raise ValueError("ExperimentSpec.formatters must not be empty.")

    if spec.reward_func is None or not callable(spec.reward_func):
        raise ValueError("ExperimentSpec.reward_func must be callable.")

    if spec.num_turns < 1:
        raise ValueError("ExperimentSpec.num_turns must be >= 1.")

    if spec.runtime.rounds < 1:
        raise ValueError("ExperimentSpec.runtime.rounds must be >= 1.")

    if spec.runtime.plot_window < 1:
        raise ValueError("ExperimentSpec.runtime.plot_window must be >= 1.")

    if len(spec.formatters) != int(spec.config.num_agents):
        raise ValueError(
            f"Number of formatters ({len(spec.formatters)}) does not match "
            f"config.num_agents ({spec.config.num_agents})."
        )

    if spec.num_turns > 1 and spec.transition_fn is None:
        raise ValueError(
            "ExperimentSpec.transition_fn is required when num_turns > 1."
        )

    if int(spec.config.max_turns) != int(spec.num_turns):
        raise ValueError(
            f"config.max_turns ({spec.config.max_turns}) must equal spec.num_turns ({spec.num_turns})."
        )

    for idx, item in enumerate(spec.dataset):
        if not isinstance(item, dict):
            raise TypeError(f"Dataset item at index {idx} must be a dict.")
        if "id" not in item:
            raise ValueError(f"Dataset item at index {idx} is missing `id`.")
        if "prompt" not in item:
            raise ValueError(f"Dataset item at index {idx} is missing `prompt`.")