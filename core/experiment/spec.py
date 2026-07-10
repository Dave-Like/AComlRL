from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from core.config.config import BaseMultiAgentConfig


Formatter = Callable[[Dict[str, Any]], str]
RewardFunc = Callable[..., float | Sequence[float]]
TransitionFunc = Callable[
    [
        str,
        Sequence[str],
        Sequence[Sequence[str]],
        Sequence[Sequence[str]],
        Dict[str, Any],
    ],
    Sequence[str],
]

ModelBuilder = Callable[["ExperimentSpec"], tuple[Sequence[Any], Sequence[Any]]]
StackBuilder = Callable[..., Any]
TrainSampleTransform = Callable[[Dict[int, List[Any]], "ExperimentSpec"], None]
RoundMetricsTransform = Callable[[Dict[str, Any], "ExperimentSpec"], Dict[str, Any]]
SummaryTransform = Callable[[Dict[str, Any], "ExperimentSpec"], Dict[str, Any]]
HookFunc = Callable[..., None]


@dataclass(slots=True)
class MetricSeriesSpec:
    """定义一个需要记录、绘图、汇总的指标序列。"""

    name: str
    metric_key: str
    fallback_keys: Sequence[str] = field(default_factory=tuple)
    default_value: float = 0.0
    title_prefix: Optional[str] = None
    filename: Optional[str] = None
    enabled_for_plot: bool = True
    enabled_for_summary: bool = True


@dataclass(slots=True)
class ExperimentIOConfig:
    """实验输出与日志相关配置。"""

    output_dir: str | Path
    log_filename: str = "experiment_log.txt"
    enable_console_log: bool = True
    enable_file_log: bool = True
    overwrite_log: bool = True
    save_plots: bool = True
    plot_show: bool = False
    adapter_subdir: str = "adapters"


@dataclass(slots=True)
class ExperimentRuntimeConfig:
    """实验运行时配置。"""

    rounds: int = 20
    plot_window: int = 5
    save_adapters: bool = False
    sample_id_key: str = "id"
    branch_selection: str = "max_reward"
    generation_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExperimentHooks:
    """
    运行期钩子。

    设计目标：
    - 后续框架重构时，可逐步把更多行为迁移到 hooks
    - 尽量避免 runner 内部硬编码算法或实验特例
    """

    before_run: Optional[HookFunc] = None
    after_run: Optional[HookFunc] = None
    before_round: Optional[HookFunc] = None
    after_round: Optional[HookFunc] = None

    train_sample_transform: Optional[TrainSampleTransform] = None
    round_metrics_transform: Optional[RoundMetricsTransform] = None
    summary_transform: Optional[SummaryTransform] = None


@dataclass(slots=True)
class ExperimentSpec:
    """
    统一实验描述对象。

    说明：
    - experiment 层只需要构建并返回 ExperimentSpec
    - core.runner 负责统一执行
    - 后续如果框架进一步重构，可优先扩展这里，而不是让 experiment 脚本继续膨胀
    """

    name: str
    algorithm_name: str
    config: BaseMultiAgentConfig

    dataset: Sequence[Dict[str, Any]]
    formatters: Sequence[Formatter]
    reward_func: RewardFunc
    transition_fn: Optional[TransitionFunc] = None
    num_turns: int = 2
    reward_processor: Optional[Callable[[float], float]] = None

    model_builder: Optional[ModelBuilder] = None
    stack_builder: Optional[StackBuilder] = None

    evaluator: Optional[Any] = None
    eval_dataset: Optional[Sequence[Dict[str, Any]]] = None

    io: ExperimentIOConfig = field(
        default_factory=lambda: ExperimentIOConfig(output_dir="outputs/default_experiment")
    )
    runtime: ExperimentRuntimeConfig = field(default_factory=ExperimentRuntimeConfig)
    hooks: ExperimentHooks = field(default_factory=ExperimentHooks)

    metric_series: Sequence[MetricSeriesSpec] = field(default_factory=tuple)
    round_record_metric_keys: Sequence[str] = field(
        default_factory=lambda: (
            "mean_return",
            "mean_advantage",
            "mean_inner_advantage",
            "mean_task_score",
            "mean_counterfactual_score",
            "mean_update_approx_kl",
            "mean_policy_loss",
            "mean_ratio",
            "positive_advantage_ratio",
        )
    )

    metadata: Dict[str, Any] = field(default_factory=dict)

    def output_path(self) -> Path:
        return Path(self.io.output_dir)

    def resolve_metric_series(self) -> List[MetricSeriesSpec]:
        if self.metric_series:
            return list(self.metric_series)
        return build_default_metric_series(self.algorithm_name)

    def to_metadata_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "algorithm_name": self.algorithm_name,
            "num_turns": self.num_turns,
            "dataset_size": len(self.dataset),
            "runtime": {
                "rounds": self.runtime.rounds,
                "plot_window": self.runtime.plot_window,
                "save_adapters": self.runtime.save_adapters,
                "sample_id_key": self.runtime.sample_id_key,
                "branch_selection": self.runtime.branch_selection,
                "generation_kwargs": dict(self.runtime.generation_kwargs),
            },
            "io": {
                "output_dir": str(self.io.output_dir),
                "log_filename": self.io.log_filename,
                "enable_console_log": self.io.enable_console_log,
                "enable_file_log": self.io.enable_file_log,
                "overwrite_log": self.io.overwrite_log,
                "save_plots": self.io.save_plots,
                "plot_show": self.io.plot_show,
                "adapter_subdir": self.io.adapter_subdir,
            },
            "metadata": dict(self.metadata),
        }


def build_default_metric_series(algorithm_name: str) -> List[MetricSeriesSpec]:
    algorithm_name = str(algorithm_name).strip().lower()
    series = [
        MetricSeriesSpec(
            name="reward",
            metric_key="mean_return",
            title_prefix=f"{algorithm_name.upper()} Reward",
            filename="reward_curves.png",
        ),
        MetricSeriesSpec(
            name="advantage",
            metric_key="mean_advantage",
            title_prefix=f"{algorithm_name.upper()} Advantage",
            filename="advantage_curves.png",
        ),
        MetricSeriesSpec(
            name="approx_kl",
            metric_key="mean_update_approx_kl",
            fallback_keys=("mean_approx_kl",),
            title_prefix=f"{algorithm_name.upper()} Approx KL",
            filename="kl_curves.png",
        ),
        MetricSeriesSpec(
            name="policy_loss",
            metric_key="mean_policy_loss",
            title_prefix=f"{algorithm_name.upper()} Policy Loss",
            filename="policy_loss_curves.png",
        ),
    ]

    if algorithm_name == "gig_grpo":
        series.insert(
            2,
            MetricSeriesSpec(
                name="inner_advantage",
                metric_key="mean_inner_advantage",
                title_prefix="GIG-GRPO Inner Advantage",
                filename="inner_advantage_curves.png",
            ),
        )

    return series