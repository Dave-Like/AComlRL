from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt


@dataclass(slots=True)
class RLSeries:
    """单条强化学习曲线的数据载体。"""

    label: str
    rewards: List[float]


@dataclass(slots=True)
class RLPlotResult:
    """统一返回绘图结果，便于实验代码继续处理。"""

    figure: plt.Figure
    axes: List[plt.Axes]
    output_path: str | None = None


def ensure_float_list(values: Iterable[float | int]) -> List[float]:
    return [float(value) for value in values]


def cumulative_sum(values: Sequence[float | int]) -> List[float]:
    totals: List[float] = []
    running_total = 0.0
    for value in values:
        running_total += float(value)
        totals.append(running_total)
    return totals


def moving_average(
    values: Sequence[float | int],
    window_size: int = 10,
) -> List[float]:
    if window_size < 1:
        raise ValueError("window_size must be >= 1.")
    numeric_values = ensure_float_list(values)
    if not numeric_values:
        return []

    averages: List[float] = []
    running_total = 0.0
    for index, value in enumerate(numeric_values):
        running_total += value
        if index >= window_size:
            running_total -= numeric_values[index - window_size]
        current_window_size = min(index + 1, window_size)
        averages.append(running_total / current_window_size)
    return averages


def episode_indices(length: int) -> List[int]:
    if length < 0:
        raise ValueError("length must be >= 0.")
    return list(range(1, length + 1))


def build_series(label: str, rewards: Sequence[float | int]) -> RLSeries:
    return RLSeries(label=str(label), rewards=ensure_float_list(rewards))


def plot_training_curves(
    rewards: Sequence[float | int],
    *,
    window_size: int = 10,
    title_prefix: str = "RL Training Metrics",
    save_path: str | Path | None = None,
    show: bool = False,
    figsize: tuple[int, int] = (10, 12),
    x_values: Sequence[float | int] | None = None,
    x_label: str = "Step",
) -> RLPlotResult:
    """绘制单实验常用强化学习曲线。"""
    reward_values = ensure_float_list(rewards)
    if x_values is None:
        resolved_x_values = episode_indices(len(reward_values))
    else:
        resolved_x_values = ensure_float_list(x_values)

    cumulative_rewards = cumulative_sum(reward_values)
    averaged_rewards = moving_average(reward_values, window_size=window_size)

    figure, axes_array = plt.subplots(3, 1, figsize=figsize, sharex=True)
    axes = list(axes_array)

    axes[0].plot(resolved_x_values, reward_values, color="tab:blue", linewidth=1.8)
    axes[0].set_title(f"{title_prefix} - Step Value")
    axes[0].set_ylabel("Value")
    axes[0].grid(True, linestyle="--", alpha=0.35)

    axes[1].plot(resolved_x_values, averaged_rewards, color="tab:orange", linewidth=2.0)
    axes[1].set_title(f"{title_prefix} - Moving Average (window={window_size})")
    axes[1].set_ylabel("Moving Avg")
    axes[1].grid(True, linestyle="--", alpha=0.35)

    axes[2].plot(resolved_x_values, cumulative_rewards, color="tab:green", linewidth=2.0)
    axes[2].set_title(f"{title_prefix} - Cumulative Value")
    axes[2].set_xlabel(x_label)
    axes[2].set_ylabel("Cumulative")
    axes[2].grid(True, linestyle="--", alpha=0.35)

    figure.tight_layout()
    output_path = _finalize_plot(figure, save_path=save_path, show=show)
    return RLPlotResult(figure=figure, axes=axes, output_path=output_path)


def plot_multi_training_curves(
    series_list: Sequence[RLSeries] | Mapping[str, Sequence[float | int]],
    *,
    window_size: int = 10,
    title_prefix: str = "RL Training Comparison",
    save_path: str | Path | None = None,
    show: bool = False,
    figsize: tuple[int, int] = (10, 12),
) -> RLPlotResult:
    """绘制多实验对比曲线。"""
    normalized_series = _normalize_series_input(series_list)

    figure, axes_array = plt.subplots(3, 1, figsize=figsize, sharex=False)
    axes = list(axes_array)

    for series in normalized_series:
        x_values = episode_indices(len(series.rewards))
        axes[0].plot(x_values, series.rewards, linewidth=1.8, label=series.label)
        axes[1].plot(
            x_values,
            moving_average(series.rewards, window_size=window_size),
            linewidth=2.0,
            label=series.label,
        )
        axes[2].plot(
            x_values,
            cumulative_sum(series.rewards),
            linewidth=2.0,
            label=series.label,
        )

    axes[0].set_title(f"{title_prefix} - Episode Reward")
    axes[0].set_ylabel("Reward")
    axes[1].set_title(f"{title_prefix} - Moving Average Reward (window={window_size})")
    axes[1].set_ylabel("Avg Reward")
    axes[2].set_title(f"{title_prefix} - Cumulative Reward")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Cumulative Reward")

    for axis in axes:
        axis.grid(True, linestyle="--", alpha=0.35)
        axis.legend()

    figure.tight_layout()
    output_path = _finalize_plot(figure, save_path=save_path, show=show)
    return RLPlotResult(figure=figure, axes=axes, output_path=output_path)


def summarize_rewards(
    rewards: Sequence[float | int],
    *,
    window_size: int = 10,
) -> dict[str, float]:
    reward_values = ensure_float_list(rewards)
    if not reward_values:
        return {
            "num_episodes": 0.0,
            "mean_reward": 0.0,
            "max_reward": 0.0,
            "min_reward": 0.0,
            "final_reward": 0.0,
            "final_moving_average": 0.0,
            "cumulative_reward": 0.0,
        }

    averaged_rewards = moving_average(reward_values, window_size=window_size)
    return {
        "num_episodes": float(len(reward_values)),
        "mean_reward": float(sum(reward_values) / len(reward_values)),
        "max_reward": float(max(reward_values)),
        "min_reward": float(min(reward_values)),
        "final_reward": float(reward_values[-1]),
        "final_moving_average": float(averaged_rewards[-1]),
        "cumulative_reward": float(sum(reward_values)),
    }


def _normalize_series_input(
    series_list: Sequence[RLSeries] | Mapping[str, Sequence[float | int]],
) -> List[RLSeries]:
    if isinstance(series_list, Mapping):
        return [
            build_series(label=label, rewards=rewards)
            for label, rewards in series_list.items()
        ]

    normalized: List[RLSeries] = []
    for item in series_list:
        if isinstance(item, RLSeries):
            normalized.append(item)
        else:
            raise TypeError("series_list must contain RLSeries objects.")
    return normalized


def _finalize_plot(
    figure: plt.Figure,
    *,
    save_path: str | Path | None,
    show: bool,
) -> str | None:
    output_path: str | None = None
    if save_path is not None:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=200, bbox_inches="tight")
        output_path = str(output)
    if show:
        plt.show()
    return output_path


__all__ = [
    "RLSeries",
    "RLPlotResult",
    "build_series",
    "cumulative_sum",
    "episode_indices",
    "moving_average",
    "plot_multi_training_curves",
    "plot_training_curves",
    "summarize_rewards",
]
