from __future__ import annotations

from core.experiment import run_experiment
from experiment.gig_grpo_experiment_base import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PLOT_WINDOW,
    build_experiment_spec,
)


def main() -> None:
    spec = build_experiment_spec(
        rounds=40,
        plot_window=DEFAULT_PLOT_WINDOW,
        output_dir=DEFAULT_OUTPUT_DIR,
        save_adapters=False,
        reset_mode="base",
    )
    run_experiment(spec)


if __name__ == "__main__":
    main()