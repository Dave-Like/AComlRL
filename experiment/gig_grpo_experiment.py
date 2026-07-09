from __future__ import annotations

from pathlib import Path

from experiment.gig_grpo_experiment_base import DEFAULT_PLOT_WINDOW, run_experiment


DEFAULT_OUTPUT_DIR = Path("outputs") / "gig_grpo_experiment"


if __name__ == "__main__":
    run_experiment(
        rounds=20,
        plot_window=DEFAULT_PLOT_WINDOW,
        output_dir=DEFAULT_OUTPUT_DIR,
        save_adapters=False,
        reset_mode="base",
    )
