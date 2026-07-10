from __future__ import annotations

from pathlib import Path
from typing import Any

from core.config.config import GIG_GRPOConfig
from core.experiment import (
    ExperimentIOConfig,
    ExperimentRuntimeConfig,
    ExperimentSpec,
    run_experiment as run_spec_experiment,
)
from core.experiment.che_build import (
    build_experiment_dataset,
    build_formatters,
    build_reward_function,
    build_transition_function,
)
from core.experiment.model_build import (
    DEFAULT_MODEL_NAME,
    build_models_for_spec,
)


DEFAULT_PLOT_WINDOW = 3
DEFAULT_OUTPUT_DIR = Path("outputs") / "gig_grpo_experiment"


def build_experiment_config() -> GIG_GRPOConfig:
    return GIG_GRPOConfig(
        num_agents=2,
        num_generations=6,
        max_turns=2,
        batch_size=1,
        discount=0.99,
        normalize_advantages=True,
        temperature=0.95,
        top_p=0.95,
        top_k=30,
        max_new_tokens=220,
        do_sample=True,
        joint_mode="aligned",
        learning_rate=1e-5,
        update_epochs=1,
        max_grad_norm=1.0,
        max_safe_kl=2.0,
        kl_coef=0.02,
        advantage_mode="zscore",
        outer_advantage_clip=5.0,
        inner_advantage_clip=3.0,
        combined_advantage_clip=5.0,
        inner_scale_mode="match_outer_mean_abs",
        min_inner_scale=0.5,
        max_inner_scale=3.0,
        inner_group_size=2,
        outer_group_size=6,
        contribution_mode="hybrid",
        task_combination="linear",
        contribution_lambda=1.25,
        contribution_mix_alpha=0.6,
        counterfactual_anchor_coef=0.25,
    )


def build_experiment_spec(
    *,
    rounds: int = 20,
    plot_window: int = DEFAULT_PLOT_WINDOW,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    save_adapters: bool = False,
    reset_mode: str = "base",
    agent_a_adapter_dir: str | Path | None = None,
    agent_b_adapter_dir: str | Path | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> ExperimentSpec:
    config = build_experiment_config()

    return ExperimentSpec(
        name="gig_grpo_experiment",
        algorithm_name="gig_grpo",
        config=config,
        dataset=build_experiment_dataset(),
        formatters=build_formatters(),
        reward_func=build_reward_function(),
        transition_fn=build_transition_function(),
        num_turns=config.max_turns,
        model_builder=build_models_for_spec,
        io=ExperimentIOConfig(
            output_dir=output_dir,
            log_filename="experiment_log.txt",
            enable_console_log=True,
            enable_file_log=True,
            overwrite_log=True,
            save_plots=True,
            plot_show=False,
            adapter_subdir="adapters",
        ),
        runtime=ExperimentRuntimeConfig(
            rounds=rounds,
            plot_window=plot_window,
            save_adapters=save_adapters,
            sample_id_key="id",
            branch_selection="max_reward",
            generation_kwargs={},
        ),
        metadata={
            "model_name": model_name,
            "reset_mode": reset_mode,
            "agent_adapter_dirs": [
                agent_a_adapter_dir,
                agent_b_adapter_dir,
            ],
            "experiment_family": "gig_grpo",
            "env_family": "che",
        },
    )


def run_experiment(
    *,
    rounds: int = 20,
    plot_window: int = DEFAULT_PLOT_WINDOW,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    save_adapters: bool = False,
    reset_mode: str = "base",
    agent_a_adapter_dir: str | Path | None = None,
    agent_b_adapter_dir: str | Path | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> dict[str, Any]:
    spec = build_experiment_spec(
        rounds=rounds,
        plot_window=plot_window,
        output_dir=output_dir,
        save_adapters=save_adapters,
        reset_mode=reset_mode,
        agent_a_adapter_dir=agent_a_adapter_dir,
        agent_b_adapter_dir=agent_b_adapter_dir,
        model_name=model_name,
    )
    return run_spec_experiment(spec)