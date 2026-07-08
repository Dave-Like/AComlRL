from __future__ import annotations

import ast
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizerBase

from core.common.types import EngineTrainSample, EngineUpdateResult, RolloutBatch
from core.config.config import GIG_GRPOConfig
from core.environment.coop_human_env import CoopHumanEnv
from core.plot.plot_tool import plot_training_curves, summarize_rewards
from core.trainers.stack_builder import AlgorithmStack, build_gig_grpo_stack


QWEN_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_TORCH_DTYPE = torch.bfloat16
DEFAULT_DEVICE_MAP = "auto"
DEFAULT_PLOT_WINDOW = 5


def build_experiment_dataset() -> List[Dict[str, object]]:
    return [
        {
            "id": "task-1",
            "prompt": "Implement a Python API `normalize_and_group_words(words)` that groups words by normalized lowercase form, removes punctuation, and keeps original spelling order.",
            "entry_function": "normalize_and_group_words",
            "required_helpers": ["normalize_word", "group_in_order"],
            "reward_keywords": ["defaultdict", "isalpha", "append", "lower", "return"],
        },
        {
            "id": "task-2",
            "prompt": "Implement a Python API `merge_user_events(events)` that sorts by timestamp, merges adjacent events with the same user id, and returns session summaries.",
            "entry_function": "merge_user_events",
            "required_helpers": ["sort_events", "merge_adjacent_events"],
            "reward_keywords": ["sorted", "lambda", "for", "return", "duration"],
        },
        {
            "id": "task-3",
            "prompt": "Implement a Python API `rank_search_results(records, query)` that tokenizes the query, scores records by token overlap plus recency bonus, and returns top records.",
            "entry_function": "rank_search_results",
            "required_helpers": ["tokenize_query", "score_record"],
            "reward_keywords": ["sorted", "set", "split", "return", "score"],
        },
    ]


def build_reward_function() -> Callable[..., float]:
    def reward_func(*completion_args, batch_items=None):
        item = (batch_items or [{}])[0]
        helper_code = completion_args[0][0] if len(completion_args) > 0 and completion_args[0] else ""
        main_code = completion_args[1][0] if len(completion_args) > 1 and completion_args[1] else ""
        required_helpers = [str(name) for name in item.get("required_helpers", [])]
        entry_function = str(item.get("entry_function", ""))
        reward_keywords = [str(keyword) for keyword in item.get("reward_keywords", [])]
        score = 0.0
        max_score = 10.0 + len(required_helpers) * 1.5 + len(reward_keywords) * 0.3

        helper_tree = _safe_parse(helper_code)
        main_tree = _safe_parse(main_code)
        joint_tree = _safe_parse(f"{helper_code}\n\n{main_code}".strip())
        if helper_tree is not None:
            score += 1.5
        if main_tree is not None:
            score += 1.5
        if joint_tree is not None:
            score += 1.0

        helper_defs = _top_level_function_names(helper_tree)
        main_defs = _top_level_function_names(main_tree)
        main_calls = _called_function_names(main_tree)
        joint_calls = _called_function_names(joint_tree)
        for helper_name in required_helpers:
            if helper_name in helper_defs:
                score += 1.0
            if helper_name in main_calls or helper_name in joint_calls:
                score += 0.75
        if entry_function and entry_function in main_defs:
            score += 1.25
        if "return" in main_code:
            score += 0.4
        if any(token in main_code for token in ["for ", "sorted(", "dict", "set("]):
            score += 0.5
        if any(token in helper_code for token in ["def ", "return", "if ", "for "]):
            score += 0.5
        joined = f"{helper_code}\n{main_code}"
        for keyword in reward_keywords:
            if keyword in joined:
                score += 0.3
        if main_tree is not None and required_helpers and any(name in main_calls for name in required_helpers):
            score += 0.8
        if helper_tree is not None and len(helper_defs) >= len(required_helpers):
            score += 0.6
        return float(max(0.0, min(score / max(max_score, 1.0), 1.0)))

    return reward_func


def build_transition_function() -> Callable[..., List[str]]:
    def transition_fn(prompt, completions, prompt_hist, response_hist, item):
        helper_code = completions[0] if len(completions) > 0 else ""
        main_code = completions[1] if len(completions) > 1 else ""
        required_helpers = ", ".join(str(name) for name in item.get("required_helpers", []))
        entry_function = str(item.get("entry_function", ""))
        return [
            "\n".join([
                f"Original task:\n{item.get('prompt', prompt)}",
                f"Main draft from Agent B:\n{main_code}",
                f"Required helper functions: {required_helpers}",
                "Revise your helper-only code so it contains valid Python helper functions with clear names and useful control flow.",
            ]),
            "\n".join([
                f"Original task:\n{item.get('prompt', prompt)}",
                f"Helper draft from Agent A:\n{helper_code}",
                f"Your API must define `{entry_function}` and call these helpers when appropriate: {required_helpers}.",
                "Revise your main-only code so it stays executable, calls helper functions explicitly, and returns the final result.",
            ]),
        ]

    return transition_fn


def build_formatters() -> Sequence[Callable[[Dict[str, object]], str]]:
    def formatter_agent_0(item: Dict[str, object]) -> str:
        helper_names = ", ".join(str(name) for name in item.get("required_helpers", []))
        return "\n".join([
            "You are Agent A, a helper-function specialist.",
            "Write only helper functions in valid Python.",
            "Do not write the final public API function.",
            f"You should strongly prefer these helper names: {helper_names}.",
            f"Task:\n{item['prompt']}",
        ])

    def formatter_agent_1(item: Dict[str, object]) -> str:
        helper_names = ", ".join(str(name) for name in item.get("required_helpers", []))
        return "\n".join([
            "You are Agent B, a main-function integrator.",
            "Write only the public API function in valid Python.",
            "Do not redefine helper functions unless absolutely necessary.",
            f"Your function must call helper functions such as: {helper_names}.",
            f"Task:\n{item['prompt']}",
        ])

    return [formatter_agent_0, formatter_agent_1]


def build_quantization_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=DEFAULT_TORCH_DTYPE)


def build_lora_config() -> LoraConfig:
    return LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], bias="none")


def build_lora_model(model_name: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=build_quantization_config(), device_map=DEFAULT_DEVICE_MAP, torch_dtype=DEFAULT_TORCH_DTYPE, trust_remote_code=True)
    base_model = prepare_model_for_kbit_training(base_model)
    lora_model = get_peft_model(base_model, build_lora_config())
    lora_model.print_trainable_parameters()
    return lora_model, tokenizer


def build_experiment_env() -> CoopHumanEnv:
    return CoopHumanEnv(formatters=build_formatters(), reward_func=build_reward_function(), transition_fn=build_transition_function(), num_turns=2)


def build_experiment_config() -> GIG_GRPOConfig:
    return GIG_GRPOConfig(num_agents=2, num_generations=6, max_turns=2, batch_size=1, discount=0.99, normalize_advantages=True, temperature=0.95, top_p=0.95, top_k=30, max_new_tokens=220, do_sample=True, joint_mode="aligned", learning_rate=2e-5, update_epochs=1, max_grad_norm=1.0, advantage_mode="zscore", inner_group_size=2, outer_group_size=6, contribution_mode="task", task_combination="linear", contribution_lambda=1.25, contribution_mix_alpha=1.0, counterfactual_anchor_coef=0.25)

def inject_reference_logprob_proxy(
    train_samples_by_agent: Dict[int, List[EngineTrainSample]],
    *,
    base_offset: float = 0.05,
) -> None:
    for agent_idx, samples in train_samples_by_agent.items():
        for sample in samples:
            if sample.old_logprob is None:
                continue
            offset = (
                base_offset
                + 0.01 * sample.branch_idx
                + 0.005 * sample.turn_idx
                + 0.0025 * agent_idx
            )
            sample.ref_logprob = float(sample.old_logprob - offset)
            sample.metadata["ref_logprob"] = float(sample.ref_logprob)


def run_experiment_round(
    stack: AlgorithmStack,
) -> tuple[dict[str, Any], EngineUpdateResult]:
    rollout_batches: List[RolloutBatch] = stack.trainer.collect_rollouts()
    stack.trainer.epoch_idx += 1
    update_batches = stack.trainer.build_update_batches(rollout_batches)

    train_samples_by_agent = stack.engine.build_train_samples(update_batches)
    inject_reference_logprob_proxy(train_samples_by_agent)

    gig_train_samples_by_agent = stack.engine.policy_updater.build_gig_train_samples(
        train_samples_by_agent
    )
    metrics = stack.engine._build_metrics(update_batches, gig_train_samples_by_agent)

    updated = stack.engine.policy_updater.is_ready(gig_train_samples_by_agent)
    status = "policy_skeleton_ready"
    if updated:
        metrics.update(stack.engine.policy_updater.run(gig_train_samples_by_agent))
        status = "updated"

    result = stack.engine.build_update_result(
        updated=updated,
        update_batches=update_batches,
        metrics=metrics,
        metadata={
            "engine_class": stack.engine.__class__.__name__,
            "status": status,
            "config": asdict(stack.engine.config),
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


def plot_experiment_metrics(
    output_dir: Path,
    *,
    reward_history: Sequence[float],
    advantage_history: Sequence[float],
    inner_advantage_history: Sequence[float],
    kl_history: Sequence[float],
    loss_history: Sequence[float],
    window_size: int = DEFAULT_PLOT_WINDOW,
) -> None:
    plot_training_curves(
        reward_history,
        window_size=window_size,
        title_prefix="GIG-GRPO Task-Structured Reward",
        save_path=output_dir / "reward_curves.png",
        show=False,
    )
    plot_training_curves(
        advantage_history,
        window_size=window_size,
        title_prefix="GIG-GRPO Advantage",
        save_path=output_dir / "advantage_curves.png",
        show=False,
    )
    plot_training_curves(
        inner_advantage_history,
        window_size=window_size,
        title_prefix="GIG-GRPO Inner Advantage",
        save_path=output_dir / "inner_advantage_curves.png",
        show=False,
    )
    plot_training_curves(
        kl_history,
        window_size=window_size,
        title_prefix="GIG-GRPO Approx KL",
        save_path=output_dir / "kl_curves.png",
        show=False,
    )
    plot_training_curves(
        loss_history,
        window_size=window_size,
        title_prefix="GIG-GRPO Policy Loss",
        save_path=output_dir / "policy_loss_curves.png",
        show=False,
    )


def save_lora_adapters(
    output_dir: Path,
    models: Sequence[PreTrainedModel],
    tokenizers: Sequence[PreTrainedTokenizerBase],
) -> None:
    adapter_dirs = [
        output_dir / "agent_a_lora",
        output_dir / "agent_b_lora",
    ]
    for model, tokenizer, adapter_dir in zip(models, tokenizers, adapter_dirs):
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))


def run_experiment(
    *,
    rounds: int = 20,
    plot_window: int = DEFAULT_PLOT_WINDOW,
    output_dir: str | Path = Path("outputs") / "gig_grpo_experiment",
    save_adapters: bool = False,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = build_experiment_dataset()
    env = build_experiment_env()
    config = build_experiment_config()

    agent_a, tokenizer_a = build_lora_model(QWEN_MODEL_NAME)
    agent_b, tokenizer_b = build_lora_model(QWEN_MODEL_NAME)
    agents = [agent_a, agent_b]
    tokenizers = [tokenizer_a, tokenizer_b]

    stack = build_gig_grpo_stack(
        config=config,
        agents=agents,
        tokenizers=tokenizers,
        env=env,
        train_dataset=dataset,
    )

    reward_history: List[float] = []
    advantage_history: List[float] = []
    inner_advantage_history: List[float] = []
    kl_history: List[float] = []
    loss_history: List[float] = []
    round_records: List[dict[str, Any]] = []

    print("Starting task-structured GIG-GRPO experiment...")
    print(f"Config: {asdict(config)}")

    for round_idx in range(1, rounds + 1):
        summary, update_result = run_experiment_round(stack)
        metrics = update_result.metrics

        reward_history.append(float(metrics.get("mean_return", 0.0)))
        advantage_history.append(float(metrics.get("mean_advantage", 0.0)))
        inner_advantage_history.append(
            float(metrics.get("mean_inner_advantage", 0.0))
        )
        kl_history.append(
            float(
                metrics.get(
                    "mean_update_approx_kl",
                    metrics.get("mean_approx_kl", 0.0),
                )
            )
        )
        loss_history.append(float(metrics.get("mean_policy_loss", 0.0)))

        record = {
            "round": round_idx,
            **summary,
            "updated": update_result.updated,
            "mean_return": metrics.get("mean_return"),
            "mean_advantage": metrics.get("mean_advantage"),
            "mean_inner_advantage": metrics.get("mean_inner_advantage"),
            "mean_task_score": metrics.get("mean_task_score"),
            "mean_counterfactual_score": metrics.get("mean_counterfactual_score"),
            "mean_update_approx_kl": metrics.get(
                "mean_update_approx_kl",
                metrics.get("mean_approx_kl"),
            ),
            "mean_policy_loss": metrics.get("mean_policy_loss"),
            "mean_ratio": metrics.get("mean_ratio"),
            "positive_advantage_ratio": metrics.get("positive_advantage_ratio"),
        }
        round_records.append(record)
        print(record)

    plot_experiment_metrics(
        output_path,
        reward_history=reward_history,
        advantage_history=advantage_history,
        inner_advantage_history=inner_advantage_history,
        kl_history=kl_history,
        loss_history=loss_history,
        window_size=plot_window,
    )

    if save_adapters:
        save_lora_adapters(output_path / "adapters", agents, tokenizers)

    summary = {
        "config": asdict(config),
        "reward_summary": summarize_rewards(
            reward_history,
            window_size=plot_window,
        ),
        "advantage_summary": summarize_rewards(
            advantage_history,
            window_size=plot_window,
        ),
        "inner_advantage_summary": summarize_rewards(
            inner_advantage_history,
            window_size=plot_window,
        ),
        "kl_summary": summarize_rewards(
            kl_history,
            window_size=plot_window,
        ),
        "loss_summary": summarize_rewards(
            loss_history,
            window_size=plot_window,
        ),
        "round_records": round_records,
        "plot_dir": str(output_path),
    }

    print("Experiment finished.")
    print(summary["reward_summary"])
    return summary


def _safe_parse(code: str) -> ast.AST | None:
    try:
        return ast.parse(str(code or ""))
    except SyntaxError:
        return None


def _top_level_function_names(tree: ast.AST | None) -> set[str]:
    if tree is None:
        return set()
    return {
        str(node.name)
        for node in getattr(tree, "body", [])
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _called_function_names(tree: ast.AST | None) -> set[str]:
    if tree is None:
        return set()

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(str(node.func.id))
            elif isinstance(node.func, ast.Attribute):
                names.add(str(node.func.attr))
    return names