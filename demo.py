from __future__ import annotations

from dataclasses import asdict
from typing import Callable, Dict, List, Sequence

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from core.config.config import GIG_GRPOConfig
from core.environment.coop_human_env import CoopHumanEnv
from core.trainers.stack_builder import build_gig_grpo_stack


QWEN_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_TORCH_DTYPE = torch.bfloat16
DEFAULT_DEVICE_MAP = "auto"


def build_demo_dataset() -> List[Dict[str, object]]:
    return [
        {
            "id": "task-1",
            "prompt": "Write a Python function `two_sum(nums, target)` that returns the indices of two numbers whose sum equals target.",
            "reference_keywords": ["def two_sum", "return", "dict", "enumerate"],
        },
        {
            "id": "task-2",
            "prompt": "Write a Python function `is_palindrome(s)` that ignores non-alphanumeric characters and case.",
            "reference_keywords": ["def is_palindrome", "lower", "isalnum", "return"],
        },
    ]


def build_reward_function() -> Callable[..., float]:
    def reward_func(*completion_args, batch_items=None):
        item = (batch_items or [{}])[0]
        keywords = item.get("reference_keywords", [])
        joined = "\n".join(values[0] for values in completion_args if values)
        score = 0.0
        for keyword in keywords:
            if keyword in joined:
                score += 1.0
        if "def " in joined:
            score += 0.5
        if "return" in joined:
            score += 0.5
        return score / max(len(keywords), 1)

    return reward_func


def build_transition_function() -> Callable[..., List[str]]:
    def transition_fn(prompt, completions, prompt_hist, response_hist, item):
        next_prompts: List[str] = []
        for agent_idx, completion in enumerate(completions):
            peer_idx = 1 - agent_idx if len(completions) == 2 else agent_idx
            peer_completion = completions[peer_idx]
            next_prompts.append(
                "\n".join(
                    [
                        f"Original task:\n{item.get('prompt', prompt)}",
                        f"Your previous draft:\n{completion}",
                        f"Peer draft:\n{peer_completion}",
                        "Revise your solution to be more correct, concise, and executable.",
                    ]
                )
            )
        return next_prompts

    return transition_fn


def build_formatters() -> Sequence[Callable[[Dict[str, object]], str]]:
    def formatter_agent_0(item: Dict[str, object]) -> str:
        return "\n".join(
            [
                "You are Agent A, a Python coding specialist.",
                "Write a correct and executable solution.",
                f"Task:\n{item['prompt']}",
            ]
        )

    def formatter_agent_1(item: Dict[str, object]) -> str:
        return "\n".join(
            [
                "You are Agent B, a Python reviewer-coder.",
                "Produce a robust implementation with edge-case awareness.",
                f"Task:\n{item['prompt']}",
            ]
        )

    return [formatter_agent_0, formatter_agent_1]


def build_quantization_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=DEFAULT_TORCH_DTYPE,
    )


def build_lora_config() -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05, 
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
    )


def build_lora_model(model_name: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=build_quantization_config(),
        device_map=DEFAULT_DEVICE_MAP,
        torch_dtype=DEFAULT_TORCH_DTYPE,
        trust_remote_code=True,
    )
    base_model = prepare_model_for_kbit_training(base_model)
    lora_model = get_peft_model(base_model, build_lora_config())
    lora_model.print_trainable_parameters()
    return lora_model, tokenizer


def build_demo_env() -> CoopHumanEnv:
    return CoopHumanEnv(
        formatters=build_formatters(),
        reward_func=build_reward_function(),
        transition_fn=build_transition_function(),
        num_turns=1,
    )


def build_demo_config() -> GIG_GRPOConfig:
    return GIG_GRPOConfig(
        num_agents=2,
        num_generations=4,
        max_turns=1,
        batch_size=1,
        discount=0.99,
        normalize_advantages=True,
        temperature=0.7,
        top_p=0.9,
        top_k=20,
        max_new_tokens=128,
        do_sample=True,
        joint_mode="aligned",
        learning_rate=2e-5,
        update_epochs=1,
        max_grad_norm=1.0,
        inner_group_size=2,
        outer_group_size=4,
        contribution_mode="counterfactual",
        task_combination="linear",
        contribution_lambda=1.0,
        contribution_mix_alpha=0.0,
        counterfactual_anchor_coef=0.25,
    )


def save_lora_adapters(
    models: Sequence[PreTrainedModel],
    tokenizers: Sequence[PreTrainedTokenizerBase],
) -> None:
    output_dirs = [
        "outputs/gig_grpo_agent_a_lora",
        "outputs/gig_grpo_agent_b_lora",
    ]
    for model, tokenizer, output_dir in zip(models, tokenizers, output_dirs):
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)


def run_demo(rounds: int = 10) -> None:
    dataset = build_demo_dataset()
    env = build_demo_env()
    config = build_demo_config()

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

    print("Starting 24GB-friendly counterfactual GIG-GRPO demo training...")
    print(f"Config: {asdict(config)}")

    for round_idx in range(1, rounds + 1):
        output = stack.trainer.train_epoch(run_update=True)
        update_output = output.get("update_output")
        metrics = {} if update_output is None else update_output.metrics
        print(
            {
                "round": round_idx,
                "epoch_idx": output["epoch_idx"],
                "num_rollout_batches": output["num_rollout_batches"],
                "num_nodes": output["num_nodes"],
                "num_branch_steps": output["num_branch_steps"],
                "updated": None if update_output is None else update_output.updated,
                "mean_return": metrics.get("mean_return"),
                "mean_advantage": metrics.get("mean_advantage"),
                "mean_inner_advantage": metrics.get("mean_inner_advantage"),
                "mean_counterfactual_score": metrics.get("mean_counterfactual_score"),
                "mean_cf_cross": metrics.get("mean_cf_cross"),
                "mean_policy_loss": metrics.get("mean_policy_loss"),
                "mean_update_approx_kl": metrics.get("mean_update_approx_kl"),
            }
        )

    print("Saving LoRA adapters...")
    save_lora_adapters(agents, tokenizers)
    print("Demo finished.")


if __name__ == "__main__":
    run_demo(rounds=10)
