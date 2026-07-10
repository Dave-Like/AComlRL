from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from core.environment.reset import build_reset_config, resolve_reset_decision
from core.experiment.spec import ExperimentSpec


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_TORCH_DTYPE = torch.bfloat16
DEFAULT_DEVICE_MAP = "auto"


def build_quantization_config(
    *,
    load_in_4bit: bool = True,
    quant_type: str = "nf4",
    use_double_quant: bool = True,
    compute_dtype: torch.dtype = DEFAULT_TORCH_DTYPE,
) -> BitsAndBytesConfig:
    """
    构建默认量化配置。

    后续如果框架支持更多量化策略，可以继续扩展参数，
    不建议把量化细节散落到 experiment 脚本中。
    """
    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def build_lora_config(
    *,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[Sequence[str]] = None,
    bias: str = "none",
    task_type: TaskType = TaskType.CAUSAL_LM,
) -> LoraConfig:
    """
    构建默认 LoRA 配置。
    """
    resolved_target_modules = list(
        target_modules
        or [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    return LoraConfig(
        task_type=task_type,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=resolved_target_modules,
        bias=bias,
    )


def build_tokenizer(
    model_name: str,
    *,
    use_fast: bool = False,
    trust_remote_code: bool = True,
) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=use_fast,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def build_base_causal_lm(
    model_name: str,
    *,
    quantization_config: Optional[BitsAndBytesConfig] = None,
    device_map: str | Dict[str, Any] = DEFAULT_DEVICE_MAP,
    torch_dtype: torch.dtype = DEFAULT_TORCH_DTYPE,
    trust_remote_code: bool = True,
) -> PreTrainedModel:
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )


def build_lora_model(
    model_name: str,
    *,
    reset_mode: str = "base",
    adapter_dir: str | Path | None = None,
    quantization_config: Optional[BitsAndBytesConfig] = None,
    lora_config: Optional[LoraConfig] = None,
    device_map: str | Dict[str, Any] = DEFAULT_DEVICE_MAP,
    torch_dtype: torch.dtype = DEFAULT_TORCH_DTYPE,
    trust_remote_code: bool = True,
    prepare_kbit_training: bool = True,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    统一构建单个 agent 的 model + tokenizer。

    支持：
    - base: 从基础模型初始化新的可训练 LoRA
    - resume: 从 adapter 继续训练
    - eval: 加载 adapter，仅评估
    """
    reset_config = build_reset_config(
        mode=reset_mode,
        resume_adapter_dir=str(adapter_dir) if reset_mode == "resume" and adapter_dir is not None else None,
        eval_adapter_dir=str(adapter_dir) if reset_mode == "eval" and adapter_dir is not None else None,
    )
    decision = resolve_reset_decision(reset_config)

    tokenizer = build_tokenizer(
        model_name,
        use_fast=False,
        trust_remote_code=trust_remote_code,
    )

    base_model = build_base_causal_lm(
        model_name,
        quantization_config=quantization_config or build_quantization_config(),
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )

    if prepare_kbit_training:
        base_model = prepare_model_for_kbit_training(base_model)

    resolved_lora_config = lora_config or build_lora_config()

    if decision.attach_new_lora:
        model = get_peft_model(base_model, resolved_lora_config)
    else:
        if decision.load_adapter_dir is None:
            raise ValueError("Adapter directory is required when loading an existing LoRA adapter.")
        model = PeftModel.from_pretrained(
            base_model,
            str(decision.load_adapter_dir),
            is_trainable=decision.trainable,
        )

    if decision.trainable:
        model.train()
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()
    else:
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()

    return model, tokenizer


def build_models_for_spec(
    spec: ExperimentSpec,
) -> Tuple[Sequence[PreTrainedModel], Sequence[PreTrainedTokenizerBase]]:
    """
    从 ExperimentSpec.metadata 中读取模型构建参数，批量构建所有 agent。
    """
    model_name = str(spec.metadata.get("model_name", DEFAULT_MODEL_NAME))
    reset_mode = str(spec.metadata.get("reset_mode", "base"))

    quantization_config = spec.metadata.get("quantization_config")
    if quantization_config is None:
        quantization_config = build_quantization_config()

    lora_config = spec.metadata.get("lora_config")
    if lora_config is None:
        lora_config = build_lora_config()

    device_map = spec.metadata.get("device_map", DEFAULT_DEVICE_MAP)
    torch_dtype = spec.metadata.get("torch_dtype", DEFAULT_TORCH_DTYPE)
    trust_remote_code = bool(spec.metadata.get("trust_remote_code", True))
    prepare_kbit_training = bool(spec.metadata.get("prepare_kbit_training", True))

    num_agents = int(spec.config.num_agents)

    raw_adapter_dirs = spec.metadata.get("agent_adapter_dirs")
    if raw_adapter_dirs is None:
        adapter_dirs: List[str | Path | None] = [None] * num_agents
    else:
        adapter_dirs = list(raw_adapter_dirs)

    if len(adapter_dirs) < num_agents:
        adapter_dirs = adapter_dirs + [None] * (num_agents - len(adapter_dirs))
    elif len(adapter_dirs) > num_agents:
        raise ValueError(
            f"Length of `agent_adapter_dirs` ({len(adapter_dirs)}) must not exceed "
            f"`config.num_agents` ({num_agents})."
        )

    agents: List[PreTrainedModel] = []
    tokenizers: List[PreTrainedTokenizerBase] = []

    for adapter_dir in adapter_dirs:
        model, tokenizer = build_lora_model(
            model_name,
            reset_mode=reset_mode,
            adapter_dir=adapter_dir,
            quantization_config=quantization_config,
            lora_config=lora_config,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            prepare_kbit_training=prepare_kbit_training,
        )
        agents.append(model)
        tokenizers.append(tokenizer)

    return agents, tokenizers


def build_models(
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    num_agents: int = 2,
    reset_mode: str = "base",
    adapter_dirs: Optional[Sequence[str | Path | None]] = None,
    quantization_config: Optional[BitsAndBytesConfig] = None,
    lora_config: Optional[LoraConfig] = None,
    device_map: str | Dict[str, Any] = DEFAULT_DEVICE_MAP,
    torch_dtype: torch.dtype = DEFAULT_TORCH_DTYPE,
    trust_remote_code: bool = True,
    prepare_kbit_training: bool = True,
) -> Tuple[Sequence[PreTrainedModel], Sequence[PreTrainedTokenizerBase]]:
    """
    不依赖 ExperimentSpec 的直接构建接口。

    适合：
    - 单元测试
    - 临时脚本
    - 后续 framework 内部重构时复用
    """
    resolved_adapter_dirs = list(adapter_dirs) if adapter_dirs is not None else [None] * int(num_agents)
    if len(resolved_adapter_dirs) != int(num_agents):
        raise ValueError(
            f"Length of `adapter_dirs` ({len(resolved_adapter_dirs)}) must equal `num_agents` ({num_agents})."
        )

    agents: List[PreTrainedModel] = []
    tokenizers: List[PreTrainedTokenizerBase] = []

    for adapter_dir in resolved_adapter_dirs:
        model, tokenizer = build_lora_model(
            model_name,
            reset_mode=reset_mode,
            adapter_dir=adapter_dir,
            quantization_config=quantization_config or build_quantization_config(),
            lora_config=lora_config or build_lora_config(),
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            prepare_kbit_training=prepare_kbit_training,
        )
        agents.append(model)
        tokenizers.append(tokenizer)

    return agents, tokenizers