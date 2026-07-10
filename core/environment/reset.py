from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


class ResetMode(str, Enum):
    """实验启动时的模型状态模式。"""

    BASE = "base"
    RESUME = "resume"
    EVAL = "eval"


@dataclass(slots=True)
class ResetConfig:
    """
    reset / resume / eval 的统一配置。

    - BASE: 从基础模型重新初始化新的可训练 LoRA
    - RESUME: 从已有 adapter/checkpoint 继续训练
    - EVAL: 从已有 adapter/checkpoint 加载，仅用于评估
    """

    mode: ResetMode = ResetMode.BASE
    resume_adapter_dir: Optional[str] = None
    eval_adapter_dir: Optional[str] = None
    strict_path_check: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ResetDecision:
    """
    经过校验后的加载决策结果。
    """

    mode: ResetMode
    load_base_model: bool
    attach_new_lora: bool
    load_adapter_dir: Optional[Path]
    trainable: bool
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def normalize_reset_mode(mode: str | ResetMode | None) -> ResetMode:
    if isinstance(mode, ResetMode):
        return mode
    raw = str(mode or ResetMode.BASE.value).strip().lower()
    try:
        return ResetMode(raw)
    except ValueError as exc:
        supported = ", ".join(value.value for value in ResetMode)
        raise ValueError(
            f"Unsupported reset mode: {mode!r}. Expected one of: {supported}."
        ) from exc


def build_reset_config(
    *,
    mode: str | ResetMode = ResetMode.BASE,
    resume_adapter_dir: str | None = None,
    eval_adapter_dir: str | None = None,
    strict_path_check: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> ResetConfig:
    return ResetConfig(
        mode=normalize_reset_mode(mode),
        resume_adapter_dir=resume_adapter_dir,
        eval_adapter_dir=eval_adapter_dir,
        strict_path_check=bool(strict_path_check),
        metadata=dict(metadata or {}),
    )


def resolve_reset_decision(config: ResetConfig) -> ResetDecision:
    mode = normalize_reset_mode(config.mode)

    if mode == ResetMode.BASE:
        _validate_base_mode(config)
        return ResetDecision(
            mode=mode,
            load_base_model=True,
            attach_new_lora=True,
            load_adapter_dir=None,
            trainable=True,
            description="Load base model and initialize a fresh trainable LoRA adapter.",
            metadata={"source": "base"},
        )

    if mode == ResetMode.RESUME:
        adapter_dir = _resolve_existing_dir(
            config.resume_adapter_dir,
            strict=config.strict_path_check,
            field_name="resume_adapter_dir",
        )
        return ResetDecision(
            mode=mode,
            load_base_model=True,
            attach_new_lora=False,
            load_adapter_dir=adapter_dir,
            trainable=True,
            description="Load base model and resume training from an existing adapter.",
            metadata={"source": "resume", "adapter_dir": str(adapter_dir)},
        )

    if mode == ResetMode.EVAL:
        adapter_dir = _resolve_existing_dir(
            config.eval_adapter_dir,
            strict=config.strict_path_check,
            field_name="eval_adapter_dir",
        )
        return ResetDecision(
            mode=mode,
            load_base_model=True,
            attach_new_lora=False,
            load_adapter_dir=adapter_dir,
            trainable=False,
            description="Load base model and attach an existing adapter for evaluation only.",
            metadata={"source": "eval", "adapter_dir": str(adapter_dir)},
        )

    raise ValueError(f"Unhandled reset mode: {mode!r}")


def summarize_reset_decision(decision: ResetDecision) -> Dict[str, Any]:
    return {
        "mode": decision.mode.value,
        "load_base_model": bool(decision.load_base_model),
        "attach_new_lora": bool(decision.attach_new_lora),
        "load_adapter_dir": (
            None if decision.load_adapter_dir is None else str(decision.load_adapter_dir)
        ),
        "trainable": bool(decision.trainable),
        "description": decision.description,
        "metadata": dict(decision.metadata),
    }


def _validate_base_mode(config: ResetConfig) -> None:
    if config.resume_adapter_dir:
        raise ValueError(
            "BASE mode should not provide `resume_adapter_dir`. "
            "Use RESUME mode if you want to continue training from an adapter."
        )
    if config.eval_adapter_dir:
        raise ValueError(
            "BASE mode should not provide `eval_adapter_dir`. "
            "Use EVAL mode if you want to load an adapter for evaluation."
        )


def _resolve_existing_dir(
    value: str | None,
    *,
    strict: bool,
    field_name: str,
) -> Path:
    if not value:
        raise ValueError(f"`{field_name}` is required for this reset mode.")

    path = Path(value)
    if strict and not path.exists():
        raise FileNotFoundError(f"{field_name} does not exist: {path}")
    if strict and path.exists() and not path.is_dir():
        raise NotADirectoryError(f"{field_name} must be a directory: {path}")
    return path