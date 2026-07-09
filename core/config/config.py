from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class BaseMultiAgentConfig:
    """多智能体训练与 rollout 的通用超参数。"""

    algorithm_name: str = ""
    num_agents: int = 2
    num_generations: int = 4
    max_turns: int = 4
    batch_size: int = 1
    discount: float = 0.99
    normalize_advantages: bool = True

    temperature: float = 0.6
    top_p: float = 0.6
    top_k: Optional[int] = 50
    max_new_tokens: int = 256
    do_sample: bool = True

    joint_mode: str = "aligned"
    early_termination_threshold: Optional[float] = None


@dataclass(slots=True)
class MAGRPOConfig(BaseMultiAgentConfig):
    """MAGRPO 专属超参数。"""

    algorithm_name: str = "magrpo"
    advantage_mode: str = "zscore"
    advantage_epsilon: float = 1e-8
    clip_range: float = 0.2
    kl_coef: float = 0.0
    learning_rate: float = 1e-5
    update_epochs: int = 1
    max_grad_norm: Optional[float] = 1.0

    max_safe_kl: Optional[float] = 2.0


@dataclass(slots=True)
class GIG_GRPOConfig(BaseMultiAgentConfig):
    """GIG-GRPO 专属超参数。"""

    algorithm_name: str = "gig_grpo"
    advantage_mode: str = "zscore"
    advantage_epsilon: float = 1e-8
    clip_range: float = 0.2
    kl_coef: float = 0.0
    learning_rate: float = 1e-5
    update_epochs: int = 1
    max_grad_norm: Optional[float] = 1.0

    max_safe_kl: Optional[float] = 2.0
    outer_advantage_clip: Optional[float] = 5.0
    inner_advantage_clip: Optional[float] = 3.0
    combined_advantage_clip: Optional[float] = 5.0

    inner_scale_mode: str = "match_outer_mean_abs"
    min_inner_scale: float = 0.5
    max_inner_scale: float = 3.0

    inner_group_size: Optional[int] = None
    outer_group_size: Optional[int] = None
    contribution_mode: str = "hybrid"
    task_combination: str = "linear"
    contribution_lambda: float = 1.0
    contribution_mix_alpha: float = 0.5
    counterfactual_anchor_coef: float = 0.25
    no_helper_token: str = "Nohelperutilitycodeavailable"