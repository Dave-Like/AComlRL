from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


BranchPrompts = List[List[str]]
BranchHistories = List[List[List[str]]]
BranchActions = List[List[str]]
RewardVector = Sequence[float]


@dataclass(slots=True)
class FlatBranchSample:
    """
    从 group `NodeSample` 中展平出的单条 branch 样本。

    该结构供 engine 侧直接消费，避免算法实现反复手动拆解
    `branch × agent` 维度。
    """

    node_id: str = ""
    parent_id: Optional[str] = None
    episode_id: str = ""
    turn_idx: int = 0
    env_step: int = 0
    depth: int = 0
    branch_idx: int = 0
    agent_prompts: List[str] = field(default_factory=list)
    prompt_history_per_agent: List[List[str]] = field(default_factory=list)
    response_history_per_agent: List[List[str]] = field(default_factory=list)
    actions_per_agent: List[str] = field(default_factory=list)
    reward: float = 0.0
    return_: float = 0.0
    next_prompts_per_agent: List[str] = field(default_factory=list)
    terminal: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EngineTrainSample:
    """
    engine 层消费的中间训练样本。

    它以 `FlatBranchSample` 为基础，补充 group-aware 算法常用的统计量，
    例如组均值、组标准差、中心化 return、归一化 advantage 等。
    """

    agent_idx: int = 0
    node_id: str = ""
    episode_id: str = ""
    turn_idx: int = 0
    env_step: int = 0
    depth: int = 0
    branch_idx: int = 0
    action_text: str = ""
    agent_prompt: str = ""
    agent_prompt_history: List[str] = field(default_factory=list)
    agent_response_history: List[str] = field(default_factory=list)
    joint_actions: List[str] = field(default_factory=list)
    reward: float = 0.0
    return_: float = 0.0
    group_mean_return: float = 0.0
    group_std_return: float = 0.0
    centered_return: float = 0.0
    normalized_advantage: float = 0.0
    logprob: Optional[float] = None
    old_logprob: Optional[float] = None
    ref_logprob: Optional[float] = None
    importance_ratio: float = 1.0
    clipped_ratio: float = 1.0
    policy_objective: float = 0.0
    clipped_policy_objective: float = 0.0
    approx_kl: float = 0.0
    terminal: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NodeSample:
    """
    多轮多智能体并行 rollout 在某一轮上的 group 节点样本。

    当前版本中，一个 `NodeSample` 不再表示“单条历史上的多个候选动作”，
    而是表示在同一轮 turn 上并行保留的 G 条分支状态切片：

    - 第 g 条分支有自己的 prompt / history；
    - 每个 agent 基于第 g 条分支历史生成一个动作；
    - 多个 agent 在第 g 条分支上的动作共同组成该分支的联合动作；
    - 环境分别对 G 条分支计算 reward，并推进到各自的下一轮历史。
    """

    turn_idx: int
    node_id: str = ""
    parent_id: Optional[str] = None
    depth: int = 0

    prompts_per_branch_per_agent: BranchPrompts = field(default_factory=list)
    prompt_history_per_branch_per_agent: BranchHistories = field(default_factory=list)
    response_history_per_branch_per_agent: BranchHistories = field(default_factory=list)
    actions_per_branch_per_agent: BranchActions = field(default_factory=list)
    joint_rewards: List[float] = field(default_factory=list)
    joint_returns: List[float] = field(default_factory=list)
    next_prompts_per_branch_per_agent: BranchPrompts = field(default_factory=list)

    terminal: bool = False
    env_step: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_branches(self) -> int:
        return len(self.actions_per_branch_per_agent)

    def flatten_branches(self) -> List[FlatBranchSample]:
        return [
            self.get_branch_sample(branch_idx)
            for branch_idx in range(self.num_branches)
        ]

    def get_branch_sample(self, branch_idx: int) -> FlatBranchSample:
        if branch_idx < 0 or branch_idx >= self.num_branches:
            raise IndexError("branch_idx out of range.")

        episode_id = str(self.metadata.get("episode_id", ""))
        branch_logprobs = self.metadata.get("logprobs_per_branch_per_agent", [])
        branch_logprobs_for_current = (
            list(branch_logprobs[branch_idx])
            if branch_idx < len(branch_logprobs)
            else []
        )
        return FlatBranchSample(
            node_id=self.node_id,
            parent_id=self.parent_id,
            episode_id=episode_id,
            turn_idx=self.turn_idx,
            env_step=self.env_step,
            depth=self.depth,
            branch_idx=branch_idx,
            agent_prompts=list(self.prompts_per_branch_per_agent[branch_idx]),
            prompt_history_per_agent=[
                list(agent_history)
                for agent_history in self.prompt_history_per_branch_per_agent[branch_idx]
            ],
            response_history_per_agent=[
                list(agent_history)
                for agent_history in self.response_history_per_branch_per_agent[branch_idx]
            ],
            actions_per_agent=list(self.actions_per_branch_per_agent[branch_idx]),
            reward=(
                float(self.joint_rewards[branch_idx])
                if branch_idx < len(self.joint_rewards)
                else 0.0
            ),
            return_=(
                float(self.joint_returns[branch_idx])
                if branch_idx < len(self.joint_returns)
                else 0.0
            ),
            next_prompts_per_agent=(
                list(self.next_prompts_per_branch_per_agent[branch_idx])
                if branch_idx < len(self.next_prompts_per_branch_per_agent)
                else []
            ),
            terminal=self.terminal,
            metadata={
                **dict(self.metadata),
                "branch_idx": branch_idx,
                "logprobs_per_agent": branch_logprobs_for_current,
            },
        )


@dataclass(slots=True)
class EnvironmentStepResult:
    """
    环境在某一轮 step/turn 执行后的标准返回值。

    在 group rollout 模式下，环境一次 step 会同步推进 G 条分支，
    因而返回下一轮的 branch×agent prompts 结构。
    """

    turn_idx: int
    node_sample: NodeSample
    next_prompts_per_branch_per_agent: Optional[BranchPrompts] = None
    should_terminate: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RolloutBatch:
    """
    一次 rollout 收集阶段产出的批对象。

    当前版本中，一个 `RolloutBatch` 表示某个样本 / episode 的完整 group
    rollout 结果。`nodes` 仍按 turn 顺序排列，但每个节点内部都保存 G 条
    并行分支的信息。
    """

    sample_id: str = ""
    episode_id: str = ""
    source_item: Optional[Dict[str, Any]] = None
    root_prompt: str = ""
    num_agents: int = 0
    num_branches: int = 0
    num_turns: int = 0
    nodes: List[NodeSample] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def flatten_branch_samples(self) -> List[FlatBranchSample]:
        flattened: List[FlatBranchSample] = []
        for node in self.nodes:
            flattened.extend(node.flatten_branches())
        return flattened


@dataclass(slots=True)
class EpisodeTree:
    """
    对一次完整多轮展开过程的树状视图。

    当前 group rollout 已显式保留 G 条并行分支，但仍保留该结构，
    便于后续扩展为更一般的树搜索/树回传框架。
    """

    episode_id: str
    root_id: str
    nodes_by_id: Dict[str, NodeSample] = field(default_factory=dict)
    children_by_id: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class UpdateBatch:
    """
    供算法引擎消费的更新批对象。

    trainer 仍向 engine 传递 `List[NodeSample]`，但每个样本内部已经是
    branch×agent 的 group rollout 结构。具体算法可自行决定如何沿分支维
    计算 return / advantage / loss。
    """

    agent_idx: int
    algorithm_name: str = ""
    samples: List[NodeSample] = field(default_factory=list)
    discount: float = 0.99
    normalize_advantages: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def flatten_branch_samples(self) -> List[FlatBranchSample]:
        flattened: List[FlatBranchSample] = []
        for sample in self.samples:
            flattened.extend(sample.flatten_branches())
        return flattened


@dataclass(slots=True)
class EngineUpdateResult:
    """算法引擎一次 update 调用后的统一返回协议。"""

    algorithm_name: str = ""
    updated: bool = False
    num_update_batches: int = 0
    num_samples: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvalRecord:
    """单条评测记录。"""

    sample_id: str = ""
    episode_id: str = ""
    agent_idx: int = 0
    turn_idx: int = 0
    prompt: str = ""
    completion: str = ""
    reward: Optional[float] = None
    return_: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluatorOutput:
    """评估器的聚合输出。"""

    records: List[EvalRecord] = field(default_factory=list)
    aggregated_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


RuntimeSample = NodeSample
RuntimeBatch = RolloutBatch
PromptHistory = BranchHistories
ResponseHistory = BranchHistories
