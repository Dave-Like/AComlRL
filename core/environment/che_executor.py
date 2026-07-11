from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from core.common.types import BranchHistories, BranchPrompts
from core.config.config import BaseMultiAgentConfig
from core.environment.coop_human_env import CoopHumanEnv, CoopHumanEnvState
from core.environment.generation_engine import AgentGenerationOutput, LLMGenerationEngine


@dataclass(slots=True)
class CHERuntimeTurnResult:
    """CHE group rollout 单轮执行后的运行时结果。"""

    turn_idx: int
    prompts_per_branch_per_agent: List[List[str]]
    prompt_history_per_branch_per_agent: BranchHistories
    response_history_per_branch_per_agent: BranchHistories
    branch_outputs: List[List[AgentGenerationOutput]]
    actions_per_branch_per_agent: List[List[str]]
    joint_rewards: List[float]
    next_prompts_per_branch_per_agent: List[List[str]]
    should_terminate: bool
    state: CoopHumanEnvState
    metadata: Dict[str, Any] = field(default_factory=dict)


class CHEExecutor:
    """负责同步执行 CHE 的 G 条并行分支。"""

    def __init__(
        self,
        *,
        env: CoopHumanEnv,
        generation_engine: LLMGenerationEngine,
        config: BaseMultiAgentConfig | None = None,
        num_generations: Optional[int] = None,
        joint_mode: Optional[str] = None,
        early_termination_threshold: Optional[float] = None,
    ) -> None:
        if generation_engine.num_agents != env.num_agents:
            raise ValueError("generation_engine.num_agents must match env.num_agents.")

        resolved_config = config or BaseMultiAgentConfig()
        resolved_num_generations = (
            resolved_config.num_generations
            if num_generations is None
            else int(num_generations)
        )
        resolved_joint_mode = (
            resolved_config.joint_mode if joint_mode is None else joint_mode
        )
        resolved_termination_threshold = (
            resolved_config.early_termination_threshold
            if early_termination_threshold is None
            else early_termination_threshold
        )

        if resolved_num_generations < 1:
            raise ValueError("num_generations must be >= 1.")

        self.env = env
        self.generation_engine = generation_engine
        self.config = resolved_config
        self.num_generations = resolved_num_generations
        self.joint_mode = str(resolved_joint_mode or "aligned").strip().lower()
        self.early_termination_threshold = resolved_termination_threshold

    def reset(self, item: Dict[str, Any]) -> CoopHumanEnvState:
        return self.env.reset(item)

    def run_turn(
        self,
        *,
        state: CoopHumanEnvState,
        turn_idx: int,
        prompt_history_per_branch_per_agent: BranchHistories,
        response_history_per_branch_per_agent: BranchHistories,
        external_prompts_per_branch_per_agent: Optional[BranchPrompts] = None,
        num_generations: Optional[int] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> CHERuntimeTurnResult:
        branch_count = (
            self.num_generations if num_generations is None else num_generations
        )

        prompts_per_branch_per_agent = self._build_branch_prompts(
            state=state,
            branch_count=branch_count,
            external_prompts_per_branch_per_agent=external_prompts_per_branch_per_agent,
        )
        prompt_history_with_current = self._append_histories(
            prompt_history_per_branch_per_agent,
            prompts_per_branch_per_agent,
        )

        branch_outputs = self._generate_branch_outputs(
            prompts_per_branch_per_agent=prompts_per_branch_per_agent,
            generation_kwargs=generation_kwargs,
        )
        actions_per_branch_per_agent = self._extract_actions_from_branch_outputs(
            branch_outputs
        )

        reward_mode, joint_rewards = self._compute_joint_rewards(
            state=state,
            actions_per_branch_per_agent=actions_per_branch_per_agent,
        )
        should_terminate = self.env.should_terminate(
            turn_idx=turn_idx,
            rewards=joint_rewards,
            early_termination_threshold=self.early_termination_threshold,
        )

        transition_mode, next_prompts_per_branch_per_agent = self._build_next_prompts(
            state=state,
            should_terminate=should_terminate,
            actions_per_branch_per_agent=actions_per_branch_per_agent,
            prompt_history_per_branch_per_agent=prompt_history_with_current,
            response_history_per_branch_per_agent=response_history_per_branch_per_agent,
        )

        return CHERuntimeTurnResult(
            turn_idx=turn_idx,
            prompts_per_branch_per_agent=prompts_per_branch_per_agent,
            prompt_history_per_branch_per_agent=prompt_history_with_current,
            response_history_per_branch_per_agent=[
                [list(agent_history) for agent_history in branch_history]
                for branch_history in response_history_per_branch_per_agent
            ],
            branch_outputs=branch_outputs,
            actions_per_branch_per_agent=actions_per_branch_per_agent,
            joint_rewards=[float(value) for value in joint_rewards],
            next_prompts_per_branch_per_agent=next_prompts_per_branch_per_agent,
            should_terminate=should_terminate,
            state=state,
            metadata={
                "joint_mode": self.joint_mode,
                "num_branches": len(actions_per_branch_per_agent),
                "execution_path": "batched_generation",
                "reward_mode": reward_mode,
                "transition_mode": transition_mode,
            },
        )

    def _generate_branch_outputs(
        self,
        *,
        prompts_per_branch_per_agent: BranchPrompts,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[List[AgentGenerationOutput]]:
        return self.generation_engine.generate_group_for_all_agents(
            prompts_per_branch_per_agent=prompts_per_branch_per_agent,
            generation_kwargs=generation_kwargs,
        )

    @staticmethod
    def _extract_actions_from_branch_outputs(
        branch_outputs: Sequence[Sequence[AgentGenerationOutput]],
    ) -> List[List[str]]:
        return [
            [agent_output.completions[0] for agent_output in branch_outputs_per_agent]
            for branch_outputs_per_agent in branch_outputs
        ]

    def _compute_joint_rewards(
        self,
        *,
        state: CoopHumanEnvState,
        actions_per_branch_per_agent: Sequence[Sequence[str]],
    ) -> tuple[str, List[float]]:
        reward_batch_fn = getattr(self.env, "compute_reward_batch", None)
        if callable(reward_batch_fn):
            rewards = reward_batch_fn(
                state=state,
                actions_per_branch_per_agent=[
                    list(branch_actions)
                    for branch_actions in actions_per_branch_per_agent
                ],
            )
            return "batch", [float(value) for value in rewards]

        rewards = [
            float(
                self.env.compute_reward(
                    state=state,
                    agent_completions=list(branch_actions),
                )
            )
            for branch_actions in actions_per_branch_per_agent
        ]
        return "serial_fallback", rewards

    def _build_next_prompts(
        self,
        *,
        state: CoopHumanEnvState,
        should_terminate: bool,
        actions_per_branch_per_agent: Sequence[Sequence[str]],
        prompt_history_per_branch_per_agent: BranchHistories,
        response_history_per_branch_per_agent: BranchHistories,
    ) -> tuple[str, List[List[str]]]:
        if should_terminate:
            return "skipped_terminal", []

        transition_batch_fn = getattr(self.env, "transition_batch", None)
        if callable(transition_batch_fn):
            next_prompts = transition_batch_fn(
                state=state,
                actions_per_branch_per_agent=[
                    list(branch_actions)
                    for branch_actions in actions_per_branch_per_agent
                ],
                prompt_history_per_branch_per_agent=[
                    [
                        list(agent_history)
                        for agent_history in branch_prompt_history
                    ]
                    for branch_prompt_history in prompt_history_per_branch_per_agent
                ],
                response_history_per_branch_per_agent=[
                    self._append_single_history(branch_response_history, branch_actions)
                    for branch_response_history, branch_actions in zip(
                        response_history_per_branch_per_agent,
                        actions_per_branch_per_agent,
                    )
                ],
            )
            return (
                "batch",
                [list(branch_prompts) for branch_prompts in next_prompts],
            )

        next_prompts_per_branch_per_agent: List[List[str]] = []
        for branch_idx, branch_actions in enumerate(actions_per_branch_per_agent):
            next_prompts_per_branch_per_agent.append(
                self.env.transition(
                    state=state,
                    agent_completions=list(branch_actions),
                    prompt_history_per_agent=prompt_history_per_branch_per_agent[
                        branch_idx
                    ],
                    response_history_per_agent=self._append_single_history(
                        response_history_per_branch_per_agent[branch_idx],
                        branch_actions,
                    ),
                )
            )
        return "serial_fallback", next_prompts_per_branch_per_agent

    def _build_branch_prompts(
        self,
        *,
        state: CoopHumanEnvState,
        branch_count: int,
        external_prompts_per_branch_per_agent: Optional[BranchPrompts],
    ) -> List[List[str]]:
        if external_prompts_per_branch_per_agent is not None:
            prompts = [
                list(branch_prompts)
                for branch_prompts in external_prompts_per_branch_per_agent
            ]
        else:
            base_prompts = self.env.build_prompts(state=state, external_prompts=None)
            prompts = [list(base_prompts) for _ in range(branch_count)]
        return prompts

    @staticmethod
    def _append_single_history(
        history_per_agent: Sequence[Sequence[str]],
        values_per_agent: Sequence[str],
    ) -> List[List[str]]:
        return [
            list(agent_history) + [value]
            for agent_history, value in zip(history_per_agent, values_per_agent)
        ]

    def _append_histories(
        self,
        histories_per_branch_per_agent: BranchHistories,
        values_per_branch_per_agent: BranchPrompts,
    ) -> List[List[List[str]]]:
        return [
            self._append_single_history(branch_history, branch_values)
            for branch_history, branch_values in zip(
                histories_per_branch_per_agent,
                values_per_branch_per_agent,
            )
        ]