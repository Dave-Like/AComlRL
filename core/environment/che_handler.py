from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.common.types import EnvironmentStepResult, NodeSample
from core.environment.che_executor import CHEExecutor, CHERuntimeTurnResult
from core.environment.coop_human_env import CoopHumanEnvState


class CoopHumanEnvHandler:
    """将 CHE group rollout 单轮执行结果适配到统一环境协议。"""

    def __init__(
        self,
        *,
        executor: CHEExecutor,
        branch_selection: str = "max_reward",
    ) -> None:
        self.executor = executor
        self.branch_selection = str(branch_selection or "max_reward").strip().lower()

    def reset(self, item: Dict[str, Any]) -> CoopHumanEnvState:
        return self.executor.reset(item)

    def step(
        self,
        *,
        state: CoopHumanEnvState,
        turn_idx: int,
        prompt_history_per_branch_per_agent: List[List[List[str]]],
        response_history_per_branch_per_agent: List[List[List[str]]],
        external_prompts_per_branch_per_agent: Optional[List[List[str]]] = None,
        num_generations: Optional[int] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
        depth: Optional[int] = None,
        env_step: Optional[int] = None,
    ) -> EnvironmentStepResult:
        runtime_result = self.executor.run_turn(
            state=state,
            turn_idx=turn_idx,
            prompt_history_per_branch_per_agent=prompt_history_per_branch_per_agent,
            response_history_per_branch_per_agent=response_history_per_branch_per_agent,
            external_prompts_per_branch_per_agent=external_prompts_per_branch_per_agent,
            num_generations=num_generations,
            generation_kwargs=generation_kwargs,
        )
        node_sample = self._build_node_sample(
            runtime_result=runtime_result,
            parent_id=parent_id,
            depth=turn_idx if depth is None else depth,
            env_step=turn_idx if env_step is None else env_step,
        )

        return EnvironmentStepResult(
            turn_idx=turn_idx,
            node_sample=node_sample,
            next_prompts_per_branch_per_agent=(
                None
                if runtime_result.should_terminate
                else [
                    list(branch_prompts)
                    for branch_prompts in runtime_result.next_prompts_per_branch_per_agent
                ]
            ),
            should_terminate=runtime_result.should_terminate,
            metadata={
                "joint_mode": runtime_result.metadata.get("joint_mode"),
                "num_branches": runtime_result.metadata.get("num_branches", 0),
            },
        )

    def _build_node_sample(
        self,
        *,
        runtime_result: CHERuntimeTurnResult,
        parent_id: Optional[str],
        depth: int,
        env_step: int,
    ) -> NodeSample:
        state = runtime_result.state
        turn_idx = runtime_result.turn_idx
        logprobs_per_branch_per_agent = [
            [
                float(agent_output.completion_logprobs[0])
                if agent_output.completion_logprobs
                else None
                for agent_output in branch_outputs_per_agent
            ]
            for branch_outputs_per_agent in runtime_result.branch_outputs
        ]
        return NodeSample(
            turn_idx=turn_idx,
            node_id=self._make_node_id(state.episode_id, turn_idx, env_step),
            parent_id=parent_id,
            depth=depth,
            prompts_per_branch_per_agent=[
                list(branch_prompts)
                for branch_prompts in runtime_result.prompts_per_branch_per_agent
            ],
            prompt_history_per_branch_per_agent=[
                [list(agent_history) for agent_history in branch_history]
                for branch_history in runtime_result.prompt_history_per_branch_per_agent
            ],
            response_history_per_branch_per_agent=[
                [list(agent_history) for agent_history in branch_history]
                for branch_history in runtime_result.response_history_per_branch_per_agent
            ],
            actions_per_branch_per_agent=[
                list(branch_actions)
                for branch_actions in runtime_result.actions_per_branch_per_agent
            ],
            joint_rewards=list(runtime_result.joint_rewards),
            joint_returns=[],
            next_prompts_per_branch_per_agent=[
                list(branch_prompts)
                for branch_prompts in runtime_result.next_prompts_per_branch_per_agent
            ],
            terminal=runtime_result.should_terminate,
            env_step=env_step,
            metadata={
                "episode_id": state.episode_id,
                "joint_mode": runtime_result.metadata.get("joint_mode"),
                "num_branches": runtime_result.metadata.get("num_branches", 0),
                "logprobs_per_branch_per_agent": logprobs_per_branch_per_agent,
            },
        )

    @staticmethod
    def _make_node_id(episode_id: str, turn_idx: int, env_step: int) -> str:
        return f"{episode_id}:turn{turn_idx}:step{env_step}"
