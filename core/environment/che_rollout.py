from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.common.types import NodeSample, RolloutBatch
from core.config.config import BaseMultiAgentConfig, GIG_GRPOConfig, MAGRPOConfig
from core.environment.che_handler import CoopHumanEnvHandler


class CHEEpisodeRollout:
    """基于 group step 接口收集保留 G 条并行轨迹的 episode。"""

    def __init__(
        self,
        *,
        handler: CoopHumanEnvHandler,
        config: BaseMultiAgentConfig | None = None,
        sample_id_key: str = "id",
        discount: Optional[float] = None,
    ) -> None:
        resolved_config = config or BaseMultiAgentConfig()
        self.handler = handler
        self.config = resolved_config
        self.sample_id_key = sample_id_key
        self.discount = (
            float(resolved_config.discount) if discount is None else float(discount)
        )

    def rollout_episode(
        self,
        *,
        item: Dict[str, str | object],
        num_generations: Optional[int] = None,
        generation_kwargs: Optional[Dict[str, object]] = None,
        max_turns: Optional[int] = None,
    ) -> RolloutBatch:
        state = self.handler.reset(item)
        num_agents = self.handler.executor.env.num_agents
        branch_count = (
            self.handler.executor.num_generations
            if num_generations is None
            else num_generations
        )
        turn_limit = (
            self.handler.executor.env.num_turns if max_turns is None else max_turns
        )

        prompt_history_per_branch_per_agent: List[List[List[str]]] = [
            [[] for _ in range(num_agents)] for _ in range(branch_count)
        ]
        response_history_per_branch_per_agent: List[List[List[str]]] = [
            [[] for _ in range(num_agents)] for _ in range(branch_count)
        ]
        current_prompts_per_branch_per_agent: Optional[List[List[str]]] = None
        parent_id: Optional[str] = None
        nodes: List[NodeSample] = []

        for turn_idx in range(turn_limit):
            step_result = self.handler.step(
                state=state,
                turn_idx=turn_idx,
                prompt_history_per_branch_per_agent=prompt_history_per_branch_per_agent,
                response_history_per_branch_per_agent=response_history_per_branch_per_agent,
                external_prompts_per_branch_per_agent=current_prompts_per_branch_per_agent,
                num_generations=branch_count,
                generation_kwargs=generation_kwargs,
                parent_id=parent_id,
                depth=turn_idx,
                env_step=turn_idx,
            )
            node = step_result.node_sample
            nodes.append(node)

            if step_result.should_terminate:
                break

            prompt_history_per_branch_per_agent = [
                [list(agent_history) for agent_history in branch_history]
                for branch_history in node.prompt_history_per_branch_per_agent
            ]
            response_history_per_branch_per_agent = self._append_branch_actions(
                node.response_history_per_branch_per_agent,
                node.actions_per_branch_per_agent,
            )
            current_prompts_per_branch_per_agent = (
                step_result.next_prompts_per_branch_per_agent
            )
            parent_id = node.node_id

            if current_prompts_per_branch_per_agent is None:
                break

        self._populate_joint_returns(nodes, discount=self.discount)

        return RolloutBatch(
            sample_id=self._resolve_sample_id(item),
            episode_id=state.episode_id,
            source_item=dict(item),
            root_prompt=self._resolve_root_prompt(item),
            num_agents=num_agents,
            num_branches=branch_count,
            num_turns=len(nodes),
            nodes=nodes,
            metadata={
                "num_branches": branch_count,
                "max_turns": turn_limit,
                "discount": self.discount,
            },
        )

    def _resolve_sample_id(self, item: Dict[str, str | object]) -> str:
        value = item.get(self.sample_id_key, "")
        return "" if value is None else str(value)

    @staticmethod
    def _resolve_root_prompt(item: Dict[str, str | object]) -> str:
        value = item.get("prompt", "")
        return "" if value is None else str(value)

    @staticmethod
    def _append_branch_actions(
        response_history_per_branch_per_agent: List[List[List[str]]],
        actions_per_branch_per_agent: List[List[str]],
    ) -> List[List[List[str]]]:
        return [
            [
                list(agent_history) + [action]
                for agent_history, action in zip(branch_history, branch_actions)
            ]
            for branch_history, branch_actions in zip(
                response_history_per_branch_per_agent,
                actions_per_branch_per_agent,
            )
        ]

    @staticmethod
    def _populate_joint_returns(
        nodes: List[NodeSample],
        discount: float = 1.0,
    ) -> None:
        if not nodes:
            return

        branch_count = max((node.num_branches for node in nodes), default=0)
        running_returns = [0.0 for _ in range(branch_count)]

        for node in reversed(nodes):
            branch_returns: List[float] = []
            for branch_idx in range(node.num_branches):
                reward = (
                    float(node.joint_rewards[branch_idx])
                    if branch_idx < len(node.joint_rewards)
                    else 0.0
                )
                running_returns[branch_idx] = (
                    reward + discount * running_returns[branch_idx]
                )
                branch_returns.append(float(running_returns[branch_idx]))
            node.joint_returns = branch_returns


class MAGRPOEpisodeRollout(CHEEpisodeRollout):
    """面向 MAGRPO 的 rollout 封装。"""

    def __init__(
        self,
        *,
        handler: CoopHumanEnvHandler,
        config: MAGRPOConfig | None = None,
        sample_id_key: str = "id",
        discount: Optional[float] = None,
    ) -> None:
        super().__init__(
            handler=handler,
            config=config or MAGRPOConfig(),
            sample_id_key=sample_id_key,
            discount=discount,
        )


class GIGGRPOEpisodeRollout(CHEEpisodeRollout):
    """面向 GIG-GRPO 的 rollout 封装。"""

    def __init__(
        self,
        *,
        handler: CoopHumanEnvHandler,
        config: GIG_GRPOConfig | None = None,
        sample_id_key: str = "id",
        discount: Optional[float] = None,
    ) -> None:
        super().__init__(
            handler=handler,
            config=config or GIG_GRPOConfig(),
            sample_id_key=sample_id_key,
            discount=discount,
        )
