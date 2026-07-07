from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from core.config.config import BaseMultiAgentConfig


@dataclass(slots=True)
class AgentGenerationOutput:
    """单个 agent 针对单个分支单轮生成结果。"""

    agent_idx: int
    prompt: str
    completions: List[str]
    prompt_input_ids: torch.Tensor
    prompt_attention_mask: torch.Tensor
    completion_token_ids: List[torch.Tensor]
    completion_attention_masks: List[torch.Tensor]
    completion_logprobs: List[float]


@dataclass(slots=True)
class JointActionBatch:
    """多 agent 候选动作的联合视图。"""

    joint_action_indices: List[tuple[int, ...]]
    completions_per_joint_action: List[List[str]]
    metadata: Dict[str, Any]


class LLMGenerationEngine:
    """
    环境模块中的第一层：物理执行层。

    当前版本新增 group rollout 支持：
    - 单分支模式下仍可生成多个候选 completion；
    - group 模式下对每个 branch 的 prompt 一一对应生成一个 completion。
    """

    def __init__(
        self,
        *,
        agents: Sequence[PreTrainedModel],
        tokenizers: Sequence[PreTrainedTokenizerBase] | PreTrainedTokenizerBase,
        config: BaseMultiAgentConfig | None = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        do_sample: Optional[bool] = None,
    ) -> None:
        if not agents:
            raise ValueError("agents must not be empty.")

        resolved_config = config or BaseMultiAgentConfig()
        config_overrides = {
            key: value
            for key, value in {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
            }.items()
            if value is not None
        }
        if config_overrides:
            resolved_config = BaseMultiAgentConfig(
                **{**asdict(resolved_config), **config_overrides}
            )

        self.agents = list(agents)
        self.num_agents = len(self.agents)
        self.tokenizers = self._normalize_tokenizers(tokenizers)
        self.config = resolved_config
        self.temperature = float(self.config.temperature)
        self.top_p = float(self.config.top_p)
        self.top_k = self.config.top_k
        self.max_new_tokens = int(self.config.max_new_tokens)
        self.do_sample = bool(self.config.do_sample)

    def generate_for_agent(
        self,
        *,
        agent_idx: int,
        prompt: str,
        num_generations: int,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> AgentGenerationOutput:
        """为单个 agent 生成候选动作。"""
        if agent_idx < 0 or agent_idx >= self.num_agents:
            raise IndexError("agent_idx out of range.")
        if num_generations < 1:
            raise ValueError("num_generations must be >= 1.")
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string.")

        model = self.agents[agent_idx]
        tokenizer = self.tokenizers[agent_idx]
        device = next(model.parameters()).device

        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        )
        prompt_input_ids = encoded["input_ids"].to(device)
        prompt_attention_mask = encoded["attention_mask"].to(device)
        prompt_len = prompt_input_ids.size(1)

        kwargs: Dict[str, Any] = {
            "input_ids": prompt_input_ids,
            "attention_mask": prompt_attention_mask,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_return_sequences": num_generations,
            "num_beams": 1,
        }
        if self.top_k is not None:
            kwargs["top_k"] = self.top_k
        if generation_kwargs:
            kwargs.update(generation_kwargs)

        was_training = model.training
        model.eval()
        with torch.no_grad():
            generated = model.generate(**kwargs)
            completion_logprobs = self._compute_completion_logprobs(
                model=model,
                sequences=generated,
                prompt_len=prompt_len,
                pad_token_id=(
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                ),
            )
        model.train(was_training)

        completions: List[str] = []
        completion_token_ids: List[torch.Tensor] = []
        completion_attention_masks: List[torch.Tensor] = []

        for seq in generated:
            completion_ids = seq[prompt_len:]
            completion_token_ids.append(completion_ids.detach().cpu())
            completion_attention_masks.append(
                torch.ones(len(completion_ids), dtype=torch.long)
            )
            completions.append(
                tokenizer.decode(completion_ids, skip_special_tokens=True)
            )

        return AgentGenerationOutput(
            agent_idx=agent_idx,
            prompt=prompt,
            completions=completions,
            prompt_input_ids=prompt_input_ids.detach().cpu(),
            prompt_attention_mask=prompt_attention_mask.detach().cpu(),
            completion_token_ids=completion_token_ids,
            completion_attention_masks=completion_attention_masks,
            completion_logprobs=completion_logprobs,
        )

    def generate_for_all_agents(
        self,
        *,
        prompts_per_agent: Sequence[str],
        num_generations: int,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[AgentGenerationOutput]:
        """为所有 agent 在当前轮同步生成候选动作。"""
        if len(prompts_per_agent) != self.num_agents:
            raise ValueError(
                "prompts_per_agent length must match number of agents."
            )

        outputs: List[AgentGenerationOutput] = []
        for agent_idx, prompt in enumerate(prompts_per_agent):
            outputs.append(
                self.generate_for_agent(
                    agent_idx=agent_idx,
                    prompt=prompt,
                    num_generations=num_generations,
                    generation_kwargs=generation_kwargs,
                )
            )
        return outputs

    def generate_group_for_all_agents(
        self,
        *,
        prompts_per_branch_per_agent: Sequence[Sequence[str]],
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[List[AgentGenerationOutput]]:
        """
        对 G 条 branch 同步生成。

        返回结构为 `[branch_idx][agent_idx] -> AgentGenerationOutput`，
        且每个输出恰有 1 个 completion，与 branch 一一对应。
        """
        outputs: List[List[AgentGenerationOutput]] = []
        for prompts_per_agent in prompts_per_branch_per_agent:
            if len(prompts_per_agent) != self.num_agents:
                raise ValueError(
                    "Each branch must provide exactly one prompt per agent."
                )
            outputs.append(
                self.generate_for_all_agents(
                    prompts_per_agent=prompts_per_agent,
                    num_generations=1,
                    generation_kwargs=generation_kwargs,
                )
            )
        return outputs

    def build_joint_actions(
        self,
        *,
        agent_outputs: Sequence[AgentGenerationOutput],
        joint_mode: str = "aligned",
    ) -> JointActionBatch:
        """
        将每个 agent 的候选动作组织成联合动作空间。

        - `aligned`：第 k 个 completion 在所有 agent 间按索引对齐。
        - `cross`：构造所有 agent completion 的笛卡尔积。
        """
        if len(agent_outputs) != self.num_agents:
            raise ValueError("agent_outputs length must match number of agents.")

        mode = str(joint_mode or self.config.joint_mode).strip().lower()
        completions_per_agent = [output.completions for output in agent_outputs]
        joint_action_indices = self._build_joint_action_indices(
            completions_per_agent=completions_per_agent,
            joint_mode=mode,
        )
        completions_per_joint_action = [
            [
                completions_per_agent[agent_idx][joint_idx[agent_idx]]
                for agent_idx in range(self.num_agents)
            ]
            for joint_idx in joint_action_indices
        ]

        return JointActionBatch(
            joint_action_indices=joint_action_indices,
            completions_per_joint_action=completions_per_joint_action,
            metadata={
                "joint_mode": mode,
                "num_agents": self.num_agents,
                "num_joint_actions": len(joint_action_indices),
            },
        )

    def pack_generation_payload(
        self,
        *,
        agent_outputs: Sequence[AgentGenerationOutput],
        joint_mode: str = "aligned",
    ) -> Dict[str, Any]:
        """为上层 handler / env 提供统一生成载荷。"""
        joint_batch = self.build_joint_actions(
            agent_outputs=agent_outputs,
            joint_mode=joint_mode,
        )
        return {
            "agent_outputs": list(agent_outputs),
            "joint_action_indices": joint_batch.joint_action_indices,
            "completions_per_joint_action": joint_batch.completions_per_joint_action,
            "metadata": joint_batch.metadata,
        }

    @staticmethod
    def _compute_completion_logprobs(
        *,
        model: PreTrainedModel,
        sequences: torch.Tensor,
        prompt_len: int,
        pad_token_id: int | None,
    ) -> List[float]:
        if sequences.ndim != 2:
            raise ValueError("sequences must be a rank-2 tensor.")

        attention_mask = torch.ones_like(sequences, dtype=torch.long)
        if pad_token_id is not None:
            attention_mask = (sequences != pad_token_id).long()

        logits = model(input_ids=sequences, attention_mask=attention_mask).logits
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        target_ids = sequences[:, 1:]
        token_log_probs = log_probs.gather(
            dim=-1,
            index=target_ids.unsqueeze(-1),
        ).squeeze(-1)

        completion_start = max(prompt_len - 1, 0)
        completion_mask = torch.zeros_like(token_log_probs, dtype=torch.bool)
        if completion_start < token_log_probs.size(1):
            completion_mask[:, completion_start:] = True
        if pad_token_id is not None:
            completion_mask &= target_ids != pad_token_id

        summed_log_probs = (token_log_probs * completion_mask).sum(dim=1)
        return [float(value) for value in summed_log_probs.detach().cpu().tolist()]

    def _build_joint_action_indices(
        self,
        *,
        completions_per_agent: Sequence[Sequence[str]],
        joint_mode: str,
    ) -> List[tuple[int, ...]]:
        if self.num_agents == 1:
            return [(idx,) for idx in range(len(completions_per_agent[0]))]

        if joint_mode in {"aligned", "align"}:
            generation_count = len(completions_per_agent[0])
            if any(len(values) != generation_count for values in completions_per_agent):
                raise ValueError(
                    "Aligned mode requires equal number of completions for all agents."
                )
            return [tuple([idx] * self.num_agents) for idx in range(generation_count)]

        if joint_mode in {"cross", "crossed"}:
            return [
                tuple(index_tuple)
                for index_tuple in itertools.product(
                    *[range(len(values)) for values in completions_per_agent]
                )
            ]

        raise ValueError(
            "joint_mode must be one of: aligned, align, cross, crossed."
        )

    def _normalize_tokenizers(
        self,
        tokenizers: Sequence[PreTrainedTokenizerBase] | PreTrainedTokenizerBase,
    ) -> List[PreTrainedTokenizerBase]:
        if isinstance(tokenizers, PreTrainedTokenizerBase):
            return [tokenizers for _ in range(self.num_agents)]

        tokenizer_list = list(tokenizers)
        if len(tokenizer_list) != self.num_agents:
            raise ValueError("tokenizers length must match agents length.")
        return tokenizer_list
