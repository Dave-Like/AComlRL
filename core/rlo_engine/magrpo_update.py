from __future__ import annotations

import math
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from torch.optim import Adam, Optimizer
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from core.common.types import EngineTrainSample


class MAGRPOPolicyUpdater:
    """封装 MAGRPO 的真实策略更新逻辑。"""

    def __init__(
        self,
        *,
        policy_models: Sequence[PreTrainedModel] | None = None,
        tokenizers: Sequence[PreTrainedTokenizerBase] | PreTrainedTokenizerBase | None = None,
        learning_rate: float = 1e-5,
        update_epochs: int = 1,
        clip_range: float = 0.2,
        kl_coef: float = 0.0,
        max_grad_norm: float | None = 1.0,
        max_safe_kl: float | None = 2.0,
        optimizers: Sequence[Optimizer] | None = None,
    ) -> None:
        self.policy_models = list(policy_models or [])
        self.learning_rate = float(learning_rate)
        self.update_epochs = int(update_epochs)
        self.clip_range = float(clip_range)
        self.kl_coef = float(kl_coef)
        self.max_grad_norm = max_grad_norm
        self.max_safe_kl = None if max_safe_kl is None else float(max_safe_kl)
        self.tokenizers = self._normalize_tokenizers(tokenizers)
        self.optimizers = self._build_optimizers(optimizers)

    def is_ready(self, train_samples_by_agent: Dict[int, List[EngineTrainSample]]) -> bool:
        return bool(
            train_samples_by_agent
            and self.policy_models
            and self.tokenizers
            and self.optimizers
            and len(self.policy_models) == len(self.tokenizers) == len(self.optimizers)
            and all(
                agent_idx < len(self.policy_models) and samples
                for agent_idx, samples in train_samples_by_agent.items()
            )
        )

    def run(self, train_samples_by_agent: Dict[int, List[EngineTrainSample]]) -> Dict[str, float]:
        losses: List[float] = []
        kls: List[float] = []
        entropies: List[float] = []
        grad_norms: List[float] = []

        invalid_sample_count = 0
        skipped_nonfinite_steps = 0
        skipped_kl_steps = 0
        skipped_grad_steps = 0
        effective_optimizer_steps = 0

        for _ in range(self.update_epochs):
            for agent_idx, samples in train_samples_by_agent.items():
                model = self.policy_models[agent_idx]
                tokenizer = self.tokenizers[agent_idx]
                optimizer = self.optimizers[agent_idx]
                model.train()
                optimizer.zero_grad(set_to_none=True)

                sample_losses: List[torch.Tensor] = []
                sample_kls: List[torch.Tensor] = []
                sample_entropies: List[torch.Tensor] = []

                for sample in samples:
                    if not self._sample_has_finite_inputs(sample):
                        invalid_sample_count += 1
                        sample.metadata["invalid_reason"] = "nonfinite_input"
                        continue

                    result = self._compute_sample_loss(
                        model=model,
                        tokenizer=tokenizer,
                        sample=sample,
                    )
                    if result is None:
                        invalid_sample_count += 1
                        continue

                    loss, approx_kl, entropy, current_logprob = result
                    sample_losses.append(loss)
                    sample_kls.append(approx_kl)
                    sample_entropies.append(entropy)

                    sample.logprob = current_logprob
                    sample.importance_ratio = self._compute_importance_ratio(
                        current_logprob,
                        sample.old_logprob,
                    )
                    sample.clipped_ratio = self._clip_ratio(sample.importance_ratio)
                    sample.policy_objective = (
                        sample.importance_ratio * sample.normalized_advantage
                    )
                    sample.clipped_policy_objective = (
                        sample.clipped_ratio * sample.normalized_advantage
                    )
                    sample.approx_kl = float(approx_kl.detach().cpu().item())

                if not sample_losses:
                    optimizer.zero_grad(set_to_none=True)
                    continue

                loss = torch.stack(sample_losses).mean()
                mean_kl = torch.stack(sample_kls).mean()
                mean_entropy = torch.stack(sample_entropies).mean()

                if not (
                    self._isfinite_tensor(loss)
                    and self._isfinite_tensor(mean_kl)
                    and self._isfinite_tensor(mean_entropy)
                ):
                    skipped_nonfinite_steps += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue

                mean_kl_value = float(mean_kl.detach().cpu().item())
                if self.max_safe_kl is not None and abs(mean_kl_value) > self.max_safe_kl:
                    skipped_kl_steps += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue

                loss.backward()

                if not self._grads_are_finite(model):
                    skipped_grad_steps += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue

                grad_norm = 0.0
                if self.max_grad_norm is not None:
                    grad_tensor = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        float(self.max_grad_norm),
                    )
                    if not self._isfinite_tensor(grad_tensor):
                        skipped_grad_steps += 1
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    grad_norm = float(grad_tensor.detach().cpu().item())

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                effective_optimizer_steps += 1
                losses.append(float(loss.detach().cpu().item()))
                kls.append(mean_kl_value)
                entropies.append(float(mean_entropy.detach().cpu().item()))
                grad_norms.append(grad_norm)

        count = len(losses)
        return {
            "optimizer_steps": float(count),
            "effective_optimizer_steps": float(effective_optimizer_steps),
            "invalid_sample_count": float(invalid_sample_count),
            "skipped_nonfinite_steps": float(skipped_nonfinite_steps),
            "skipped_kl_steps": float(skipped_kl_steps),
            "skipped_grad_steps": float(skipped_grad_steps),
            "mean_policy_loss": sum(losses) / count if count else 0.0,
            "mean_update_approx_kl": sum(kls) / count if count else 0.0,
            "mean_entropy": sum(entropies) / count if count else 0.0,
            "mean_grad_norm": sum(grad_norms) / len(grad_norms) if grad_norms else 0.0,
        }

    def _compute_sample_loss(
        self,
        *,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        sample: EngineTrainSample,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float] | None:
        current_logprob, entropy = self._compute_logprob_and_entropy(
            model=model,
            tokenizer=tokenizer,
            prompt=sample.agent_prompt,
            completion=sample.action_text,
        )
        if not self._isfinite_tensor(current_logprob) or not self._isfinite_tensor(entropy):
            sample.metadata["invalid_reason"] = "nonfinite_logprob_or_entropy"
            return None

        old_logprob = 0.0 if sample.old_logprob is None else float(sample.old_logprob)
        if not math.isfinite(old_logprob):
            sample.metadata["invalid_reason"] = "nonfinite_old_logprob"
            return None

        advantage_value = float(sample.normalized_advantage)
        if not math.isfinite(advantage_value):
            sample.metadata["invalid_reason"] = "nonfinite_advantage"
            return None

        log_ratio = current_logprob - current_logprob.new_tensor(old_logprob)
        log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio)

        advantage = current_logprob.new_tensor(advantage_value)
        unclipped = ratio * advantage
        clipped_ratio = torch.clamp(
            ratio,
            1.0 - self.clip_range,
            1.0 + self.clip_range,
        )
        clipped = clipped_ratio * advantage
        loss = -torch.minimum(unclipped, clipped)

        approx_kl = current_logprob.new_tensor(0.0)
        if sample.ref_logprob is not None:
            ref_logprob = float(sample.ref_logprob)
            if not math.isfinite(ref_logprob):
                sample.metadata["invalid_reason"] = "nonfinite_ref_logprob"
                return None
            approx_kl = current_logprob - current_logprob.new_tensor(ref_logprob)
            loss = loss + self.kl_coef * approx_kl

        if not self._isfinite_tensor(loss) or not self._isfinite_tensor(approx_kl):
            sample.metadata["invalid_reason"] = "nonfinite_loss_or_kl"
            return None

        return loss, approx_kl, entropy, float(current_logprob.detach().cpu().item())

    def _compute_logprob_and_entropy(
        self,
        *,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        prompt: str,
        completion: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        device = next(model.parameters()).device
        prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True)["input_ids"].to(device)
        encoded = tokenizer(f"{prompt}{completion}", return_tensors="pt", truncation=True)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        shifted_logits = logits[:, :-1, :]
        shifted_targets = input_ids[:, 1:]
        log_probs = F.log_softmax(shifted_logits, dim=-1)
        probs = log_probs.exp()
        token_log_probs = log_probs.gather(-1, shifted_targets.unsqueeze(-1)).squeeze(-1)
        token_entropies = -(probs * log_probs).sum(dim=-1)

        completion_mask = torch.zeros_like(token_log_probs, dtype=torch.bool)
        start_idx = max(prompt_ids.size(1) - 1, 0)
        if start_idx < token_log_probs.size(1):
            completion_mask[:, start_idx:] = True
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if pad_token_id is not None:
            completion_mask &= shifted_targets != pad_token_id

        completion_logprob = token_log_probs.masked_select(completion_mask).sum()
        entropy_values = token_entropies.masked_select(completion_mask)
        entropy = (
            completion_logprob.new_tensor(0.0)
            if entropy_values.numel() == 0
            else entropy_values.mean()
        )
        return completion_logprob, entropy

    def _normalize_tokenizers(
        self,
        tokenizers: Sequence[PreTrainedTokenizerBase] | PreTrainedTokenizerBase | None,
    ) -> List[PreTrainedTokenizerBase]:
        if tokenizers is None:
            return []
        if isinstance(tokenizers, PreTrainedTokenizerBase):
            return [tokenizers for _ in range(len(self.policy_models))]
        values = list(tokenizers)
        if self.policy_models and len(values) != len(self.policy_models):
            raise ValueError("tokenizers length must match policy_models length.")
        return values

    def _build_optimizers(
        self,
        optimizers: Sequence[Optimizer] | None,
    ) -> List[Optimizer]:
        if optimizers is not None:
            values = list(optimizers)
            if self.policy_models and len(values) != len(self.policy_models):
                raise ValueError("optimizers length must match policy_models length.")
            return values
        return [Adam(model.parameters(), lr=self.learning_rate) for model in self.policy_models]

    def _compute_importance_ratio(self, logprob: float | None, old_logprob: float | None) -> float:
        if logprob is None or old_logprob is None:
            return 1.0
        if not math.isfinite(logprob) or not math.isfinite(old_logprob):
            return 1.0
        delta = max(min(logprob - old_logprob, 20.0), -20.0)
        return float(torch.exp(torch.tensor(delta)).item())

    def _clip_ratio(self, ratio: float) -> float:
        lower = 1.0 - self.clip_range
        upper = 1.0 + self.clip_range
        return float(min(max(ratio, lower), upper))

    @staticmethod
    def _isfinite_tensor(value: torch.Tensor) -> bool:
        return bool(torch.isfinite(value).all().item())

    @staticmethod
    def _sample_has_finite_inputs(sample: EngineTrainSample) -> bool:
        if not math.isfinite(float(sample.normalized_advantage)):
            return False
        if sample.old_logprob is not None and not math.isfinite(float(sample.old_logprob)):
            return False
        if sample.ref_logprob is not None and not math.isfinite(float(sample.ref_logprob)):
            return False
        return True

    @staticmethod
    def _grads_are_finite(model: PreTrainedModel) -> bool:
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            if not torch.isfinite(parameter.grad).all():
                return False
        return True