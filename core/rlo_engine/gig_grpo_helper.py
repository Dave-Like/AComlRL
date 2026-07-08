from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from core.common.types import EngineTrainSample
from core.rlo_engine.magrpo_update import MAGRPOPolicyUpdater


@dataclass(slots=True)
class TaskStructureFeatures:
    exists: float = 0.0
    called: float = 0.0
    used: float = 0.0
    ignored: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "exists": float(self.exists),
            "called": float(self.called),
            "used": float(self.used),
            "ignored": float(self.ignored),
        }



class ContributionAnalyzer:
    def __init__(
        self,
        *,
        task_combination: str = "linear",
        task_weights: Dict[str, float] | None = None,
        counterfactual_weights: Dict[str, float] | None = None,
        anchor_coef: float = 0.25,
        no_helper_token: str = "Nohelperutilitycodeavailable",
    ) -> None:
        default_task_weights = {
            "exists": 1.0,
            "called": 1.0,
            "used": 1.0,
            "ignored": 1.0,
            "gate_used": 1.0,
        }
        default_counterfactual_weights = {
            "ablation": 1.0,
            "cross": 1.0,
            "anchor": 1.0,
        }
        self.task_combination = str(task_combination or "linear").strip().lower()
        self.task_weights = {**default_task_weights, **dict(task_weights or {})}
        self.counterfactual_weights = {
            **default_counterfactual_weights,
            **dict(counterfactual_weights or {}),
        }
        self.anchor_coef = float(anchor_coef)
        self.no_helper_token = str(no_helper_token)

    def analyze_task_structure(self, text: str) -> TaskStructureFeatures:
        stripped = str(text or "").strip()
        if not stripped:
            return TaskStructureFeatures(ignored=1.0)

        try:
            tree = ast.parse(stripped)
            parse_ok = True
        except SyntaxError:
            tree = None
            parse_ok = False

        exists = 1.0 if parse_ok and self._has_meaningful_python_structure(tree) else 0.0
        helper_names = self._extract_helper_names(tree) if tree is not None else []
        called = self._estimate_called_ratio(tree, helper_names) if tree is not None else 0.0
        used = self._estimate_used_ratio(tree, helper_names) if tree is not None else 0.0
        ignored = self._estimate_ignored_ratio(text=stripped, tree=tree, parse_ok=parse_ok, helper_names=helper_names)
        return TaskStructureFeatures(
            exists=float(exists),
            called=float(called),
            used=float(used),
            ignored=float(ignored),
        )

    def task_score(self, text: str) -> tuple[float, Dict[str, float]]:
        features = self.analyze_task_structure(text)
        weights = self.task_weights
        if self.task_combination in {"gatedproduct", "gated_product", "gate", "gated"}:
            score = (
                features.exists
                * features.called
                * (1.0 + weights["gate_used"] * features.used)
                * max(0.0, 1.0 - features.ignored)
            )
        else:
            score = (
                weights["exists"] * features.exists
                + weights["called"] * features.called
                + weights["used"] * features.used
                - weights["ignored"] * features.ignored
            )
        return float(score), features.as_dict()

    def counterfactual_score(
        self,
        *,
        sample: EngineTrainSample,
        peer_samples_in_group: Sequence[EngineTrainSample],
    ) -> tuple[float, Dict[str, float]]:
        current_task_score, _ = self.task_score(sample.action_text)
        ablated_text = self._ablate_helper_content(sample.action_text)
        ablated_task_score, _ = self.task_score(ablated_text)
        ablation = current_task_score - ablated_task_score

        peer_task_scores = [self.task_score(peer.action_text)[0] for peer in peer_samples_in_group]
        if peer_task_scores:
            cross = current_task_score - (sum(peer_task_scores) / len(peer_task_scores))
        else:
            cross = 0.0

        anchor = 0.0
        if sample.logprob is not None and sample.ref_logprob is not None:
            anchor = float(sample.logprob - sample.ref_logprob)
        elif sample.approx_kl != 0.0:
            anchor = float(sample.approx_kl)
        anchor *= self.anchor_coef

        weights = self.counterfactual_weights
        total = (
            weights["ablation"] * ablation
            + weights["cross"] * cross
            + weights["anchor"] * anchor
        )
        return float(total), {
            "cf_ablation": float(ablation),
            "cf_cross": float(cross),
            "cf_anchor": float(anchor),
        }

    @staticmethod
    def _has_meaningful_python_structure(tree: ast.AST | None) -> bool:
        if tree is None:
            return False
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom, ast.Call)):
                return True
        return False

    @staticmethod
    def _extract_helper_names(tree: ast.AST | None) -> List[str]:
        if tree is None:
            return []
        names: List[str] = []
        for node in getattr(tree, "body", []):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.append(str(node.name))
        return names

    @staticmethod
    def _estimate_called_ratio(tree: ast.AST | None, helper_names: Sequence[str]) -> float:
        if tree is None or not helper_names:
            return 0.0
        called_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in helper_names:
                    called_names.add(func.id)
                elif isinstance(func, ast.Attribute) and func.attr in helper_names:
                    called_names.add(func.attr)
        return float(len(called_names) / max(len(helper_names), 1))

    @staticmethod
    def _estimate_used_ratio(tree: ast.AST | None, helper_names: Sequence[str]) -> float:
        if tree is None or not helper_names:
            return 0.0
        call_counts = {name: 0 for name in helper_names}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in call_counts:
                    call_counts[func.id] += 1
                elif isinstance(func, ast.Attribute) and func.attr in call_counts:
                    call_counts[func.attr] += 1
        used = sum(1 for count in call_counts.values() if count > 0)
        density_bonus = min(sum(call_counts.values()) / max(len(helper_names), 1), 1.0)
        return float((used / max(len(helper_names), 1) + density_bonus) / 2.0)

    def _estimate_ignored_ratio(
        self,
        *,
        text: str,
        tree: ast.AST | None,
        parse_ok: bool,
        helper_names: Sequence[str],
    ) -> float:
        if not parse_ok:
            return 1.0
        if tree is None:
            return 1.0 if text.strip() else 0.0

        total_lines = max(len([line for line in text.splitlines() if line.strip()]), 1)
        dead_helper_lines = 0
        helper_line_spans = self._collect_helper_line_spans(tree)
        called_ratio = self._estimate_called_ratio(tree, helper_names)
        for helper_name, span in helper_line_spans.items():
            if helper_name not in helper_names:
                continue
            if called_ratio <= 0.0:
                dead_helper_lines += span
        junk_lines = len([line for line in text.splitlines() if self._looks_like_placeholder_or_error(line)])
        return float(min(max((dead_helper_lines + junk_lines) / total_lines, 0.0), 1.0))

    @staticmethod
    def _collect_helper_line_spans(tree: ast.AST) -> Dict[str, int]:
        spans: Dict[str, int] = {}
        for node in getattr(tree, "body", []):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = int(getattr(node, "lineno", 1))
                end = int(getattr(node, "end_lineno", start))
                spans[str(node.name)] = max(end - start + 1, 1)
        return spans

    @staticmethod
    def _looks_like_placeholder_or_error(line: str) -> bool:
        lowered = line.strip().lower()
        if not lowered:
            return False
        patterns = [
            "todo",
            "pass",
            "fixme",
            "syntaxerror",
            "nameerror",
            "nohelperutilitycodeavailable",
            "placeholder",
            "not implemented",
        ]
        return any(token in lowered for token in patterns)

    def _ablate_helper_content(self, text: str) -> str:
        stripped = str(text or "")
        if not stripped:
            return stripped
        try:
            tree = ast.parse(stripped)
        except SyntaxError:
            return self.no_helper_token

        helper_names = self._extract_helper_names(tree)
        if not helper_names:
            collapsed = re.sub(r"\s+", " ", stripped).strip()
            return self.no_helper_token if collapsed else ""

        lines = stripped.splitlines()
        helper_spans = self._collect_helper_line_spans(tree)
        kept_lines: List[str] = []
        for idx, line in enumerate(lines, start=1):
            in_helper = False
            for helper_name in helper_names:
                span = helper_spans.get(helper_name, 0)
                node = next(
                    (
                        n
                        for n in getattr(tree, "body", [])
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == helper_name
                    ),
                    None,
                )
                if node is None:
                    continue
                start = int(getattr(node, "lineno", idx))
                end = int(getattr(node, "end_lineno", start + span - 1))
                if start <= idx <= end:
                    in_helper = True
                    break
            if not in_helper:
                kept_lines.append(line)
        ablated = "\n".join(kept_lines).strip()
        return ablated if ablated else self.no_helper_token


def stable_mean(values: Sequence[float], default: float = 0.0) -> float:
    return float(sum(values) / len(values)) if values else float(default)


def stable_std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean_value = stable_mean(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return float(math.sqrt(max(variance, 0.0)))
