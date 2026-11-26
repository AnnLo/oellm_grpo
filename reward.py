"""VERL custom reward function for the converted datasets.

Use with::

    custom_reward_function.path=src/datatest/reward.py
    custom_reward_function.name=compute_score

Dispatches on ``extra_info["verifier"]``. Verifiers ``ifeval``, ``logic`` and
``workbench`` need external checkers and raise NotImplementedError.
"""

from __future__ import annotations

import json
import re
from typing import Any


def _final_answer_text(solution_str: str) -> str:
    """Drop a reasoning block so answers quoted while thinking are not graded."""
    return solution_str.rsplit("</think>", 1)[-1]


def score_math(solution_str: str, ground_truth: str) -> float:
    from verl.utils.reward_score import math_verify

    return math_verify.compute_score(_final_answer_text(solution_str), ground_truth)


def score_mcqa(solution_str: str, ground_truth: str, kwargs: dict[str, Any]) -> float:
    matches = re.findall(kwargs["output_regex"], _final_answer_text(solution_str))
    return float(bool(matches) and matches[-1].strip().upper() == ground_truth.strip().upper())


def score_code(solution_str: str, ground_truth: str) -> float:
    from verl.utils.reward_score import prime_code

    result = prime_code.compute_score(
        _final_answer_text(solution_str), ground_truth, continuous=True
    )
    score = result[0] if isinstance(result, tuple) else result
    return float(score)


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def score_json_schema(solution_str: str, ground_truth: str) -> float:
    import jsonschema

    text = _final_answer_text(solution_str).strip()
    fenced = _JSON_FENCE.findall(text)
    candidate = fenced[-1] if fenced else text
    try:
        jsonschema.validate(json.loads(candidate), json.loads(ground_truth))
    except (json.JSONDecodeError, jsonschema.ValidationError):
        return 0.0
    return 1.0


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> float:
    if extra_info is None:
        raise ValueError(f"{data_source}: extra_info with a `verifier` key is required")
    verifier = extra_info["verifier"]
    kwargs = json.loads(extra_info.get("verifier_kwargs") or "{}")
    if verifier == "math":
        return score_math(solution_str, ground_truth)
    if verifier == "mcqa":
        return score_mcqa(solution_str, ground_truth, kwargs)
    if verifier == "code":
        return score_code(solution_str, ground_truth)
    if verifier == "json_schema":
        return score_json_schema(solution_str, ground_truth)
    raise NotImplementedError(f"{data_source}: no checker for verifier {verifier!r}")
