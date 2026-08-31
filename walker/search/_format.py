from __future__ import annotations

import re
from typing import Any

from walker.search.env import (
    NODE_SEARCH_CONFIGS,
    _extract_first_valid_search_id,
)

def postprocess_responses(resp: str) -> str:
    
    if "</walk>" in resp:
        return resp.split("</walk>")[0] + "</walk>"
    if "</answer>" in resp:
        return resp.split("</answer>")[0] + "</answer>"
    return resp

def _score_prediction(
    proposed: str,
    label_id: int,
    label_names: list[str],
    task_type: str = "node_class",
) -> bool:
    
    proposed = proposed.strip()
    if not proposed:
        return False
    if "<" in proposed or ">" in proposed:
        return False
    if task_type == "link_pred":
        return proposed == str(label_id)

    if task_type in ("node_class", "relation_class", "graph_reason"):
        if proposed == str(label_id):
            return True
        if 0 <= label_id < len(label_names):
            if proposed.lower() == label_names[label_id].lower():
                return True
        return False

    if proposed == str(label_id):
        return True
    if 0 <= label_id < len(label_names):
        if proposed.lower() == label_names[label_id].lower():
            return True
    return False

def _extract_answer(text: str) -> str | None:
    
    matches = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()

def _rebuild_policy_text(sample: Any, tokenizer: Any) -> str:
    
    tokens = getattr(sample, "tokens", None)
    loss_mask = getattr(sample, "loss_mask", None)
    response_length = getattr(sample, "response_length", None) or 0
    if tokens is None or loss_mask is None or response_length <= 0:
        return sample.response or ""
    response_token_ids = list(tokens[-response_length:])
    loss_mask_list = list(loss_mask)
    if len(loss_mask_list) != len(response_token_ids):
        return sample.response or ""
    policy_ids = [tid for tid, m in zip(response_token_ids, loss_mask_list) if int(m) == 1]
    if not policy_ids:
        return ""
    return tokenizer.decode(policy_ids, skip_special_tokens=False)

_SEQ_TAG_SPLIT = re.compile(r"(</?(?:think(?:ing)?|walk|answer)>)")
_SEQ_TAG_MATCH = re.compile(r"</?(?:think(?:ing)?|walk|answer)>")

def _is_valid_sequence(text: str) -> tuple[bool, str]:

    for _ctl in ("<|im_end|>", "<|end|>", "<|assistant|>", "<|im_start|>"):
        text = text.replace(_ctl, "")

    for label, open_re, close_re in (
        ("thinking", r"<think(?:ing)?>", r"</think(?:ing)?>"),
        ("walk", r"<walk>", r"</walk>"),
        ("answer", r"<answer>", r"</answer>"),
    ):
        if len(re.findall(open_re, text)) != len(re.findall(close_re, text)):
            return False, f"unbalanced {label}"

    state = "start"
    think_tag = None  
    for part in _SEQ_TAG_SPLIT.split(text):
        if not part:
            continue
        if _SEQ_TAG_MATCH.fullmatch(part):
            if part in ("<thinking>", "<think>") and state in ("start", "after_search", "after_answer"):
                state = "in_think"
                think_tag = part
            elif part in ("</thinking>", "</think>") and state == "in_think":
                if part != "</" + think_tag[1:]:
                    return False, f"mismatched {part} (opened {think_tag})"
                state = "after_think"
                think_tag = None
            elif part == "<walk>" and state in ("after_think", "after_search"):
                state = "in_search"
            elif part == "</walk>" and state == "in_search":
                state = "after_search"
            elif part == "<answer>" and state in ("after_think", "after_search"):
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "after_answer"
            else:
                return False, f"unexpected {part} in state {state}"
        else:
            if state in ("in_think", "in_search", "in_answer"):
                continue
            if part.strip():
                return False, f"text between tags in state {state}"

    if state != "after_answer":
        return False, f"ended in state {state}"
    return True, "ok"

def _evaluate_policy_output(policy_text: str, valid_node_ids: set[int]) -> dict:
    
    valid_format, format_reason = _is_valid_sequence(policy_text)
    answer_content = _extract_answer(policy_text)
    has_answer = answer_content is not None
    has_think = bool(re.search(r"<think(?:ing)?>.*?</think(?:ing)?>", policy_text, re.DOTALL))
    search_node_id = _extract_first_valid_search_id(policy_text, valid_node_ids)
    valid_search = search_node_id is not None
    has_search = bool(re.search(r"<walk>.*?</walk>", policy_text, re.DOTALL))
    return {
        "valid_format": valid_format,
        "format_reason": format_reason,
        "has_answer": has_answer,
        "answer_content": answer_content,
        "has_think": has_think,
        "has_search": has_search,
        "valid_search": valid_search,
        "search_node_id": search_node_id,
    }

def _compute_reward_tier(
    correct: bool,
    valid_format: bool,
    valid_search: bool,
    has_answer: bool,
) -> float:
    
    cfg = NODE_SEARCH_CONFIGS
    if correct and valid_format:
        return float(cfg["reward_correct_valid"])
    if correct:
        return float(cfg["reward_correct_invalid"])
    if valid_format and valid_search:
        return float(cfg["reward_valid_search"])
    if valid_format:
        return float(cfg["reward_valid_only"])
    if has_answer:
        return float(cfg["reward_attempt_only"])
    return 0.0
