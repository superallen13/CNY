from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Callable

from walker.search._format import (
    _compute_reward_tier,
    _evaluate_policy_output,
    _score_prediction,
    postprocess_responses,
)
from walker.search.env import GraphEnv, GraphEnvConfig, _normalize_label
from walker.tag.task_registry import task_type_for

ChatFn = Callable[[list[dict]], Awaitable[tuple[str, dict]]]

async def run_episode(
    chat_fn: ChatFn,
    row: dict,
    env_cfg: GraphEnvConfig,
    *,
    data_path: str = "data/raw_datasets",
    loose_format: bool = False,
    summarizer: Any = None,
    answer_style: str = "tag",
) -> dict:
    
    label = _normalize_label(row.get("solution"))
    label_id = int(label.get("ground_truth", -1))
    label_names = list(label.get("label_names", []))
    dataset_name = str(label.get("dataset_name", "unknown"))
    task_type = str(label.get("task_type") or task_type_for(dataset_name))

    messages = list(row["messages"])

    user_prompt_text = ""
    for m in messages:
        if m.get("role") == "user":
            user_prompt_text = str(m.get("content") or "")
            break
    sample = SimpleNamespace(label=row.get("solution"), prompt=user_prompt_text)

    env = GraphEnv(dataset_name=dataset_name, data_path=data_path, cfg=env_cfg,
                   summarizer=summarizer)
    env.reset(sample)

    policy_text = ""
    rejected_reasons: list[str] = []
    truncated = False
    prompt_tokens = 0
    completion_tokens = 0
    cost = 0.0
    has_cost = False  
    for _turn in range(env_cfg.max_turns):
        resp_raw, usage = await chat_fn(messages)
        resp = postprocess_responses(resp_raw)
        step = await env.step(resp)
        policy_text += resp
        messages.append({"role": "assistant", "content": resp})
        if step.rejected_reason:
            rejected_reasons.append(step.rejected_reason)
        if step.obs_text:
            messages.append({"role": "user", "content": step.obs_text})
        prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        if usage.get("cost") is not None:
            cost += float(usage["cost"])
            has_cost = True
        if step.done:
            break
    else:
        truncated = True

    return _build_episode_result(
        env=env, policy_text=policy_text, label_id=label_id,
        label_names=label_names, task_type=task_type, dataset_name=dataset_name,
        truncated=truncated, rejected_reasons=rejected_reasons,
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        cost=cost, has_cost=has_cost, row=row, loose_format=loose_format,
        messages=messages, answer_style=answer_style,
    )

def _evaluate_rethink_output(policy_text: str, valid_node_ids: set[int]) -> dict:

    matches = list(re.finditer(r"[Aa]nswer:\s*([0-9]+)", policy_text))
    answer_content = matches[0].group(1) if matches else None
    has_answer = answer_content is not None
    has_think = bool(re.search(r"<think(?:ing)?>.*?</think(?:ing)?>", policy_text, re.DOTALL))
    return {
        "valid_format": has_answer,
        "format_reason": "rethink_answer" if has_answer else "no_rethink_answer",
        "has_answer": has_answer,
        "answer_content": answer_content,
        "has_think": has_think,
        "has_search": False,
        "valid_search": False,
        "search_node_id": None,
    }

def _build_episode_result(
    *, env, policy_text, label_id, label_names, task_type, dataset_name,
    truncated, rejected_reasons, prompt_tokens, completion_tokens,
    cost, has_cost, row, loose_format, messages, answer_style="tag",
) -> dict:
    
    if answer_style == "rethink":
        eval_fn = _evaluate_rethink_output
    elif loose_format:
        eval_fn = _evaluate_policy_output_loose
    else:
        eval_fn = _evaluate_policy_output
    eval_info = eval_fn(policy_text, env.valid_node_ids)
    correct = (
        eval_info["has_answer"]
        and _score_prediction(eval_info["answer_content"] or "", label_id, label_names, task_type)
    )
    tier = _compute_reward_tier(
        correct=correct,
        valid_format=eval_info["valid_format"],
        valid_search=eval_info["valid_search"],
        has_answer=eval_info["has_answer"],
    )

    searched_node_ids: list[int] = []
    for m in re.finditer(r"<walk>(.*?)</walk>", policy_text, re.DOTALL):
        try:
            searched_node_ids.append(int(m.group(1).strip()))
        except (TypeError, ValueError):
            continue
    view = env.graph_state if hasattr(env, "graph_state") else None
    searched_node_labels: list[int | None] = []
    if view is not None and label_id is not None:
        for nid in searched_node_ids:
            try:
                searched_node_labels.append(view.node_label(nid))
            except Exception:
                searched_node_labels.append(None)
    n_searches_resolved = sum(1 for ll in searched_node_labels if ll is not None)
    n_relevant = sum(1 for ll in searched_node_labels if ll is not None and ll == label_id)
    relevant_search_rate = (n_relevant / n_searches_resolved) if n_searches_resolved else None
    any_relevant_searched = bool(n_relevant > 0)

    node_id = None
    try:
        node_id = int((row.get("metadata") or {}).get("node_id"))
    except (TypeError, ValueError):
        pass

    return {
        "dataset": dataset_name,
        "node_id": node_id,
        "label_id": label_id,
        "correct": bool(correct),
        "reward_tier": float(tier),
        "valid_format": bool(eval_info["valid_format"]),
        "format_reason": str(eval_info["format_reason"]),
        "has_answer": bool(eval_info["has_answer"]),
        "answer_content": eval_info["answer_content"],
        "has_search": bool(eval_info["has_search"]),
        "valid_search": bool(eval_info["valid_search"]),
        "search_node_id": eval_info["search_node_id"],
        "num_turns": int(env.turns_used),
        "num_searches": int(env.successful_search_count),
        
        "searched_node_ids": searched_node_ids,
        "searched_node_labels": searched_node_labels,
        "n_relevant_searches": n_relevant,
        "relevant_search_rate": relevant_search_rate,
        "any_relevant_searched": any_relevant_searched,
        "truncated": bool(truncated),
        "rejected_reasons": rejected_reasons,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cost_usd": cost if has_cost else None,
        "policy_text": policy_text,
        "messages": messages,
    }

async def run_episode_text(
    gen_fn, row: dict, env_cfg: GraphEnvConfig, *, tokenizer,
    data_path: str = "data/raw_datasets", loose_format: bool = False,
    summarizer: Any = None, answer_style: str = "tag",
) -> dict:
    
    label = _normalize_label(row.get("solution"))
    label_id = int(label.get("ground_truth", -1))
    label_names = list(label.get("label_names", []))
    dataset_name = str(label.get("dataset_name", "unknown"))
    task_type = str(label.get("task_type") or task_type_for(dataset_name))

    messages = list(row["messages"])
    user_prompt_text = ""
    for m in messages:
        if m.get("role") == "user":
            user_prompt_text = str(m.get("content") or "")
            break

    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    sample = SimpleNamespace(label=row.get("solution"), prompt=user_prompt_text)

    env = GraphEnv(dataset_name=dataset_name, data_path=data_path, cfg=env_cfg,
                   summarizer=summarizer)
    env.reset(sample)

    response = ""      
    policy_text = ""   
    rejected_reasons: list[str] = []
    truncated = False
    prompt_tokens = 0
    completion_tokens = 0
    cost = 0.0
    has_cost = False
    for _turn in range(env_cfg.max_turns):
        turn_raw, usage = await gen_fn(prompt_text + response)
        turn = postprocess_responses(turn_raw)
        step = await env.step(turn)
        policy_text += turn
        response += turn
        if step.rejected_reason:
            rejected_reasons.append(step.rejected_reason)
        prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        if usage.get("cost") is not None:
            cost += float(usage["cost"])
            has_cost = True
        if step.done:
            break
        if step.obs_text:
            response += step.obs_text  
    else:
        truncated = True

        force_ctx = (
            prompt_text + response
            + "\nYou have used your entire walk budget and cannot walk again. "
            "Give your final answer now as <thinking>...</thinking>"
            "<answer>N</answer>, choosing N from the category list above.\n"
        )
        turn_raw, usage = await gen_fn(force_ctx)
        turn = postprocess_responses(turn_raw)
        step = await env.step(turn)
        policy_text += turn
        prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        if usage.get("cost") is not None:
            cost += float(usage["cost"])
            has_cost = True

    return _build_episode_result(
        env=env, policy_text=policy_text, label_id=label_id,
        label_names=label_names, task_type=task_type, dataset_name=dataset_name,
        truncated=truncated, rejected_reasons=rejected_reasons,
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        cost=cost, has_cost=has_cost, row=row, loose_format=loose_format,
        messages=messages, answer_style=answer_style,
    )

async def run_episode_single_turn(
    chat_fn: ChatFn, row: dict, *, loose_format: bool = False,
    answer_style: str = "tag",
) -> dict:
    
    label = _normalize_label(row.get("solution"))
    label_id = int(label.get("ground_truth", -1))
    label_names = list(label.get("label_names", []))
    dataset_name = str(label.get("dataset_name", "unknown"))
    task_type = str(label.get("task_type") or task_type_for(dataset_name))

    messages = list(row["messages"])
    resp_raw, usage = await chat_fn(messages)
    policy_text = postprocess_responses(resp_raw)
    messages.append({"role": "assistant", "content": policy_text})

    env = SimpleNamespace(
        valid_node_ids=set(), graph_state=None,
        turns_used=1, successful_search_count=0,
    )
    cost = float(usage["cost"]) if usage.get("cost") is not None else 0.0
    return _build_episode_result(
        env=env, policy_text=policy_text, label_id=label_id,
        label_names=label_names, task_type=task_type, dataset_name=dataset_name,
        truncated=False, rejected_reasons=[],
        prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
        completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        cost=cost, has_cost=usage.get("cost") is not None, row=row,
        loose_format=loose_format, messages=messages, answer_style=answer_style,
    )

def _make_generate_fn(base_url: str, *, max_tokens: int, temperature: float,
                      semaphore: asyncio.Semaphore,
                      skip_special_tokens: bool = False,
                      repetition_penalty: float = 1.0):
    
    import httpx

    gen_url = base_url.rstrip("/")
    if gen_url.endswith("/v1"):
        gen_url = gen_url[: -len("/v1")]
    gen_url = gen_url + "/generate"
    stop = ["</walk>", "</answer>"]
    client = httpx.AsyncClient(timeout=httpx.Timeout(600.0))

    async def _gen(full_text: str) -> tuple[str, dict]:
        async with semaphore:
            _sp = {
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "stop": stop,
                "no_stop_trim": True,
                "skip_special_tokens": skip_special_tokens,
            }

            if repetition_penalty != 1.0:
                _sp["repetition_penalty"] = repetition_penalty
            payload = {
                "text": full_text,
                "sampling_params": _sp,
            }
            r = await client.post(gen_url, json=payload)
            r.raise_for_status()
            out = r.json()
            text = out.get("text", "") or ""
            meta = out.get("meta_info", {}) or {}
            usage: dict = {}
            for k in ("prompt_tokens", "completion_tokens"):
                if meta.get(k) is not None:
                    usage[k] = meta[k]
            return text, usage

    return _gen

def apply_eval_mask(
    results: list[dict], mask_ids: set[int]
) -> tuple[list[dict], int]:
    
    if not mask_ids:
        return results, 0
    kept: list[dict] = []
    n_excluded = 0
    for row in results:
        raw = row.get("label_id")
        try:
            lid = int(raw)  
        except (TypeError, ValueError):
            kept.append(row)
            continue
        if lid in mask_ids:
            n_excluded += 1
        else:
            kept.append(row)
    return kept, n_excluded

def _summarize(results: list[dict]) -> dict[str, Any]:
    n = len(results)
    if n == 0:
        return {"n": 0}
    def _avg(key: str) -> float:
        return sum(float(r.get(key, 0)) for r in results) / n
    def _rate(key: str) -> float:
        return sum(1 for r in results if r.get(key)) / n

    def _cond_acc(pred) -> float | None:
        sub = [r for r in results if pred(r)]
        if not sub:
            return None
        return sum(1 for r in sub if r.get("correct")) / len(sub)

    def _cond_n(pred) -> int:
        return sum(1 for r in results if pred(r))

    loose_pass = [_is_valid_action_sequence((r.get("policy_text") or ""))[0] for r in results]
    loose_n = sum(loose_pass)
    loose_correct = sum(1 for r, lp in zip(results, loose_pass) if lp and r.get("correct"))

    by_class: dict[Any, list[int]] = {}  
    for r in results:
        cls = r.get("label_id")
        if cls is None:
            continue
        bucket = by_class.setdefault(cls, [0, 0])
        bucket[0] += 1
        bucket[1] += int(bool(r.get("correct")))
    per_class_acc = {
        str(cls): (b[1] / b[0]) for cls, b in sorted(by_class.items(), key=lambda kv: kv[0])
    }
    per_class_n = {
        str(cls): b[0] for cls, b in sorted(by_class.items(), key=lambda kv: kv[0])
    }

    tp: dict[str, int] = {}
    fp: dict[str, int] = {}
    fn: dict[str, int] = {}
    for r in results:
        cls = r.get("label_id")
        if cls is None:
            continue
        true_c = str(cls)
        if r.get("correct"):
            pred_c = true_c
        else:
            a = r.get("answer_content")
            pred_c = None if a is None else str(a)
        if pred_c == true_c:
            tp[true_c] = tp.get(true_c, 0) + 1
        else:
            fn[true_c] = fn.get(true_c, 0) + 1
            if pred_c is not None:
                fp[pred_c] = fp.get(pred_c, 0) + 1
    per_class_f1: dict[str, float] = {}
    for true_c in sorted(by_class.keys(), key=lambda c: c):
        c = str(true_c)
        c_tp, c_fp, c_fn = tp.get(c, 0), fp.get(c, 0), fn.get(c, 0)
        prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) else 0.0
        rec = c_tp / (c_tp + c_fn) if (c_tp + c_fn) else 0.0
        per_class_f1[c] = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    macro_f1 = (sum(per_class_f1.values()) / len(per_class_f1)) if per_class_f1 else 0.0

    pred_dist: dict[str, int] = {}
    for r in results:
        a = r.get("answer_content")
        if a is None:
            continue
        pred_dist[str(a)] = pred_dist.get(str(a), 0) + 1
    pred_dist = dict(sorted(pred_dist.items(), key=lambda kv: -kv[1]))

    summary: dict[str, Any] = {
        "n": n,
        "accuracy": _rate("correct"),
        "valid_format_rate": _rate("valid_format"),
        "has_answer_rate": _rate("has_answer"),
        "has_search_rate": _rate("has_search"),
        "valid_search_rate": _rate("valid_search"),
        "truncated_rate": _rate("truncated"),
        "avg_num_searches": _avg("num_searches"),
        "avg_num_turns": _avg("num_turns"),
        "avg_reward_tier": _avg("reward_tier"),
        
        "accuracy_given_valid_format":  _cond_acc(lambda r: r.get("valid_format")),
        "accuracy_given_has_answer":    _cond_acc(lambda r: r.get("has_answer")),
        "accuracy_given_not_truncated": _cond_acc(lambda r: not r.get("truncated")),
        "n_given_valid_format":         _cond_n(lambda r: r.get("valid_format")),
        "n_given_has_answer":           _cond_n(lambda r: r.get("has_answer")),
        "n_given_not_truncated":        _cond_n(lambda r: not r.get("truncated")),
        
        "loose_format_rate": loose_n / n,
        "accuracy_given_loose_format": (loose_correct / loose_n) if loose_n else None,
        
        "format_reason_breakdown": _format_reason_breakdown(results),
        
        "per_class_accuracy": per_class_acc,
        "per_class_n": per_class_n,
        "per_class_f1": per_class_f1,
        "macro_f1": macro_f1,
        "predicted_class_distribution": pred_dist,

        "avg_relevant_search_rate": (
            sum(r["relevant_search_rate"] for r in results
                if r.get("relevant_search_rate") is not None)
            / max(1, sum(1 for r in results if r.get("relevant_search_rate") is not None))
        ) if any(r.get("relevant_search_rate") is not None for r in results) else None,
        "any_relevant_searched_rate": _rate("any_relevant_searched"),
        "accuracy_given_any_relevant": _cond_acc(
            lambda r: r.get("any_relevant_searched") is True
        ),
        "accuracy_given_no_relevant":  _cond_acc(
            lambda r: r.get("num_searches", 0) > 0 and r.get("any_relevant_searched") is False
        ),
        
        "total_prompt_tokens": sum(int(r.get("prompt_tokens", 0) or 0) for r in results),
        "total_completion_tokens": sum(int(r.get("completion_tokens", 0) or 0) for r in results),
    }
    summary["total_tokens"] = summary["total_prompt_tokens"] + summary["total_completion_tokens"]
    
    n_correct = sum(1 for r in results if r.get("correct"))
    summary["n_correct"] = n_correct
    summary["avg_prompt_tokens"] = summary["total_prompt_tokens"] / n
    summary["avg_completion_tokens"] = summary["total_completion_tokens"] / n
    summary["avg_total_tokens"] = summary["total_tokens"] / n
    
    summary["tokens_per_correct"] = (summary["total_tokens"] / n_correct) if n_correct else None
    summary["completion_tokens_per_correct"] = (
        summary["total_completion_tokens"] / n_correct) if n_correct else None
    costs = [float(r["cost_usd"]) for r in results if r.get("cost_usd") is not None]
    if costs:
        total = sum(costs)
        summary["total_cost_usd"] = total
        summary["avg_cost_usd_per_prompt"] = total / n
    return summary

_LOOSE_SPLIT = re.compile(r"(</?(?:walk|answer)>)")
_LOOSE_MATCH = re.compile(r"</?(?:walk|answer)>")

def _is_valid_action_sequence(text: str) -> tuple[bool, str]:
    
    cleaned = text
    for _ctl in ("<|im_end|>", "<|end|>", "<|assistant|>", "<|im_start|>"):
        cleaned = cleaned.replace(_ctl, "")  
    cleaned = re.sub(r"<thinking>.*?</thinking>", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"</?thinking>", "", cleaned)  

    for tag in ("walk", "answer"):
        if len(re.findall(f"<{tag}>", cleaned)) != len(re.findall(f"</{tag}>", cleaned)):
            return False, f"unbalanced {tag}"

    state = "start"
    for part in _LOOSE_SPLIT.split(cleaned):
        if not part:
            continue
        if _LOOSE_MATCH.fullmatch(part):
            if part == "<walk>" and state in ("start", "after_search"):
                state = "in_search"
            elif part == "</walk>" and state == "in_search":
                state = "after_search"
            elif part == "<answer>" and state in ("start", "after_search"):
                state = "in_answer"
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False, f"unexpected {part} in state {state}"
        else:
            if state in ("in_search", "in_answer"):
                continue
            if part.strip():
                return False, f"text between tags in state {state}"

    if state != "end":
        return False, f"ended in state {state}"
    return True, "ok"

def _evaluate_policy_output_loose(policy_text: str, valid_node_ids: set[int]) -> dict:
    
    valid_format, format_reason = _is_valid_action_sequence(policy_text)
    base = _evaluate_policy_output(policy_text, valid_node_ids)
    base["valid_format"] = valid_format
    base["format_reason"] = format_reason
    return base

def write_summary_sidecar(jsonl_path: Path, summary: dict, meta: dict) -> Path:
    
    sidecar_path = jsonl_path.parent / (jsonl_path.stem + ".summary.json")
    payload = {"summary": summary, "meta": meta}
    sidecar_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return sidecar_path

def resummarize_jsonl(
    jsonl_path: Path | str,
    mask_labels: list[str],
    label_emb_dir: str,
    out_dir: Path | str | None,
) -> Path:
    
    from walker.tag.eval_filter import load_label_index, resolve_mask_ids

    jsonl_path = Path(jsonl_path)
    results = _read_jsonl(jsonl_path)

    dataset: str | None = results[0].get("dataset") if results else None

    mask_ids: set[int] = set()
    if mask_labels and dataset:
        idx = load_label_index(label_emb_dir, dataset)
        mask_ids = resolve_mask_ids(mask_labels, idx)

    kept, n_excluded = apply_eval_mask(results, mask_ids)
    summary = _summarize(kept)
    summary["n_excluded_masked"] = n_excluded
    summary["n_total_before_mask"] = len(results)

    meta = {
        "model": "(resummarized)",
        "dataset": dataset,
        "source_jsonl": str(jsonl_path),
        "eval_mask_labels": mask_labels,
        "n_excluded_masked": n_excluded,
    }

    if out_dir is not None:
        dest_dir = Path(out_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        synthetic = dest_dir / jsonl_path.name
    else:
        synthetic = jsonl_path

    return write_summary_sidecar(synthetic, summary, meta)

def _format_reason_breakdown(results: list[dict]) -> dict[str, int]:
    
    counts: dict[str, int] = {}
    for r in results:
        key = str(r.get("format_reason") or "(none)")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

def _print_sample_dump(results: list[dict], n: int) -> None:
    
    if n <= 0:
        return
    print()
    print("=" * 78)
    print(f"=== first {min(n, len(results))} samples ===")
    print("=" * 78)
    for i, r in enumerate(results[:n], start=1):
        print()
        print(f"--- sample {i}/{min(n, len(results))} "
              f"(dataset={r.get('dataset')} label={r.get('label_id')} "
              f"correct={r.get('correct')} valid_fmt={r.get('valid_format')} "
              f"fmt_reason={r.get('format_reason')!r}) ---")

        for msg in r.get("messages", []):
            role = msg.get("role", "?")
            content = (msg.get("content") or "")
            if role == "system":
                continue
            print(f"  [{role}]\n    {content[:1500]}{'...[truncated]' if len(content) > 1500 else ''}")
        print(f"  [result] correct={r.get('correct')} reward={r.get('reward_tier'):.2f} "
              f"searches={r.get('num_searches')} turns={r.get('num_turns')} "
              f"truncated={r.get('truncated')} answer={r.get('answer_content')!r}")
    print()
    print("=" * 78)
    print("=== format_reason histogram ===")
    print("=" * 78)
    breakdown = _format_reason_breakdown(results)
    total = sum(breakdown.values()) or 1
    for reason, c in breakdown.items():
        print(f"  {c:>4}  ({c / total:>5.1%})  {reason}")

def _reattach_stop_tag(text: str) -> str:
    
    if "<walk>" in text and "</walk>" not in text:
        return text + "</walk>"
    if "<answer>" in text and "</answer>" not in text:
        return text + "</answer>"
    return text

def _extract_reasoning(message) -> str:
    
    for attr in ("reasoning", "reasoning_content"):
        v = getattr(message, attr, None)
        if v:
            return str(v)
    if hasattr(message, "model_dump"):
        d = message.model_dump()
        return str(d.get("reasoning") or d.get("reasoning_content") or "")
    return ""

def _wrap_reasoning_as_thinking(text: str, reasoning: str) -> str:
    
    if not reasoning:
        return text
    safe_reasoning = reasoning
    for tag in ("<thinking>", "</thinking>", "<walk>", "</walk>",
                "<answer>", "</answer>"):
        safe_reasoning = safe_reasoning.replace(tag, "[" + tag[1:-1] + "]")
    stripped = re.sub(r"^\s*<thinking>.*?</thinking>\s*", "", text, count=1, flags=re.DOTALL)
    return f"<thinking>{safe_reasoning}</thinking>{stripped}"

def _make_chat_fn(client, *, model: str, max_tokens: int, temperature: float,
                  semaphore: asyncio.Semaphore, no_think: bool = False,
                  reasoning_as_thinking: bool = False,
                  reasoning_budget: int | None = None,
                  strict_greedy: bool = False,
                  stop: list[str] | None = None) -> ChatFn:
    
    if stop is None:
        stop = ["</walk>", "</answer>"]

    extra_body: dict = {"usage": {"include": True}}
    if no_think:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    if reasoning_budget is not None:
        extra_body["reasoning"] = {"max_tokens": int(reasoning_budget)}

    if strict_greedy:
        extra_body["repetition_penalty"] = 1.0
        extra_body["top_p"] = 1.0
        extra_body["top_k"] = -1
        extra_body["frequency_penalty"] = 0.0
        extra_body["presence_penalty"] = 0.0

    async def _chat(messages: list[dict]) -> tuple[str, dict]:
        async with semaphore:
            r = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                extra_body=extra_body,
            )
            msg = r.choices[0].message
            text = msg.content or ""
            if reasoning_as_thinking:
                text = _wrap_reasoning_as_thinking(text, _extract_reasoning(msg))
            text = _reattach_stop_tag(text)
            usage: dict = {}
            if getattr(r, "usage", None) is not None:
                u = r.usage
                for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    v = getattr(u, k, None)
                    if v is not None:
                        usage[k] = v
                
                cost = getattr(u, "cost", None)
                if cost is None and hasattr(u, "model_dump"):
                    cost = u.model_dump().get("cost")
                if cost is not None:
                    usage["cost"] = cost
                
                details = getattr(u, "completion_tokens_details", None)
                if details is not None:
                    rt = getattr(details, "reasoning_tokens", None)
                    if rt is not None:
                        usage["reasoning_tokens"] = rt
            return text, usage

    return _chat

def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def _make_run_id(model: str, prompt_path: Path) -> str:
    safe_model = model.replace("/", "_").replace(":", "_")
    return f"{safe_model}__{prompt_path.stem}__{datetime.now():%Y%m%d_%H%M%S}"

async def _run_all(
    rows: list[dict],
    episode_coro_fn: Callable[[dict], Awaitable[dict]],
    *,
    out_path: Path,
    chunk_size: int = 256,
) -> list[dict]:
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(rows)
    results: list[dict] = []
    t0 = time.time()
    last_report = t0
    
    step = max(total // 20, 5)
    with out_path.open("w") as f:
        for chunk_start in range(0, total, chunk_size):
            coros = [episode_coro_fn(row)
                     for row in rows[chunk_start:chunk_start + chunk_size]]
            for fut in asyncio.as_completed(coros):
                result = await fut
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                results.append(result)
                done = len(results)
                now = time.time()
                if done == total or done % step == 0 or (now - last_report) >= 30:
                    acc_so_far = sum(1 for r in results if r.get("correct")) / done
                    fmt_so_far = sum(1 for r in results if r.get("valid_format")) / done
                    elapsed = now - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (total - done) / rate if rate > 0 else 0
                    print(
                        f"[bench] {done}/{total} ({done/total:>5.1%})  "
                        f"acc={acc_so_far:.3f}  valid_fmt={fmt_so_far:.3f}  "
                        f"elapsed={elapsed:.0f}s  eta={eta:.0f}s",
                        file=sys.stderr,
                        flush=True,
                    )
                    last_report = now
    return results

def _maybe_init_wandb(cfg, run_id: str):
    try:
        import wandb  
    except ImportError:
        print("[bench] wandb requested but not installed; skipping", file=sys.stderr)
        return None
    return wandb.init(
        project=cfg["wandb_project"],
        name=run_id,
        group=cfg.get("wandb_group"),
        notes=cfg.get("wandb_notes"),
        tags=cfg["wandb_tags"].split(",") if cfg.get("wandb_tags") else None,
        config={k: v for k, v in cfg.items()
                if k not in {"wandb_project", "wandb_group", "wandb_notes", "wandb_tags"}},
    )

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Walker benchmark: run inference on a prompt jsonl, score with training-side reward.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--prompt-data", default=None, help="Path to a jsonl produced by walker.tag")
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible /v1 endpoint")
    ap.add_argument("--model", default=None, help="Model name passed in chat.completions.create")
    ap.add_argument("--resummarize", default=None, metavar="JSONL_PATH",
                    help="Skip inference entirely: read an existing per-example jsonl, "
                         "apply --eval-mask-labels, recompute the summary, and write "
                         "the sidecar. Exits after writing; --prompt-data/--base-url/"
                         "--model are not required when this flag is set.")
    ap.add_argument("--api-key-env", default=None,
                    help="Env var holding the API key (e.g. OPENROUTER_API_KEY). "
                         "Omit for local SGLang.")
    ap.add_argument("--data-path", default="data/raw_datasets",
                    help="Root for raw graph data (used by GraphEnv search side)")
    ap.add_argument("--max-in-flight", type=int, default=32)
    ap.add_argument("--chunk-size", type=int, default=256,
                    help="Episodes processed per batch (each builds a GraphEnv). "
                         "Lower it when GraphEnv is heavy (large graph) AND "
                         "generations are long: "
                         "with chunk >> in-flight, the waiting coroutines hold many "
                         "constructed GraphEnv objects and thrash memory. ~2x "
                         "--max-in-flight is safe for those cases.")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-searches", type=int, default=None, metavar="N",
                    help="Override env_cfg.max_searches (and set "
                         "enforce_search_budget=True). Pass 0 to forbid "
                         "any <walk> action — used by direct (single-turn) prompt "
                         "forms in the prompt-form benchmark sweep. "
                         "When unset, falls back to GraphEnvConfig.from_env().")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.0,
                    help="SGLang repetition_penalty for text-mode /generate. "
                         "1.0 = off (default; payload unchanged). >1.0 breaks "
                         "greedy repetition loops on hard/OOD tasks (e.g. edge "
                         "classification) that else degenerate + skip </thinking>.")
    ap.add_argument("--generate-mode", choices=["chat", "text"], default="chat",
                    help="chat (default): /v1/chat/completions, walk results "
                         "appended as user messages (chat template inserts role "
                         "markers each turn). text: SGLang native /generate, the "
                         "whole trajectory is one continuous text stream with walk "
                         "results inlined (no role markers) — faithful to the "
                         "training rollout. Use 'text' to evaluate a trained walk "
                         "policy (CNY); 'chat' underrates it by ~5pp (OOD format). "
                         "Requires a local --model path (loads tokenizer for the "
                         "chat template).")
    ap.add_argument("--answer-style", choices=["tag", "rethink"], default="tag",
                    help="tag (default): extract <answer>N</answer> (walker format). "
                         "rethink: extract the integer after 'Answer:' (Graph-R1's "
                         "rethink template emits a long <think> block then "
                         "'Answer: <id>'). Use with --generate-mode chat "
                         "--max-searches 0 to evaluate the Graph-R1 baseline.")
    ap.add_argument("--no-think", action="store_true",
                    help="Pass chat_template_kwargs={enable_thinking: False} for "
                         "Qwen3 hybrid (dense) models so they skip <think> blocks. "
                         "Ignored by other model families.")
    ap.add_argument("--reasoning-as-thinking", action="store_true",
                    help="For reasoning models (o1 / r1 / qwen-thinking): extract "
                         "message.reasoning and wrap as <thinking>...</thinking> "
                         "in the response, so walker format check accepts it.")
    ap.add_argument("--reasoning-budget", type=int, default=None, metavar="N",
                    help="OpenRouter only — separate token budget for chain-of-"
                         "thought (kept distinct from --max-tokens). When "
                         "--reasoning-as-thinking is set without an explicit "
                         "--reasoning-budget, defaults to 4096 to prevent the "
                         "reasoner from eating max_tokens and leaving content empty.")
    ap.add_argument("--loose-format", action="store_true",
                    help="Use action-only format validator: only check "
                         "<walk>/<answer> ordering, ignore <thinking> tags. "
                         "For benchmarking models that don't follow walker's "
                         "training-target format (reasoning models, Llama, etc.). "
                         "Auto-defaulted ON when --reasoning-as-thinking is set.")
    ap.add_argument("--strict-greedy", action="store_true",
                    help="Override generation_config.json defaults: send explicit "
                         "repetition_penalty=1.0, top_p=1.0, top_k=-1, "
                         "freq_penalty=0, presence_penalty=0. Some SGLang versions "
                         "inherit these silently from the model config, biasing "
                         "logits even at temperature=0 and producing platform-"
                         "dependent drift between SGLang versions.")
    ap.add_argument("--show-samples", type=int, default=0, metavar="N",
                    help="After run, print the first N completed samples in full "
                         "(policy_text + format_reason + reward fields). Also "
                         "prints a histogram of format_reason. Use with --limit "
                         "for fast prompt iteration. 0 = silent (default).")
    ap.add_argument("--out-dir", default="outputs/bench")
    ap.add_argument("--eval-mask-labels", default="",
                    help="Pipe-separated (|) humanized label names to EXCLUDE from "
                         "scoring (gold label in this set → row dropped from summary "
                         "computation). Raw per-example jsonl is unaffected. Pipe (not "
                         "comma) because some label names contain commas. "
                         "Requires --label-emb-dir. Example: 'Quantum Physics|Robotics'")
    ap.add_argument("--label-emb-dir", default="data/label_embeddings",
                    help="Directory holding <dataset>.json label-index files "
                         "produced by scripts/build_label_embeddings.py. "
                         "Used to resolve --eval-mask-labels names to integer ids.")
    ap.add_argument("--run-label", default=None,
                    help="Table-row label for walker.eval.aggregate (e.g. "
                         "'ego-full', 'Top-K'). Stored in the summary sidecar's "
                         "meta.row_label; aggregate groups rows by it (falling "
                         "back to --model). Use to give each frozen prompt form "
                         "its own row in the prompt-selection block.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap number of rows (debug aid; subsample is built upstream)")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="graphwalker_bench")
    ap.add_argument("--wandb-group", default=None)
    ap.add_argument("--wandb-notes", default=None)
    ap.add_argument("--wandb-tags", default=None)
    args = ap.parse_args(argv)

    if args.resummarize:
        mask_names: list[str] = []
        if args.eval_mask_labels:
            mask_names = [s.strip() for s in args.eval_mask_labels.split("|") if s.strip()]
        sidecar_path = resummarize_jsonl(
            jsonl_path=Path(args.resummarize),
            mask_labels=mask_names,
            label_emb_dir=args.label_emb_dir,
            out_dir=Path(args.out_dir) if args.out_dir else None,
        )
        print(f"[bench] resummarize -> {sidecar_path}", file=sys.stderr)
        return 0

    for flag, val in [("--prompt-data", args.prompt_data),
                      ("--base-url", args.base_url),
                      ("--model", args.model)]:
        if not val:
            print(f"[bench] {flag} is required when --resummarize is not set",
                  file=sys.stderr)
            return 2

    prompt_path = Path(args.prompt_data)
    if not prompt_path.exists():
        print(f"[bench] no prompt jsonl at {prompt_path}", file=sys.stderr)
        return 2
    rows = _read_jsonl(prompt_path)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print(f"[bench] empty jsonl: {prompt_path}", file=sys.stderr)
        return 2

    from openai import AsyncOpenAI

    api_key = "EMPTY"
    if args.api_key_env:
        api_key = os.environ.get(args.api_key_env, "")
        if not api_key:
            print(f"[bench] {args.api_key_env} not set in env", file=sys.stderr)
            return 2

    client = AsyncOpenAI(base_url=args.base_url, api_key=api_key)
    sem = asyncio.Semaphore(args.max_in_flight)

    reasoning_budget = args.reasoning_budget
    if args.reasoning_as_thinking and reasoning_budget is None:
        reasoning_budget = 4096
        print(f"[bench] --reasoning-as-thinking set; auto-defaulted "
              f"--reasoning-budget {reasoning_budget} (override with --reasoning-budget N)",
              file=sys.stderr)

    loose_format = args.loose_format
    if args.reasoning_as_thinking and not loose_format:
        loose_format = True
        print("[bench] --reasoning-as-thinking set; auto-enabled --loose-format "
              "(action-only validator).", file=sys.stderr)

    chat_stop = ["Brief_reasoning"] if args.answer_style == "rethink" else None
    chat_fn = _make_chat_fn(client, model=args.model,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            semaphore=sem,
                            no_think=args.no_think,
                            reasoning_as_thinking=args.reasoning_as_thinking,
                            reasoning_budget=reasoning_budget,
                            strict_greedy=args.strict_greedy,
                            stop=chat_stop)

    env_cfg = GraphEnvConfig.from_env()
    if args.max_searches is not None:
        from dataclasses import replace
        env_cfg = replace(
            env_cfg,
            max_searches=int(args.max_searches),
            max_turns=max(env_cfg.max_turns, int(args.max_searches) + 1),
            enforce_search_budget=True,
        )
        print(f"[bench] --max-searches override: max_searches={env_cfg.max_searches} "
              f"max_turns={env_cfg.max_turns} enforce=True", file=sys.stderr)

    from walker.search._summarize import Summarizer
    summarizer = Summarizer.from_env_cfg(env_cfg, client=client, model=args.model)

    run_id = _make_run_id(args.model, prompt_path)
    out_path = Path(args.out_dir) / f"{run_id}.jsonl"

    wb = None
    if args.wandb:
        wb = _maybe_init_wandb(
            {
                "wandb_project": args.wandb_project,
                "wandb_group": args.wandb_group,
                "wandb_notes": args.wandb_notes,
                "wandb_tags": args.wandb_tags,
                "model": args.model,
                "base_url": args.base_url,
                "prompt_data": str(prompt_path),
                "max_in_flight": args.max_in_flight,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "n_prompts": len(rows),
            },
            run_id,
        )

    if env_cfg.max_searches == 0:

        print("[bench] single-turn baseline (max_searches=0): no GraphEnv",
              file=sys.stderr)

        def _episode_coro(row):
            return run_episode_single_turn(chat_fn, row, loose_format=loose_format,
                                           answer_style=args.answer_style)
    elif args.generate_mode == "text":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        gen_fn = _make_generate_fn(args.base_url, max_tokens=args.max_tokens,
                                   temperature=args.temperature, semaphore=sem,
                                   repetition_penalty=args.repetition_penalty)
        print("[bench] generate-mode=text (training-faithful /generate, inline obs)",
              file=sys.stderr)

        def _episode_coro(row):
            return run_episode_text(gen_fn, row, env_cfg, tokenizer=tokenizer,
                                    data_path=args.data_path,
                                    loose_format=loose_format, summarizer=summarizer,
                                    answer_style=args.answer_style)
    else:
        def _episode_coro(row):
            return run_episode(chat_fn, row, env_cfg, data_path=args.data_path,
                               loose_format=loose_format, summarizer=summarizer,
                               answer_style=args.answer_style)

    print(f"[bench] {args.model} prompt={prompt_path.name} n={len(rows)} "
          f"in_flight={args.max_in_flight} → {out_path}", file=sys.stderr)
    t0 = time.time()
    results = asyncio.run(_run_all(rows, _episode_coro, out_path=out_path,
                                   chunk_size=args.chunk_size))
    wall = time.time() - t0

    mask_names_post: list[str] = []
    n_excluded = 0
    if args.eval_mask_labels:
        mask_names_post = [s.strip() for s in args.eval_mask_labels.split("|") if s.strip()]
    if mask_names_post:
        from walker.tag.eval_filter import load_label_index, resolve_mask_ids
        dataset_for_mask = results[0].get("dataset") if results else None
        idx = load_label_index(args.label_emb_dir, dataset_for_mask)
        mask_ids = resolve_mask_ids(mask_names_post, idx)
        kept_results, n_excluded = apply_eval_mask(results, mask_ids)
        summary = _summarize(kept_results)
        summary["n_excluded_masked"] = n_excluded
        summary["n_total_before_mask"] = len(results)
    else:
        summary = _summarize(results)

    summary["wall_time_sec"] = wall
    _n_run = len(results)
    summary["examples_per_sec"] = (_n_run / wall) if wall > 0 else None
    _tot_tok = sum(int(r.get("prompt_tokens", 0) or 0)
                   + int(r.get("completion_tokens", 0) or 0) for r in results)
    summary["tokens_per_sec"] = (_tot_tok / wall) if wall > 0 else None
    summary["generate_mode"] = args.generate_mode
    summary["answer_style"] = args.answer_style
    summary["max_in_flight"] = args.max_in_flight

    meta = {
        "model": args.model,
        "row_label": args.run_label,
        "prompt_data": str(prompt_path),
        "dataset": (results[0].get("dataset") if results else "unknown"),
        "max_searches": env_cfg.max_searches,
        "run_id": run_id,
        "n": summary.get("n"),
        "eval_mask_labels": mask_names_post,
        "n_excluded_masked": n_excluded,
    }
    sidecar_path = write_summary_sidecar(out_path, summary, meta)
    print(f"[bench] summary -> {sidecar_path}", file=sys.stderr)

    cost_str = ""
    if "total_cost_usd" in summary:
        cost_str = (
            f" cost=${summary['total_cost_usd']:.4f} "
            f"(${summary['avg_cost_usd_per_prompt']:.5f}/prompt)"
        )
    acc_fmt = summary.get("accuracy_given_valid_format")
    acc_fmt_str = f"{acc_fmt:.3f}" if acc_fmt is not None else "n/a"
    masked_excl_str = f" masked_excl={n_excluded}" if mask_names_post else ""
    print(
        f"[bench] done {args.model} n={summary['n']} acc={summary['accuracy']:.3f} "
        f"macro_f1={summary['macro_f1']:.3f} "
        f"valid_fmt={summary['valid_format_rate']:.3f} "
        f"acc|fmt={acc_fmt_str} "
        f"loose_fmt={summary['loose_format_rate']:.3f} "
        f"avg_searches={summary['avg_num_searches']:.2f} "
        f"truncated={summary['truncated_rate']:.3f} "
        f"tokens={summary['total_tokens']}"
        f"{cost_str}"
        f"{masked_excl_str} "
        f"t={wall:.0f}s"
    )
    if args.show_samples:
        _print_sample_dump(results, args.show_samples)
    if wb is not None:
        wb.log(summary)
        wb.finish()
    return 0

if __name__ == "__main__":
    sys.exit(main())
