from __future__ import annotations

import re
from typing import Any

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

from walker.search._env_compat import get_env
from walker.search._format import (
    _compute_reward_tier,
    _evaluate_policy_output,
    _extract_answer,
    _is_valid_sequence,
    _rebuild_policy_text,
    _score_prediction,
    postprocess_responses,
)
from walker.search._token_align import find_search_action_span
from walker.search.env import (
    GraphEnv,
    GraphEnvStep,
    NODE_SEARCH_CONFIGS,
    WalkEgoSnapshot,
    _FORCE_MIN_SEARCH,
    _POST_SEARCH_NUDGE,
    _WALK_EXTEND_ENABLED,
    _WALK_NEIGHBOR_LIMIT,
    _WALK_NEIGHBOR_SAMPLING,
    _WALK_NODE_CONTENT_TOKENS,
    _WALK_PREVIEW_TOKENS,
    _extract_first_valid_search_id,
    _get_view,
    _normalize_label,
    _normalize_search_id,
    execute_predictions,
    postprocess_predictions,
)

def _get_reward_alpha(args: Any) -> float:
    
    alpha = getattr(args, "reward_alpha", get_env("WALKER_REWARD_ALPHA", default="1.0"))
    try:
        return float(alpha)
    except (TypeError, ValueError):
        return 1.0

def _lookup_token_id(tokenizer, token_str: str) -> int | None:
    try:
        tok_id = tokenizer.convert_tokens_to_ids(token_str)
    except Exception:
        tok_id = None
    if isinstance(tok_id, list):
        tok_id = tok_id[0] if tok_id else None
    if tok_id is None or tok_id == getattr(tokenizer, "unk_token_id", None):
        try:
            encoded = tokenizer.encode(token_str, add_special_tokens=False)
            if len(encoded) == 1:
                tok_id = encoded[0]
        except Exception:
            tok_id = None
    return tok_id

def _chat_stop_token_ids(tokenizer) -> set[int]:
    stop_ids: set[int] = set()
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        stop_ids.add(eos_id)
    for token_str in ("<|im_end|>", "<|end|>", "<|assistant|>"):
        tok_id = _lookup_token_id(tokenizer, token_str)
        if tok_id is not None:
            stop_ids.add(tok_id)
    return stop_ids

def _strip_chat_stop_tokens(tokenizer, token_ids: list[int], log_probs: list[float] | None = None):
    stop_ids = _chat_stop_token_ids(tokenizer)
    trimmed_ids = list(token_ids)
    trimmed_log = list(log_probs) if log_probs is not None else None
    changed = False
    while trimmed_ids and trimmed_ids[-1] in stop_ids:
        trimmed_ids.pop()
        if trimmed_log is not None and trimmed_log:
            trimmed_log.pop()
        changed = True
    trimmed_text = None
    if changed:
        trimmed_text = tokenizer.decode(trimmed_ids, skip_special_tokens=False).rstrip()
    return trimmed_ids, trimmed_log, trimmed_text

def _truncate_to_first_action(
    tokenizer,
    token_ids: list[int],
    log_probs: list[float] | None = None,
) -> tuple[list[int], list[float] | None, str, str | None, bool]:
    if not token_ids:
        return token_ids, log_probs, "", None, False
    decoded_chunks: list[str] = []
    action = None
    end_idx = len(token_ids)
    for idx, tid in enumerate(token_ids):
        decoded_chunks.append(tokenizer.decode([tid], skip_special_tokens=False))
        current_text = "".join(decoded_chunks)
        match = re.search(r"<(walk|answer)>.*?</\1>", current_text, re.DOTALL)
        if match:

            action = "search" if match.group(1) == "walk" else match.group(1)
            end_idx = idx + 1
            break
    truncated_ids = token_ids[:end_idx]
    truncated_logs = log_probs[:end_idx] if log_probs is not None else None
    truncated_text = tokenizer.decode(truncated_ids, skip_special_tokens=False) if truncated_ids else ""
    tail_dropped = len(token_ids) > end_idx
    return truncated_ids, truncated_logs, truncated_text, action, tail_dropped

async def generate(args, sample: Sample, sampling_params, evaluation: bool = False) -> Sample:
    
    assert not args.partial_rollout, "Partial rollout is not supported by this generate function."

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    env = GraphEnv.from_args(args, sample)
    env.reset(sample)

    prompt_text = sample.prompt
    prompt_token_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    response = ""
    response_token_ids: list[int] = []
    loss_mask: list[int] = []
    rollout_log_probs: list[float] | None = [] if NODE_SEARCH_CONFIGS["return_logprob"] else None

    truncated_tail = False
    total_tail_tokens = 0  
    turns_used = 0
    bpe_drift_skipped_count = 0  
    search_action_spans: list[tuple[int, int]] = []

    walk_ego_snapshots: list[WalkEgoSnapshot] = []

    finish_reason = "stop"  

    max_turns = env.max_turns
    for _turn_idx in range(max_turns):
        turns_used = _turn_idx + 1
        payload: dict = {
            "text": prompt_text + response,
            "sampling_params": sampling_params,
        }
        if NODE_SEARCH_CONFIGS["return_logprob"]:
            payload["return_logprob"] = True

        output = await post(url, payload)
        finish_reason = output["meta_info"]["finish_reason"]["type"]

        if finish_reason == "abort":
            sample.tokens = prompt_token_ids + response_token_ids
            sample.response_length = len(response_token_ids)
            sample.response = response
            sample.loss_mask = loss_mask
            sample.prompt = prompt_text
            if NODE_SEARCH_CONFIGS["return_logprob"] and rollout_log_probs is not None:
                sample.rollout_log_probs = rollout_log_probs
            abort_meta = dict(sample.metadata or {})
            abort_meta["node_turns_used"] = int(turns_used)
            abort_meta["node_answered"] = False
            abort_meta["node_budget_exhausted"] = False
            abort_meta["node_finish_reason"] = "abort"
            abort_meta["node_tail_token_count"] = int(total_tail_tokens)
            abort_meta["node_walk_extend_enabled"] = bool(env.walk_extend_enabled)
            abort_meta["node_walk_hops"] = int(len(env.walk_path))
            abort_meta["node_walk_path"] = list(env.walk_path)
            abort_meta["node_walk_valid_ids"] = sorted(env.valid_node_ids)
            abort_meta["node_force_min_triggers"] = int(env.force_min_triggers)
            abort_meta["node_non_canonical_search_attempts"] = int(env.non_canonical_search_attempts)
            sample.metadata = abort_meta
            sample.status = Sample.Status.ABORTED
            return sample

        cur_response = output["text"]

        if NODE_SEARCH_CONFIGS["return_logprob"]:
            if "output_token_logprobs" not in output["meta_info"]:
                raise RuntimeError(
                    "output_token_logprobs not found in output meta_info. "
                    "Ensure 'return_logprob': True is passed in the payload."
                )
            cur_response_token_ids = [
                item[1] for item in output["meta_info"]["output_token_logprobs"]
            ]
            cur_response_log_probs = [
                item[0] for item in output["meta_info"]["output_token_logprobs"]
            ]
            cur_response = state.tokenizer.decode(cur_response_token_ids, skip_special_tokens=False)
        else:
            cur_response = postprocess_responses(cur_response)
            cur_response_token_ids = state.tokenizer(
                cur_response, add_special_tokens=False
            )["input_ids"]
            cur_response_log_probs = None

        cur_response_token_ids, cur_response_log_probs, trimmed_text = _strip_chat_stop_tokens(
            state.tokenizer, cur_response_token_ids, cur_response_log_probs
        )
        if trimmed_text is not None:
            cur_response = trimmed_text
        else:
            cur_response = cur_response.rstrip()

        clean_ids, _clean_logs, clean_text, _action_tag, tail_dropped = (
            _truncate_to_first_action(
                state.tokenizer,
                cur_response_token_ids,
                cur_response_log_probs,
            )
        )
        clean_text = clean_text.rstrip()
        tail_token_count = len(cur_response_token_ids) - len(clean_ids)
        if tail_dropped:
            truncated_tail = True
        total_tail_tokens += tail_token_count

        response_token_ids += cur_response_token_ids
        loss_mask += [1] * len(cur_response_token_ids)
        if NODE_SEARCH_CONFIGS["return_logprob"] and cur_response_log_probs is not None:
            rollout_log_probs += cur_response_log_probs  

        response += clean_text
        cur_response = clean_text  

        if finish_reason == "length":
            action, _ = postprocess_predictions(cur_response)
            if action is None:
                break  

        step = await env.step(cur_response)

        if (
            step.successful_search
            and step.new_valid_ids
            and step.searched_node_id is not None
        ):
            local_span = find_search_action_span(state.tokenizer, cur_response_token_ids)
            if local_span is None:

                bpe_drift_skipped_count += 1
            else:
                turn_start_abs = len(response_token_ids) - len(cur_response_token_ids)
                action_abs_start = turn_start_abs + local_span[0]
                action_abs_end = turn_start_abs + local_span[1]
                search_action_spans.append((action_abs_start, action_abs_end))
                env.record_walk_step(step.searched_node_id)

                assert step.walk_ego is not None, (
                    "GraphEnvStep.walk_ego must be set when successful_search "
                    "and new_valid_ids both hold"
                )
                walk_ego_snapshots.append(step.walk_ego)

        if step.done:
            break

        assert step.obs_text != "", "Observation should not be empty for non-terminal actions."
        obs_token_ids = state.tokenizer(step.obs_text, add_special_tokens=False)["input_ids"]
        response += step.obs_text
        response_token_ids += obs_token_ids
        loss_mask += [0] * len(obs_token_ids)

        if NODE_SEARCH_CONFIGS["return_logprob"]:
            rollout_log_probs += [0.0] * len(obs_token_ids)  
            assert len(response_token_ids) == len(rollout_log_probs), (  
                f"Token/logprob length mismatch: "
                f"{len(response_token_ids)} tokens vs {len(rollout_log_probs)} logprobs"
            )

    sample.tokens = prompt_token_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_mask
    sample.prompt = prompt_text

    if NODE_SEARCH_CONFIGS["return_logprob"]:
        sample.rollout_log_probs = rollout_log_probs if rollout_log_probs else None

    meta = dict(sample.metadata or {})
    if truncated_tail:
        meta["node_truncated_action"] = True

    meta["node_tail_token_count"] = int(total_tail_tokens)
    meta["node_turns_used"] = int(turns_used)
    meta["node_answered"] = bool(env.answered)
    meta["node_budget_exhausted"] = bool((not env.answered) and turns_used >= max_turns)
    meta["node_max_searches"] = int(env.max_searches)
    meta["node_search_count"] = int(env.successful_search_count)
    meta["node_search_budget_enforced"] = bool(env.enforce_search_budget)
    meta["node_bpe_drift_skipped"] = int(bpe_drift_skipped_count)
    meta["node_finish_reason"] = str(finish_reason)
    meta["node_walk_extend_enabled"] = bool(env.walk_extend_enabled)
    meta["node_walk_hops"] = int(len(env.walk_path))
    meta["node_walk_path"] = list(env.walk_path)

    meta["node_walk_valid_ids"] = sorted(env.valid_node_ids)
    meta["search_action_spans"] = list(search_action_spans)
    meta["walk_ego_snapshots"] = list(walk_ego_snapshots)

    meta["node_force_min_triggers"] = int(env.force_min_triggers)

    meta["node_non_canonical_search_attempts"] = int(env.non_canonical_search_attempts)
    sample.metadata = meta

    if (
        not evaluation
        and getattr(args, "method", None) == "opd"
        and getattr(args, "enable_opd", False)
    ):
        from walker.opd.rollout_branch import apply_opd_branch
        await apply_opd_branch(
            args,
            sample,
            tokenizer=state.tokenizer,
            prompt_text=prompt_text,
            response_token_ids=response_token_ids,
            graph_state=env.graph_state,
            meta=meta,
        )

    match finish_reason:
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED
        case "stop":
            sample.status = Sample.Status.COMPLETED

    return sample

async def reward_func(args, sample: Sample, **kwargs) -> float:
    
    if not isinstance(sample, Sample):
        raise TypeError("sample must be an instance of Sample.")

    label = _normalize_label(sample.label)

    label_id: int     = int(label.get("ground_truth", -1))
    label_names: list = label.get("label_names", [])
    task_type: str    = str(label.get("task_type", "node_class"))
    target_neighbor_ids = {int(x) for x in label.get("neighbor_ids", [])}

    walk_valid_ids = {int(x) for x in (sample.metadata or {}).get("node_walk_valid_ids", [])}
    valid_node_ids = target_neighbor_ids | walk_valid_ids

    tokenizer = GenerateState(args).tokenizer
    policy_text = _rebuild_policy_text(sample, tokenizer)
    eval_info = _evaluate_policy_output(policy_text, valid_node_ids)

    correct = (
        eval_info["has_answer"]
        and _score_prediction(eval_info["answer_content"] or "", label_id, label_names, task_type)
    )
    base_tier = _compute_reward_tier(
        correct=correct,
        valid_format=eval_info["valid_format"],
        valid_search=eval_info["valid_search"],
        has_answer=eval_info["has_answer"],
    )

    alpha = _get_reward_alpha(args)

    metadata = dict(sample.metadata or {})
    metadata["node_policy_text_len"] = len(policy_text)
    metadata["node_valid_format"] = bool(eval_info["valid_format"])
    metadata["node_format_reason"] = str(eval_info["format_reason"])
    metadata["node_has_answer"] = bool(eval_info["has_answer"])
    metadata["node_has_search"] = bool(eval_info["has_search"])
    metadata["node_valid_search"] = bool(eval_info["valid_search"])
    metadata["node_search_id"] = eval_info["search_node_id"]
    metadata["node_correct"] = bool(correct)
    metadata["node_reward_base"] = float(base_tier)
    metadata["node_reward_accuracy"] = 1.0 if correct else 0.0
    metadata["node_reward_format"] = float(base_tier)
    if eval_info["has_answer"]:
        metadata["node_answer_without_search"] = not eval_info["has_search"]
        metadata["node_answer_after_search"] = bool(eval_info["has_search"])

    reward = alpha * base_tier
    metadata["node_reward_penalty"] = 0.0
    metadata["node_reward_total_before_clip"] = float(reward)
    sample.metadata = metadata
    final_reward = max(reward, 0.0)

    if get_env("WALKER_DUMP_PER_TOKEN", default="0") == "1":
        from walker.opd.per_token_dump import emit_reward
        emit_reward(
            cell_id=get_env("WALKER_DUMP_CELL_ID", default="unknown"),
            step_idx=int(get_env("WALKER_TRAINING_STEP", default="0") or "0"),
            sample_idx=int(sample.index if sample.index is not None else 0),
            reward=final_reward,
            correct=correct,
            metadata=metadata,
        )

    return final_reward

def post_process_rewards(args, samples: list[Sample], **kwargs):
    
    rewards = [float(s.reward) if isinstance(s.reward, (int, float)) else 0.0 for s in samples]
    return rewards, rewards

from walker.search._reward_metrics import (  
    _collect_reward_metrics,
    _maybe_log_sample_table,
    log_eval_rollout_data,
    log_rollout_data,
)
