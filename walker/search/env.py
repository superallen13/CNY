from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from typing import Any

from walker.search._base_env import BaseInteractionEnv
from walker.search._env_compat import get_env

@dataclass(frozen=True)
class GraphEnvConfig:

    max_turns: int = 2
    max_searches: int = 2
    enforce_search_budget: bool = False

    force_min_search: int = 0

    walk_extend_enabled: bool = False
    walk_preview_tokens: int = 30
    walk_node_content_tokens: int = 200
    walk_neighbor_limit: int = 3
    walk_neighbor_sampling: str = "deterministic"

    post_search_nudge: str = ""

    return_logprob: bool = False

    summary_enabled: bool = False
    summary_max_tokens: int = 40
    summary_temperature: float = 0.0
    summary_template: str = (
        "Summarize the following passage in <= 40 tokens. "
        "Keep concrete entities and terminology that would help a graph "
        "classifier decide its category. Output the summary text only.\n\n"
        "Passage:\n{text}\n\nSummary:"
    )

    strict_search_format: bool = True

    @classmethod
    def from_env(cls) -> "GraphEnvConfig":
        
        legacy_max_hops = int(get_env("WALKER_MAX_HOPS", default="2"))
        max_searches_raw = get_env("WALKER_MAX_SEARCHES")
        if max_searches_raw is not None:
            max_searches = max(int(max_searches_raw), 0)
            max_turns = max(legacy_max_hops, max_searches + 1)
            enforce = True
        else:
            max_searches = legacy_max_hops
            max_turns = legacy_max_hops
            enforce = False

        return_logprob = False
        rl_raw = get_env("WALKER_RETURN_LOGPROB")
        if rl_raw is not None:
            try:
                return_logprob = bool(int(rl_raw))
            except ValueError:
                pass

        summary_enabled = bool(int(get_env("WALKER_SUMMARY_ENABLED", default="0")))
        summary_template = get_env(
            "WALKER_SUMMARY_TEMPLATE",
            default=(
                "Summarize the following passage in <= 40 tokens. "
                "Keep concrete entities and terminology that would help a graph "
                "classifier decide its category. Output the summary text only.\n\n"
                "Passage:\n{text}\n\nSummary:"
            ),
        )
        if summary_enabled and "{text}" not in summary_template:
            raise ValueError(
                "summary_template (WALKER_SUMMARY_TEMPLATE) must contain a literal "
                f"'{{text}}' placeholder; got: {summary_template!r}"
            )

        return cls(
            max_turns=max_turns,
            max_searches=max_searches,
            enforce_search_budget=enforce,
            force_min_search=int(get_env("WALKER_FORCE_MIN_SEARCH", default="0")),
            walk_extend_enabled=bool(int(get_env("WALKER_WALK_EXTEND", default="0"))),
            walk_preview_tokens=int(get_env("WALKER_PREVIEW_TOKENS", default="30")),
            walk_node_content_tokens=int(get_env("WALKER_NODE_CONTENT_TOKENS", default="200")),
            walk_neighbor_limit=int(get_env("WALKER_NEIGHBOR_LIMIT", default="3")),
            walk_neighbor_sampling=get_env("WALKER_NEIGHBOR_SAMPLING", default="deterministic"),
            post_search_nudge=get_env("WALKER_POST_SEARCH_NUDGE", default="") or "",
            return_logprob=return_logprob,
            summary_enabled=summary_enabled,
            summary_max_tokens=int(get_env("WALKER_SUMMARY_MAX_TOKENS", default="40")),
            summary_temperature=float(get_env("WALKER_SUMMARY_TEMPERATURE", default="0.0")),
            summary_template=summary_template,
            strict_search_format=bool(int(get_env("WALKER_STRICT_SEARCH_FORMAT", default="1"))),
        )

_DEFAULT_CONFIG: GraphEnvConfig = GraphEnvConfig.from_env()

NODE_SEARCH_CONFIGS: dict[str, Any] = {
    "max_turns": _DEFAULT_CONFIG.max_turns,
    "max_searches": _DEFAULT_CONFIG.max_searches,
    "enforce_search_budget": _DEFAULT_CONFIG.enforce_search_budget,
    "return_logprob": _DEFAULT_CONFIG.return_logprob,
    
    "reward_correct_valid":   1.0,
    "reward_correct_invalid": 0.6,
    "reward_valid_search":    0.3,
    "reward_valid_only":      0.2,
    "reward_attempt_only":    0.1,
}

_WALK_EXTEND_ENABLED = _DEFAULT_CONFIG.walk_extend_enabled
_WALK_PREVIEW_TOKENS = _DEFAULT_CONFIG.walk_preview_tokens
_WALK_NODE_CONTENT_TOKENS = _DEFAULT_CONFIG.walk_node_content_tokens
_WALK_NEIGHBOR_LIMIT = _DEFAULT_CONFIG.walk_neighbor_limit
_WALK_NEIGHBOR_SAMPLING = _DEFAULT_CONFIG.walk_neighbor_sampling
_FORCE_MIN_SEARCH = _DEFAULT_CONFIG.force_min_search
_POST_SEARCH_NUDGE = _DEFAULT_CONFIG.post_search_nudge

_STATE_LOCK = threading.Lock()
_VIEWS: dict[tuple[str, str], Any] = {}

def _get_view(dataset_name: str, data_path: str):
    
    from pathlib import Path
    from walker.tag.graph_view import GraphView

    cache_key = (dataset_name, str(Path(data_path).expanduser().resolve()))
    if cache_key in _VIEWS:
        return _VIEWS[cache_key]

    with _STATE_LOCK:
        if cache_key in _VIEWS:
            return _VIEWS[cache_key]
        _VIEWS[cache_key] = GraphView.from_path(dataset_name, data_path)

    return _VIEWS[cache_key]

def _fetch_node_text(node_id: int, dataset_name: str, data_path: str) -> str:
    
    return _get_view(dataset_name, data_path).node_text(node_id)

def _subsample_neighbors(
    node_id: int,
    neighbor_ids: list[int],
    cfg: GraphEnvConfig,
) -> list[int]:
    
    limit = cfg.walk_neighbor_limit
    if limit < 0 or len(neighbor_ids) <= limit:
        return neighbor_ids
    strategy = cfg.walk_neighbor_sampling
    if strategy == "first":
        return sorted(neighbor_ids)[:limit]
    if strategy == "random":
        import random as _r
        return _r.sample(neighbor_ids, limit)

    import random as _r
    rng = _r.Random(int(node_id))
    return rng.sample(neighbor_ids, limit)

async def _fetch_node_neighbors_preview(
    node_id: int,
    dataset_name: str,
    data_path: str,
    cfg: GraphEnvConfig,
    summarizer: Any = None,
) -> tuple[list[int], list[dict]]:
    
    view = _get_view(dataset_name, data_path)
    neighbor_ids = _subsample_neighbors(int(node_id), view.neighbors(node_id), cfg)

    if cfg.summary_enabled and summarizer is not None:
        neighbor_fulls = [view.node_text(m) for m in neighbor_ids]
        summaries = await summarizer.summarize_batch(neighbor_fulls)
        previews = [{"id": m, "text": s} for m, s in zip(neighbor_ids, summaries)]
        return neighbor_ids, previews

    from walker.tag.graph_view import GraphView
    previews = [
        {"id": m, "text": GraphView.truncate(view.node_text(m), cfg.walk_preview_tokens)}
        for m in neighbor_ids
    ]
    return neighbor_ids, previews

def _format_walk_extension(
    node_id: int,
    node_text: str,
    new_neighbor_previews: list[dict],
    accumulated_valid_ids: set[int],
    cfg: GraphEnvConfig,
) -> str:
    
    if cfg.walk_node_content_tokens > 0:
        
        from walker.tag.graph_view import GraphView
        body = GraphView.truncate(node_text, cfg.walk_node_content_tokens)
    else:
        body = node_text

    parts = [f"Node {node_id}: {body}"]
    if new_neighbor_previews:
        parts.append(
            f"\n\nNode {node_id}'s neighbors "
            f"(short preview — use <walk> to read full content):"
        )
        for nb in new_neighbor_previews:
            parts.append(f"\n- [{nb['id']}]: {nb['text']}")
    if accumulated_valid_ids:
        sorted_ids = ", ".join(str(i) for i in sorted(accumulated_valid_ids))
        parts.append(f"\n\nAvailable node IDs: {sorted_ids}")
    nudge = f"{cfg.post_search_nudge}\n\n" if cfg.post_search_nudge else ""
    return f"\n\n<information>{''.join(parts)}</information>\n\n{nudge}"

def _normalize_label(raw_label: Any) -> dict[str, Any]:
    
    if isinstance(raw_label, dict):
        return raw_label
    if isinstance(raw_label, str):
        try:
            parsed = json.loads(raw_label)
        except (ValueError, TypeError):
            parsed = raw_label
        if isinstance(parsed, dict):
            return parsed
        try:
            return {"ground_truth": int(parsed)}
        except (ValueError, TypeError):
            return {}
    if isinstance(raw_label, int):
        return {"ground_truth": int(raw_label)}
    return {}

def _normalize_search_id(
    raw_search: str,
    valid_node_ids: set[int],
    *,
    strict: bool | None = None,
) -> int | None:
    
    if strict is None:
        strict = _DEFAULT_CONFIG.strict_search_format
    if strict:
        if not re.fullmatch(r"\s*\d+\s*", raw_search):
            return None
        node_id = int(raw_search.strip())
    else:
        id_match = re.search(r"\d+", raw_search)
        if id_match is None:
            return None
        node_id = int(id_match.group())
    if valid_node_ids and node_id not in valid_node_ids:
        return None
    return node_id

def _extract_first_valid_search_id(
    response: str,
    valid_node_ids: set[int],
    *,
    strict: bool | None = None,
) -> int | None:
    for match in re.finditer(r"<walk>(.*?)</walk>", response, re.DOTALL):
        node_id = _normalize_search_id(match.group(1), valid_node_ids, strict=strict)
        if node_id is not None:
            return node_id
    return None

def postprocess_predictions(prediction: str) -> tuple[str | None, str]:
    
    pattern = r"<(walk|answer)>(.*?)</\1>"
    matches = list(re.finditer(pattern, prediction, re.DOTALL))
    if not matches:
        return None, ""
    walk_match = next((m for m in matches if m.group(1) == "walk"), None)
    if walk_match:
        return "search", walk_match.group(2).strip()
    first = matches[0]
    return first.group(1), first.group(2).strip()

async def execute_predictions(
    prediction: str,
    valid_node_ids: set[int],
    dataset_name: str,
    data_path: str,
    cfg: GraphEnvConfig | None = None,
    summarizer: Any = None,
) -> tuple[str, bool, set[int], "WalkEgoSnapshot | None"]:
    
    if cfg is None:
        cfg = _DEFAULT_CONFIG
    action, content = postprocess_predictions(prediction)

    _example_id = (sorted(valid_node_ids)[0] if valid_node_ids else 1943)

    if action == "search":
        if cfg.strict_search_format:

            if not re.fullmatch(r"\s*\d+\s*", content):
                return (
                    f"\nThe content inside <walk>...</walk> must be ONLY an "
                    f"integer node ID — no words, prefixes, or surrounding text. "
                    f"For example: <walk>{_example_id}</walk>. Let me try again.\n",
                    False,
                    set(),
                    None,
                )
            node_id = int(content.strip())
        else:

            id_match = re.search(r"\d+", content)
            if not id_match:
                return (
                    f"\nThe node ID must be an integer (NOT the literal text "
                    f"'NODE_ID' or 'INTEGER_ID' — those are placeholders). "
                    f"Use <walk>{_example_id}</walk> with a real integer "
                    f"node ID from the available list. Let me try again.\n",
                    False,
                    set(),
                    None,
                )
            node_id = int(id_match.group())

        if valid_node_ids and node_id not in valid_node_ids:
            _id_preview = ", ".join(str(i) for i in sorted(valid_node_ids)[:5])
            return (
                f"\nNode {node_id} is not a neighbor of the target node. "
                f"Please choose a node ID from the neighbors listed in the prompt "
                f"(e.g., {_id_preview}). For example: <walk>{_example_id}</walk>. "
                f"Let me try again.\n",
                False,
                set(),
                None,
            )

        node_text = _fetch_node_text(node_id, dataset_name, data_path)

        if not cfg.walk_extend_enabled:
            nudge = f"{cfg.post_search_nudge}\n\n" if cfg.post_search_nudge else ""
            next_obs = f"\n\n<information>Node {node_id}: {node_text}</information>\n\n{nudge}"
            return next_obs, False, set(), None

        new_neighbor_ids, new_previews = await _fetch_node_neighbors_preview(
            node_id, dataset_name, data_path, cfg, summarizer=summarizer,
        )

        accumulated = set(valid_node_ids) | set(new_neighbor_ids)
        accumulated.discard(node_id)
        next_obs = _format_walk_extension(
            node_id=node_id,
            node_text=node_text,
            new_neighbor_previews=new_previews,
            accumulated_valid_ids=accumulated,
            cfg=cfg,
        )
        view = _get_view(dataset_name, data_path)
        dest_label = view.node_label(int(node_id))
        dest_label_name = (
            view.tag.label_names[dest_label]
            if dest_label is not None and getattr(view.tag, "label_names", None)
            else None
        )
        snapshot = WalkEgoSnapshot(
            node_id=int(node_id),
            content=node_text,
            neighbor_ids=tuple(int(nid) for nid in new_neighbor_ids),
            neighbor_previews=tuple(str(p["text"]) for p in new_previews),
            label=dest_label,
            label_name=dest_label_name,
        )
        return next_obs, False, set(new_neighbor_ids), snapshot

    elif action == "answer":
        return "", True, set(), None

    else:
        return (
            f"\nMy previous action is invalid. "
            f"To look up a neighbor's full content, output <walk>NODE_ID</walk> "
            f"with NODE_ID replaced by an integer from the available list — "
            f"for example <walk>{_example_id}</walk>. "
            f"To give the final classification, output <answer>CATEGORY_ID</answer> "
            f"with CATEGORY_ID replaced by an integer category index — for example "
            f"<answer>0</answer>. "
            f"Do NOT output the literal strings 'node_id', 'NODE_ID', or 'label'. "
            f"Let me try again.\n",
            False,
            set(),
            None,
        )

@dataclass(frozen=True)
class WalkEgoSnapshot:

    node_id: int
    content: str
    neighbor_ids: tuple[int, ...]
    neighbor_previews: tuple[str, ...]
    label: int | None = None
    label_name: str | None = None

@dataclass
class GraphEnvStep:

    obs_text: str
    done: bool
    searched_node_id: int | None
    successful_search: bool
    new_valid_ids: set[int]
    rejected_reason: str | None = None
    walk_ego: WalkEgoSnapshot | None = None

class GraphEnv(BaseInteractionEnv):

    def __init__(
        self,
        *,
        dataset_name: str,
        data_path: str,
        cfg: GraphEnvConfig | None = None,
        summarizer: Any = None,
    ) -> None:
        self.dataset_name = str(dataset_name)
        self.data_path = str(data_path)

        self.cfg: GraphEnvConfig = cfg if cfg is not None else _DEFAULT_CONFIG
        self.summarizer = summarizer

        self.valid_node_ids: set[int] = set()
        self.walk_path: list[int] = []
        self.successful_search_count: int = 0
        self.turns_used: int = 0
        self.answered: bool = False

        self.force_min_triggers: int = 0

        self.non_canonical_search_attempts: int = 0

    @classmethod
    def from_args(
        cls,
        args: Any,
        sample: Any,
        *,
        cfg: GraphEnvConfig | None = None,
    ) -> "GraphEnv":
        
        label = _normalize_label(getattr(sample, "label", None))
        dataset_name = label.get("dataset_name") or get_env("WALKER_DATASET_NAME", default="cora")
        data_path = get_env("WALKER_DATA_PATH", default="data/raw_datasets")
        return cls(
            dataset_name=str(dataset_name),
            data_path=str(data_path),
            cfg=cfg,
        )

    @property
    def max_turns(self) -> int:
        return int(self.cfg.max_turns)

    @property
    def max_searches(self) -> int:
        return int(self.cfg.max_searches)

    @property
    def enforce_search_budget(self) -> bool:
        return bool(self.cfg.enforce_search_budget)

    @property
    def force_min_search(self) -> int:
        return int(self.cfg.force_min_search)

    @property
    def walk_extend_enabled(self) -> bool:
        return bool(self.cfg.walk_extend_enabled)

    @property
    def graph_state(self) -> Any:
        
        return _get_view(self.dataset_name, self.data_path)

    def reset(self, sample: Any) -> tuple[dict, dict]:  
        
        label = _normalize_label(getattr(sample, "label", None))
        target_neighbor_ids: set[int] = {int(x) for x in label.get("neighbor_ids", [])}

        if not target_neighbor_ids:
            prompt = getattr(sample, "prompt", None)
            if isinstance(prompt, str):
                prompt_match = re.search(
                    r"Initial allowed node IDs:\s*([\d,\s]+)",
                    prompt,
                )
                if prompt_match:
                    target_neighbor_ids = {
                        int(x) for x in re.findall(r"\d+", prompt_match.group(1))
                    }

        self.valid_node_ids = set(target_neighbor_ids)
        self.walk_path = []
        self.successful_search_count = 0
        self.turns_used = 0
        self.answered = False
        self.force_min_triggers = 0
        self.non_canonical_search_attempts = 0
        return {}, {"target_neighbor_ids": sorted(target_neighbor_ids)}

    def close(self) -> None:  
        
        return

    async def step(self, response_text: str) -> GraphEnvStep:  
        
        self.turns_used += 1

        next_obs, done, new_valid_ids, walk_ego = await execute_predictions(
            response_text,
            self.valid_node_ids,
            self.dataset_name,
            self.data_path,
            self.cfg,
            summarizer=self.summarizer,
        )

        if self.cfg.strict_search_format:
            for m in re.finditer(r"<walk>(.*?)</walk>", response_text, re.DOTALL):
                if not re.fullmatch(r"\s*\d+\s*", m.group(1)):
                    self.non_canonical_search_attempts += 1

        searched_match = re.search(r"<walk>\s*(\d+)\s*</walk>", response_text)
        searched_node_id_raw = int(searched_match.group(1)) if searched_match is not None else None

        search_in_obs = "<information>" in next_obs
        rejected_reason: str | None = None

        if (
            self.enforce_search_budget
            and search_in_obs
            and self.successful_search_count >= self.max_searches
        ):
            next_obs = (
                f"\n\n<information>You have used all "
                f"{self.max_searches} search action(s) "
                f"allowed. No further <walk> will be accepted; please "
                f"emit <answer>X</answer> with an integer category index "
                f"on your next turn.</information>\n\n"
            )
            search_in_obs = False
            new_valid_ids = set()
            done = False
            rejected_reason = "over_budget"

        if search_in_obs:
            self.successful_search_count += 1

        if (
            done
            and self.force_min_search > 0
            and self.successful_search_count < self.force_min_search
        ):
            done = False
            rejected_reason = "force_min"
            self.force_min_triggers += 1
            _force_example = (sorted(self.valid_node_ids)[0] if self.valid_node_ids else 1943)
            next_obs = (
                f"\nYou must complete at least {self.force_min_search} successful "
                f"walk action(s) (a real one looks like "
                f"<walk>{_force_example}</walk>, with the actual integer "
                f"substituted from the available list) before producing the "
                f"final <answer>. You have completed {self.successful_search_count} "
                f"so far. Please emit another walk action now using a real "
                f"integer node ID — for example <walk>{_force_example}</walk>.\n"
            )

        if search_in_obs and new_valid_ids:
            self.valid_node_ids |= new_valid_ids

        if done:
            self.answered = True

        return GraphEnvStep(
            obs_text=next_obs,
            done=done,
            searched_node_id=searched_node_id_raw,
            successful_search=search_in_obs,
            new_valid_ids=set(new_valid_ids) if search_in_obs else set(),
            rejected_reason=rejected_reason,
            walk_ego=walk_ego if (search_in_obs and new_valid_ids) else None,
        )

    def record_walk_step(self, node_id: int) -> None:
        
        self.walk_path.append(int(node_id))
