from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path

from walker._data import eval_jsonl, fetch_hint, graph_pt
from walker.eval.benchmark import main as benchmark_main
from walker.eval.config import EvalConfig, load_eval

def to_walker_env(cfg: EvalConfig) -> dict[str, str]:
    
    env = {
        "WALKER_FORCE_MIN_SEARCH": str(cfg.walker_force_min_search),
        "WALKER_MAX_HOPS": str(cfg.walker_max_hops),
        "WALKER_NODE_CONTENT_TOKENS": str(cfg.walker_node_content_tokens),
        "WALKER_PREVIEW_TOKENS": str(cfg.walker_preview_tokens),
        "WALKER_NEIGHBOR_LIMIT": str(cfg.walker_neighbor_limit),
        "WALKER_WALK_EXTEND": "1" if cfg.walker_walk_extend else "0",
        
        "WALKER_RETURN_LOGPROB": "0",
    }
    if cfg.eval_subsample is not None:
        env["WALKER_EVAL_SUBSAMPLE_N"] = str(cfg.eval_subsample)
    if cfg.walker_summary_enabled:
        env["WALKER_SUMMARY_ENABLED"] = "1"
        env["WALKER_SUMMARY_MAX_TOKENS"] = str(cfg.walker_summary_max_tokens)
        env["WALKER_SUMMARY_TEMPERATURE"] = str(cfg.walker_summary_temperature)
        if cfg.walker_summary_template:
            env["WALKER_SUMMARY_TEMPLATE"] = cfg.walker_summary_template
    return env

def find_eval_jsonl(cfg: EvalConfig) -> Path:
    prompts = eval_jsonl(cfg.eval_dataset)
    missing = [p for p in (prompts, graph_pt(cfg.eval_dataset)) if not p.exists()]
    if missing:
        raise FileNotFoundError(fetch_hint(missing))
    return prompts

def build_benchmark_argv(cfg: EvalConfig, prompt_path: Path) -> list[str]:
    
    argv = [
        "--prompt-data", str(prompt_path),
        "--base-url", cfg.backend_url,
        "--model", cfg.model,
        "--max-in-flight", str(cfg.max_in_flight),
        "--max-tokens", str(cfg.max_tokens),
        "--temperature", str(cfg.temperature),
    ]
    if cfg.api_key_env:
        argv += ["--api-key-env", cfg.api_key_env]
    if cfg.no_think:
        argv += ["--no-think"]
    if cfg.use_wandb:
        argv += ["--wandb",
                 "--wandb-project", cfg.wandb_project]
        if cfg.wandb_group:
            argv += ["--wandb-group", cfg.wandb_group]
        if cfg.wandb_notes:
            argv += ["--wandb-notes", cfg.wandb_notes]
        if cfg.wandb_tags:
            argv += ["--wandb-tags", cfg.wandb_tags]
    return argv

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Walker eval driver: resolve config, run benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--exp", required=True, help="Path to configs/eval/<name>.yaml")
    ap.add_argument("--set", action="append", default=[],
                    metavar="KEY=VAL", help="Override a config value (can repeat)")
    ap.add_argument("--show", action="store_true",
                    help="Print resolved config and exit")
    ap.add_argument("--no-bench", action="store_true",
                    help="Resolve config + data paths but do not invoke benchmark")
    args = ap.parse_args(argv)

    cfg = load_eval(args.exp, overrides=args.set)

    if args.show:
        print(f"=== Effective eval config: {args.exp} ===")
        for k, v in sorted(asdict(cfg).items()):
            print(f"  {k:32s} = {v!r}")
        return 0

    prompt_path = find_eval_jsonl(cfg)
    print(f"[eval.run] eval prompt jsonl: {prompt_path}", file=sys.stderr)

    os.environ.update(to_walker_env(cfg))

    if args.no_bench:
        print("[eval.run] --no-bench set; skipping benchmark.", file=sys.stderr)
        return 0

    bench_argv = build_benchmark_argv(cfg, prompt_path)
    return benchmark_main(bench_argv)

if __name__ == "__main__":
    sys.exit(main())
