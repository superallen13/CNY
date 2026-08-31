from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from walker._data import eval_jsonl, fetch_hint, graph_pt, train_jsonl
from walker.config import ExpConfig, load_exp, render_resolved

def _resolve_teacher_url(cfg: ExpConfig) -> str | None:
    
    if not cfg.enable_opd:
        return None
    if cfg.teacher_reuse_student:
        return "auto-router"
    return f"http://127.0.0.1:{cfg.teacher_port}/generate"

def build_slime_args(
    cfg: ExpConfig,
    *,
    prompt_data: str,
    hf_checkpoint: str,
    ref_load: str,
    megatron_to_hf_mode: str,
    model_arch_args: list[str],
    eval_prompt_data: list[tuple[str, str]] | None = None,
    teacher_url: str | None = None,
) -> list[str]:
    
    args: list[str] = []
    args += model_arch_args  

    args += [
        "--megatron-to-hf-mode", megatron_to_hf_mode,
        "--hf-checkpoint", hf_checkpoint,
        "--ref-load", ref_load,
        "--save", "outputs/walker/ckpt",
        "--save-interval", str(cfg.save_interval),
    ]

    args += [
        "--prompt-data", prompt_data,
        "--input-key", "messages",
        "--label-key", "solution",
        "--apply-chat-template",
        "--rollout-shuffle",
        "--num-rollout", str(cfg.num_rollout),
        "--rollout-batch-size", str(cfg.rollout_batch_size),
        "--n-samples-per-prompt", str(cfg.n_samples_per_prompt),
        "--rollout-max-response-len", str(cfg.rollout_max_response_len),
        "--rollout-temperature", str(cfg.rollout_temperature),
        "--global-batch-size", str(cfg.global_batch_size),
        "--balance-data",
    ]
    if cfg.apply_chat_template_kwargs:
        args += [
            "--apply-chat-template-kwargs",
            json.dumps(cfg.apply_chat_template_kwargs),
        ]

    args += [
        "--tensor-model-parallel-size", str(cfg.tensor_parallel_size),
        "--pipeline-model-parallel-size", str(cfg.pipeline_parallel_size),
        "--use-dynamic-batch-size",
        "--max-tokens-per-gpu", str(cfg.max_tokens_per_gpu),
        "--sequence-parallel",
        "--log-probs-chunk-size", str(cfg.log_probs_chunk_size),
        "--cross-entropy-loss-fusion",
        "--cross-entropy-fusion-impl", "te",
    ]
    if cfg.use_recompute:
        args += [
            "--recompute-granularity", "full",
            "--recompute-method", "uniform",
            "--recompute-num-layers", "1",
        ]

    args += [
        "--advantage-estimator", "grpo",
        "--eps-clip", "0.2",
        "--eps-clip-high", "0.28",
    ]
    if cfg.enable_opd:
        args += [
            "--enable-opd",
            "--use-opd",
            "--opd-type", "sglang",
            "--opd-kl-coef", str(cfg.opd_kl_coef),
        ]
        if teacher_url is not None:
            args += [
                "--opd-teacher-url", teacher_url,
                
                "--custom-megatron-before-train-step-hook-path",
                "walker.train.hooks.teacher_update_hook",
            ]

    args += [
        "--optimizer", "adam",
        "--lr", str(cfg.lr),
        "--lr-decay-style", "constant",
        "--weight-decay", "0.01",
        "--adam-beta1", "0.9",
        "--adam-beta2", "0.98",
    ]
    if cfg.precision_aware_adam:
        args += ["--use-precision-aware-optimizer"]
    if cfg.use_cpu_offload:

        args += [
            "--no-use-megatron-fsdp",
            "--use-distributed-optimizer",
            "--optimizer-cpu-offload",
            "--optimizer-offload-fraction", "1.0",
            "--overlap-cpu-optimizer-d2h-h2d",
            "--use-torch-optimizer-for-cpu-offload",
            "--exp-avg-dtype", "bf16",
            "--exp-avg-sq-dtype", "bf16",
            "--main-grads-dtype", "bf16",
        ]

    if cfg.rollout_num_gpus is not None:
        args += ["--rollout-num-gpus", str(cfg.rollout_num_gpus)]
    args += [
        "--rollout-num-gpus-per-engine", str(cfg.rollout_num_gpus_per_engine),
        "--sglang-mem-fraction-static", str(cfg.sglang_mem),
        "--sglang-context-length", str(cfg.sglang_context_length),
        "--sglang-max-running-requests", str(cfg.sglang_max_running_requests),
        
        "--router-disable-circuit-breaker",
    ]
    if cfg.sglang_disable_cuda_graph:
        args += ["--sglang-disable-cuda-graph"]
    if cfg.sglang_chunked_prefill_size is not None:
        args += ["--sglang-chunked-prefill-size", str(cfg.sglang_chunked_prefill_size)]
    if cfg.sglang_attention_backend is not None:
        args += ["--sglang-attention-backend", cfg.sglang_attention_backend]
    if cfg.sglang_deterministic:
        args += ["--sglang-enable-deterministic-inference"]

    args += [
        "--attention-dropout", "0.0",
        "--hidden-dropout", "0.0",
        "--attention-softmax-in-fp32",
        "--attention-backend", "flash",
    ]

    if not cfg.use_cpu_offload:
        args += ["--accumulate-allreduce-grads-in-fp32"]
    if not cfg.eval_before_train:
        args += ["--skip-eval-before-train"]
    if cfg.debug_rollout_only:
        args += ["--debug-rollout-only"]
    if cfg.eval_interval != 999999:
        args += ["--eval-interval", str(cfg.eval_interval)]
    if eval_prompt_data:
        flat = [s for pair in eval_prompt_data for s in pair]
        args += ["--eval-prompt-data", *flat]

    args += ["--colocate"]
    args += [
        "--offload-train",
        "--train-memory-margin-bytes", str(cfg.train_memory_margin_bytes),
    ]

    if cfg.use_wandb:
        args += ["--use-wandb", "--wandb-project", cfg.wandb_project]
        if cfg.wandb_group:
            args += ["--wandb-group", cfg.wandb_group]

    return args

def to_walker_env(cfg: ExpConfig) -> dict[str, str]:
    
    env: dict[str, str] = {
        "WALKER_FORCE_MIN_SEARCH": str(cfg.walker_force_min_search),
        "WALKER_MAX_HOPS": str(cfg.walker_max_hops),
        "WALKER_NODE_CONTENT_TOKENS": str(cfg.walker_node_content_tokens),
        "WALKER_PREVIEW_TOKENS": str(cfg.walker_preview_tokens),
        "WALKER_NEIGHBOR_LIMIT": str(cfg.walker_neighbor_limit),
        "WALKER_RETURN_LOGPROB": "1" if cfg.walker_return_logprob else "0",
        "WALKER_DUMP_OPD_CSV": "1" if cfg.walker_dump_opd_csv else "0",
        "WALKER_DUMP_PER_TOKEN": "1" if cfg.walker_dump_per_token else "0",
        "WALKER_DUMP_CELL_ID": cfg.walker_dump_cell_id,
        "WALKER_WALK_EXTEND": "1" if cfg.walker_walk_extend else "0",
        "WALKER_TEACHER_SLEEP": "1" if cfg.walker_teacher_sleep else "0",
        "WALKER_HINT_TEMPLATE": str(cfg.walker_hint_template),
    }
    if cfg.eval_subsample is not None:
        env["WALKER_EVAL_SUBSAMPLE_N"] = str(cfg.eval_subsample)
    if cfg.balance_to_dataset:
        env["BALANCE_TO_DATASET"] = cfg.balance_to_dataset
    
    env["WANDB_LOG_SAMPLE_TABLE"] = "1" if cfg.wandb_log_sample_table else "0"
    env["WANDB_SAMPLE_TABLE_INTERVAL"] = str(cfg.wandb_sample_table_interval)
    env["WANDB_SAMPLE_TABLE_MAX_ROWS"] = str(cfg.wandb_sample_table_max_rows)
    if cfg.wandb_tags:
        env["WANDB_TAGS"] = cfg.wandb_tags
    if cfg.wandb_notes:
        env["WANDB_NOTES"] = cfg.wandb_notes
    return env

def require_data(cfg: ExpConfig) -> None:
    missing: list[Path] = []
    if not train_jsonl().exists():
        missing.append(train_jsonl())
    for ds in cfg.eval_datasets:
        if not eval_jsonl(ds).exists():
            missing.append(eval_jsonl(ds))
    for ds in dict.fromkeys([*cfg.train_datasets, *cfg.eval_datasets]):
        if not graph_pt(ds).exists():
            missing.append(graph_pt(ds))
    if missing:
        raise FileNotFoundError(fetch_hint(missing))

def combine_train_jsonl(cfg: ExpConfig) -> Path:
    p = train_jsonl()
    if not p.exists():
        raise FileNotFoundError(fetch_hint([p]))
    return p

def find_eval_paths(cfg: ExpConfig) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    missing: list[Path] = []
    for ds in cfg.eval_datasets:
        p = eval_jsonl(ds)
        if not p.exists():
            missing.append(p)
            continue
        out.append((ds, str(p)))
    if missing:
        raise FileNotFoundError(fetch_hint(missing))
    return out

def _resolve_paths(cfg: ExpConfig) -> tuple[str, str, str]:
    
    in_sif = os.environ.get("WALKER_IN_SIF", "") == "1"
    if in_sif:
        hf_ckpt = cfg.hf_local_dir_in_sif or cfg.hf_local_dir
    else:
        hf_ckpt = cfg.hf_local_dir
    ref_load = hf_ckpt
    mode = "bridge"
    return hf_ckpt, ref_load, mode

def _source_model_arch(arch_path: str) -> list[str]:
    
    script = (
        f"source '{arch_path}'; "
        r'printf "%s\0" "${MODEL_ARGS[@]}"'
    )
    out = subprocess.check_output(["bash", "-c", script])
    parts = out.rstrip(b"\0").split(b"\0")
    return [p.decode() for p in parts if p]

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Walker experiment driver.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--exp", required=True, help="Path to configs/exp/<name>.yaml")
    ap.add_argument("--set", action="append", default=[],
                    metavar="KEY=VAL", help="Override a config value (can repeat)")
    ap.add_argument("--show", action="store_true",
                    help="Print resolved config and exit")
    ap.add_argument("--print-slime-args", action="store_true",
                    help="Print resolved slime CLI args (newline-separated) and exit")
    ap.add_argument("--dry-args", action="store_true",
                    help="Like --print-slime-args but skip filesystem deps "
                         "(require_data, combine_train_jsonl, find_eval_paths). "
                         "Uses placeholder paths; for parity-verification only.")
    ap.add_argument("--print-walker-env", action="store_true",
                    help="Print walker_env as 'export KEY=VAL' lines and exit")
    ap.add_argument("--no-data-check", action="store_true",
                    help="Skip the data/ presence check before launching")
    ap.add_argument("--no-train", action="store_true",
                    help="Resolve config + data paths but do not invoke slime")
    args = ap.parse_args(argv)

    if args.show:
        print(render_resolved(args.exp))
        return 0

    cfg = load_exp(args.exp, overrides=args.set)

    if args.print_walker_env:
        for k, v in to_walker_env(cfg).items():
            print(f"export {k}={v!r}")
        return 0

    if args.dry_args:

        model_arch_args = _source_model_arch(cfg.model_arch)
        if cfg.override_rotary_base is not None:
            model_arch_args += ["--rotary-base", str(cfg.override_rotary_base)]
        hf_ckpt, ref_load, mode = _resolve_paths(cfg)
        teacher_url = (
            f"http://127.0.0.1:{cfg.teacher_port}/generate" if cfg.enable_opd else None
        )
        slime_args = build_slime_args(
            cfg,
            prompt_data="<PROMPT_DATA>",
            hf_checkpoint=hf_ckpt or "<HF_LOCAL_DIR_IN_SIF>",
            ref_load=ref_load or "<REF_LOAD>",
            megatron_to_hf_mode=mode,
            model_arch_args=model_arch_args,
            eval_prompt_data=None,
            teacher_url=teacher_url,
        )
        for a in slime_args:
            print(a)
        return 0

    if not args.no_data_check:
        require_data(cfg)

    train_jsonl = combine_train_jsonl(cfg)
    eval_pairs = find_eval_paths(cfg) if not cfg.enable_opd else None

    hf_ckpt, ref_load, mode = _resolve_paths(cfg)
    teacher_url = _resolve_teacher_url(cfg)
    model_arch_args = _source_model_arch(cfg.model_arch)
    if cfg.override_rotary_base is not None:
        model_arch_args += ["--rotary-base", str(cfg.override_rotary_base)]

    slime_args = build_slime_args(
        cfg,
        prompt_data=str(train_jsonl),
        hf_checkpoint=hf_ckpt,
        ref_load=ref_load,
        megatron_to_hf_mode=mode,
        model_arch_args=model_arch_args,
        eval_prompt_data=eval_pairs,
        teacher_url=teacher_url,
    )

    if args.print_slime_args:
        for a in slime_args:
            print(a)
        return 0

    if args.no_train:
        print("[walker.run] --no-train set; build done, skipping slime invocation.",
              file=sys.stderr)
        return 0

    return _invoke_slime(cfg, slime_args)

def _invoke_slime(cfg: ExpConfig, slime_args: list[str]) -> int:

    os.environ.update(to_walker_env(cfg))

    cluster_num_nodes = cfg.cluster_num_nodes or 1
    cmd = [
        sys.executable, "-m", "walker.train.walker",
        "--actor-num-nodes", str(cluster_num_nodes),
        "--actor-num-gpus-per-node", os.environ.get("NUM_GPUS", "1"),
        *slime_args,
    ]
    print(f"[walker.run] launching walker.train.walker ({len(slime_args)} slime args)",
          file=sys.stderr)
    return subprocess.run(cmd).returncode

if __name__ == "__main__":
    sys.exit(main())
