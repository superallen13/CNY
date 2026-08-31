from __future__ import annotations

import argparse
import glob
import os
import sys

def _parse_walker_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--enable-opd",
        action="store_true",
        help="Use OPD path instead of slime's default GRPO/PPO.",
    )
    parser.add_argument(
        "--opd-teacher-url",
        type=str,
        default=None,
        help="URL of the frozen-teacher SGLang /generate endpoint. Read by walker.search.rollout when method=opd.",
    )
    return parser.parse_known_args(argv)

def _install_walker_defaults(slime_argv: list[str]) -> list[str]:
    
    defaults = [
        ("--custom-generate-function-path", "walker.train.rollout_adapter.generate"),
        ("--custom-rm-path", "walker.search.rollout.reward_func"),
        ("--custom-rollout-log-function-path", "walker.train.rollout_adapter.log_rollout_data"),
        ("--custom-eval-rollout-log-function-path", "walker.train.rollout_adapter.log_eval_rollout_data"),
    ]
    out = list(slime_argv)
    for flag, default in defaults:
        if flag not in slime_argv:
            out.extend([flag, default])
    return out

def _install_eval_args(slime_argv: list[str]) -> list[str]:
    
    out = list(slime_argv)
    if "--eval-prompt-data" in out:
        return out

    env_datasets = os.environ.get("EVAL_DATASETS", "").strip()
    if not env_datasets:
        return out

    sub_n = os.environ.get("WALKER_EVAL_SUBSAMPLE_N", "").strip()
    flat: list[str] = []
    for name in env_datasets.split():
        candidates: list[str] = []
        if sub_n:
            candidates = sorted(
                glob.glob(f"data/cache/{name}/*/test_sub{sub_n}.jsonl"),
                key=os.path.getmtime,
            )
        if not candidates:
            candidates = sorted(
                glob.glob(f"data/cache/{name}/*/test.jsonl"), key=os.path.getmtime
            )
        if not candidates:
            continue
        flat.extend([name, candidates[-1]])
    if flat:
        out.extend(["--eval-prompt-data", *flat])
    return out

def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    walker_args, slime_argv = _parse_walker_args(argv)
    slime_argv = _install_walker_defaults(slime_argv)

    if walker_args.enable_opd:
        slime_argv = _install_eval_args(slime_argv)

    sys.argv = [sys.argv[0]] + slime_argv

    from train import train as slime_train  
    from slime.utils.arguments import parse_args

    args = parse_args()

    if walker_args.enable_opd:

        args.method = "opd"
        args.enable_opd = True
        if walker_args.opd_teacher_url is not None:
            args.opd_teacher_url = walker_args.opd_teacher_url

    slime_train(args)

if __name__ == "__main__":
    main()
