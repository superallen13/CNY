from __future__ import annotations

import os
from typing import Any

def teacher_update_hook(
    args: Any,
    rollout_id: int,  
    step_id: int,  
    model: Any,  
    optimizer: Any,  
    opt_param_scheduler: Any,  
) -> None:
    
    if getattr(args, "method", "opd") == "sft":
        return

    if os.environ.get("WALKER_TEACHER_SLEEP", "0") != "1":
        return

    teacher_url = getattr(args, "opd_teacher_url", None)
    if not teacher_url:
        return

    if teacher_url == "auto-router":
        return

    try:
        import requests

        from walker.opd.teacher_sleep_state import mark_released

        base = teacher_url.rsplit("/", 1)[0] if teacher_url.endswith("/generate") else teacher_url
        resp = requests.post(
            f"{base}/release_memory_occupation",

            json={"tags": ["weights"]},
            timeout=10,
        )
        if resp.status_code == 200:
            mark_released(teacher_url)
    except Exception:  
        pass
