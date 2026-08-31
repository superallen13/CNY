from __future__ import annotations

import fcntl
import threading
from pathlib import Path
from urllib.parse import urlparse

_STATE_RELEASED = "released"
_STATE_ACTIVE = "active"

_inproc_lock = threading.Lock()

def _state_path(teacher_url: str) -> Path:
    parsed = urlparse(teacher_url)
    port = parsed.port or 0
    return Path(f"/tmp/walker_teacher_sleep_{port}.state")

def mark_released(teacher_url: str) -> None:
    
    _state_path(teacher_url).write_text(_STATE_RELEASED)

def claim_resume(teacher_url: str) -> bool:
    
    p = _state_path(teacher_url)
    if not p.exists():
        return False
    with _inproc_lock:
        with p.open("r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                state = f.read().strip()
                if state != _STATE_RELEASED:
                    return False
                f.seek(0)
                f.truncate()
                f.write(_STATE_ACTIVE)
                f.flush()
                return True
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
