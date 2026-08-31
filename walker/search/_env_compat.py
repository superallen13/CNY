from __future__ import annotations

import os
from typing import Optional

def get_env(canonical: str, default: Optional[str] = None) -> Optional[str]:
    
    return os.environ.get(canonical, default)
