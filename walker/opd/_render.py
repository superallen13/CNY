from __future__ import annotations

from pathlib import Path
from typing import Any

import jinja2

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_ENV = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(_TEMPLATES_DIR)),
    autoescape=False,
    keep_trailing_newline=True,
)

def render_template(filename: str, /, **ctx: Any) -> str:
    
    template = _ENV.get_template(filename)
    return template.render(**ctx)
