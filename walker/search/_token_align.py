from __future__ import annotations

import re

_SEARCH_TAG_RE = re.compile(r"<walk>\s*\d+\s*</walk>")

def find_search_action_span(
    tokenizer, token_ids: list[int]
) -> tuple[int, int] | None:
    
    if not token_ids:
        return None

    canonical_text = tokenizer.decode(token_ids, skip_special_tokens=False)
    m = _SEARCH_TAG_RE.search(canonical_text)
    if m is None:
        return None
    cs, ce = m.start(), m.end()

    enc = tokenizer(canonical_text, add_special_tokens=False, return_offsets_mapping=True)
    if enc["input_ids"] == list(token_ids):
        offsets = enc["offset_mapping"]
        start = next((i for i, (_, e) in enumerate(offsets) if e > cs), None)
        end = next((i for i, (s, _) in enumerate(offsets) if s >= ce), len(offsets))
        if start is None or start >= end:
            return None
        return (start, end)

    cum: list[int] = [0]
    text = ""
    for i in range(1, len(token_ids) + 1):
        text = tokenizer.decode(token_ids[:i], skip_special_tokens=False)
        cum.append(len(text))

    m = _SEARCH_TAG_RE.search(text)
    if m is None:
        return None
    cs, ce = m.start(), m.end()

    start = next((i for i in range(len(token_ids)) if cum[i + 1] > cs), None)
    end = next((i + 1 for i in range(len(token_ids)) if cum[i + 1] >= ce), len(token_ids))
    if start is None or start >= end:
        return None
    return (start, end)
