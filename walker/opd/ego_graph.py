from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class NeighborPreview:
    id: int
    title: str
    preview: str

@dataclass(frozen=True)
class EgoGraph:
    node_id: int
    title: str
    content: str
    neighbors: tuple[NeighborPreview, ...]
    label: int | None = None
    label_name: str | None = None
