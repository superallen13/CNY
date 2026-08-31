from __future__ import annotations

from pathlib import Path
from typing import Optional

from walker.tag.core import TagData, load_tag
from walker.tag.graph_ops import build_adjacency_list

class GraphView:

    def __init__(self, tag: TagData):
        self.tag = tag
        self.adj_list = build_adjacency_list(tag.edge_index, tag.num_nodes)

    @classmethod
    def from_path(cls, dataset_name: str, data_path: str) -> "GraphView":
        p = Path(data_path).expanduser().resolve()
        if p.is_dir():
            p = p / f"{dataset_name}.pt"
        return cls(load_tag(str(p)))

    def node_text(self, node_id: int) -> str:
        if self.tag.node_texts is None:
            return ""
        if 0 <= node_id < len(self.tag.node_texts):
            return self.tag.node_texts[node_id]
        return ""

    def neighbors(self, node_id: int) -> list[int]:
        if 0 <= node_id < len(self.adj_list):
            return [int(n) for n in self.adj_list[node_id]]
        return []

    def node_label(self, node_id: int) -> int | None:
        
        if self.tag.y is None:
            return None
        if 0 <= node_id < self.tag.y.shape[0]:
            return int(self.tag.y[node_id].item())
        return None

    @staticmethod
    def truncate(text: str, max_tokens: int, tokenizer: Optional[object] = None) -> str:
        
        if max_tokens == -1 or not text:
            return text
        if tokenizer is None:
            words = text.split()
            return text if len(words) <= max_tokens else " ".join(words[:max_tokens]) + " ..."
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            return text
        return tokenizer.decode(tokens[:max_tokens]) + " ..."
