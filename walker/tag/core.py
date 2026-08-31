from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Union

import torch
from torch import Tensor

@dataclass
class TagData:

    edge_index: Optional[Tensor] = None  
    node_texts: Optional[List[str]] = None
    edge_texts: Optional[List[str]] = None
    label_idx: Optional[List[str]] = None
    label_names: Optional[List[str]] = None
    graph_texts: Optional[List[Dict[str, Any]]] = None  
    y: Optional[Tensor] = None
    train_mask: Optional[Tensor] = None
    val_mask: Optional[Tensor] = None
    test_mask: Optional[Tensor] = None
    num_nodes_hint: Optional[int] = None
    num_graphs_hint: Optional[int] = None  
    node_attrs: Optional[Dict[str, List[Any]]] = None
    edge_attrs: Optional[Dict[str, List[Any]]] = None

    def __post_init__(self) -> None:
        if self.edge_index is not None:
            if self.edge_index.dtype != torch.long:
                raise TypeError("edge_index must be LongTensor.")
            if self.edge_index.dim() != 2 or self.edge_index.size(0) != 2:
                raise ValueError("edge_index must have shape [2, E].")

    @property
    def num_nodes(self) -> int:
        if self.num_nodes_hint is not None:
            return int(self.num_nodes_hint)
        if self.edge_index is None or self.edge_index.numel() == 0:
            return 0
        return int(self.edge_index.max().item()) + 1

    @property
    def num_edges(self) -> int:
        if self.edge_index is None:
            return 0
        return int(self.edge_index.size(1))

    @property
    def num_graphs(self) -> int:
        
        if self.num_graphs_hint is not None:
            return int(self.num_graphs_hint)
        if self.graph_texts is not None:
            return len(self.graph_texts)
        if self.y is not None:
            return int(self.y.shape[0])
        if self.train_mask is not None:
            return int(self.train_mask.shape[0])
        return 0

    @property
    def num_classes(self) -> int:
        if self.y is None:
            return 0
        return int(self.y.max().item()) + 1

    @property
    def level(self) -> Literal["node", "edge", "graph"]:
        
        if self.edge_index is None:
            return "graph"
        if self.y is None:
            raise ValueError("Cannot infer level from TagData without y.")
        n = self.y.shape[0]
        if n == self.num_nodes:
            return "node"
        if n == self.num_edges:
            return "edge"
        raise ValueError(
            f"y shape {tuple(self.y.shape)} matches neither num_nodes={self.num_nodes} "
            f"nor num_edges={self.num_edges}"
        )

    def to(self, device: Union[str, torch.device]) -> "TagData":
        dev = torch.device(device)
        return TagData(
            edge_index=self.edge_index.to(dev) if self.edge_index is not None else None,
            node_texts=self.node_texts,
            edge_texts=self.edge_texts,
            label_idx=self.label_idx,
            label_names=self.label_names,
            graph_texts=self.graph_texts,
            y=self.y.to(dev) if self.y is not None else None,
            train_mask=self.train_mask.to(dev) if self.train_mask is not None else None,
            val_mask=self.val_mask.to(dev) if self.val_mask is not None else None,
            test_mask=self.test_mask.to(dev) if self.test_mask is not None else None,
            num_nodes_hint=self.num_nodes_hint,
            num_graphs_hint=self.num_graphs_hint,
            node_attrs=self.node_attrs,
            edge_attrs=self.edge_attrs,
        )

    def get_mask(self, split: str) -> Tensor:
        mask = getattr(self, f"{split}_mask", None)
        if mask is None:
            raise ValueError(f"Split '{split}' not found or mask is None.")
        return mask

    def get_split_indices(self, split: str) -> Tensor:
        return self.get_mask(split).nonzero(as_tuple=False).view(-1)

def load_tag(path: str) -> TagData:
    
    data: Any = torch.load(path, map_location="cpu", weights_only=False)
    return TagData(**data)
