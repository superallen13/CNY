from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from torch import Tensor

def build_adjacency_list(edge_index: Optional[Tensor], num_nodes: int) -> Dict[int, List[int]]:
    
    adj_list = {i: [] for i in range(num_nodes)}
    if edge_index is None:
        return adj_list
    for src, dst in edge_index.t().tolist():
        adj_list[src].append(dst)
    return adj_list

def sample_khop_neighbors(
    node_idx: int,
    adj_list: Dict[int, List[int]],
    k: int,
    max_neighbors: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[int]:
    
    visited = set()
    current_layer = {node_idx}

    for _ in range(k):
        next_layer = set()
        for node in current_layer:
            neighbors = adj_list.get(node, [])
            if max_neighbors is not None and max_neighbors > 0 and len(neighbors) > max_neighbors:
                if rng is not None:
                    neighbors = rng.choice(neighbors, size=max_neighbors, replace=False).tolist()
                else:
                    neighbors = neighbors[:max_neighbors]
            next_layer.update(neighbors)

        visited.update(next_layer)
        current_layer = next_layer

    return list(visited - {node_idx})
