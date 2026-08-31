from __future__ import annotations

_DATASET_TASK_TYPE: dict[str, str] = {
    "cora": "node_class",
    "citeseer": "node_class",
    "pubmed": "node_class",
    "photo": "node_class",
    "computer": "node_class",
    "history": "node_class",
    "sportsfit": "node_class",
    "instagram": "node_class",
    "wikics": "node_class",
    "products": "node_class",
    "arxiv": "node_class",
    "wn18rr_edge_tag": "relation_class",
    "fb15k237_edge_tag": "relation_class",
    "cora_link": "link_pred",
    "expla_graph_tag": "graph_reason",
    "expla_graph": "graph_reason",
}

_DEFAULT_TASK_TYPE = "node_class"


def task_type_for(dataset_name: str) -> str:
    return _DATASET_TASK_TYPE.get(dataset_name, _DEFAULT_TASK_TYPE)
