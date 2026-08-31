#!/bin/bash
# Download the CNY prompts and graphs, then lay them out as the code expects.
#
#   bash scripts/fetch_data.sh
#
# Layout produced (what walker.run and walker.eval.run resolve in walker/_data.py):
#   data/train.jsonl                training prompts (8-dataset mixture, 317,495)
#   data/eval/<dataset>.jsonl       held-out prompts, one per evaluation setting
#   data/raw_datasets/<dataset>.pt  graphs read during <walk>
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$REPO_ROOT/data"
HUB="$DATA/_hub"
: "${CNY_DATA_REPO:=Allen-UQ/CNY-data}"

command -v hf >/dev/null 2>&1 || { echo "ERROR: pip install 'huggingface_hub[cli]'" >&2; exit 3; }

echo "[cny] downloading $CNY_DATA_REPO"
hf download "$CNY_DATA_REPO" --repo-type dataset --local-dir "$HUB"

mkdir -p "$DATA/eval" "$DATA/raw_datasets"

# Graphs are already in the layout the code wants.
for f in "$HUB"/raw_datasets/*; do
  ln -sf "$f" "$DATA/raw_datasets/$(basename "$f")"
done

# Training prompts: the 8-dataset mixture the reported models were trained on.
ln -sf "$HUB/train/mixture/train.jsonl" "$DATA/train.jsonl"

# Evaluation prompts. `eval_dataset` names one prompt file AND one graph, so each
# n-way variant gets a graph alias pointing at the graph it was built from.
#   <eval_dataset name>  <hub eval config>  <graph basename>
while read -r name cfg graph; do
  [ -n "$name" ] || continue
  ln -sf "$HUB/eval/$cfg/test.jsonl" "$DATA/eval/$name.jsonl"
  ln -sf "$HUB/raw_datasets/$graph.pt" "$DATA/raw_datasets/$name.pt"
done <<'MAP'
cora               cora-7way        cora
cora_2way          cora-2way        cora
wikics             wikics-10way     wikics
wikics_5way        wikics-5way      wikics
products           products-full    products
products_10way     products-10way   products_sub
products_5way      products-5way    products_sub
fb15k237_edge_tag  fb15k237-10way   fb15k237_edge_tag
expla_graph_walk   expla-graph      expla_graph_walk
webqsp_walk        webqsp           webqsp_walk
MAP

echo "[cny] data ready under $DATA"
echo "      eval settings: $(ls "$DATA/eval" | sed 's/\.jsonl//' | tr '\n' ' ')"
