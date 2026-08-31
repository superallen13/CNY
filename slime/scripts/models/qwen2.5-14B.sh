MODEL_ARGS=(
   --swiglu
   --num-layers 48
   --hidden-size 5120
   --ffn-hidden-size 13824
   --num-attention-heads 40
   --group-query-attention
   --num-query-groups 8
   --use-rotary-position-embeddings
   --disable-bias-linear
   --add-qkv-bias
   --normalization "RMSNorm"
   --norm-epsilon 1e-06
   --rotary-base 1000000
   --vocab-size 152064
   --untie-embeddings-and-output-weights
)
