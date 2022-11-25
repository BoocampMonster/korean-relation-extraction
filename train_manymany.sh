#!/bin/bash
CONFIGS=("roberta_large_entity_marker_punct_tokens" "roberta_large_entity_marker_tokens" "roberta_large_entity_mask_tokens" "roberta_large_typed_entity_marker_punct_tokens" "roberta_large_typed_entity_marker_tokens")

for (( i=0; i<5; i++ ))
do
    python3 train.py --config ${CONFIGS[$i]}
done