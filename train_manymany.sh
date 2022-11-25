#!/bin/bash
CONFIGS=("roberta_large_entity_marker_punct_cls" "roberta_large_entity_marker_cls" "roberta_large_entity_mask_cls" "roberta_large_typed_entity_marker_punct_cls" "roberta_large_typed_entity_marker_cls")

for (( i=0; i<5; i++ ))
do
    python3 train.py --config ${CONFIGS[$i]}
done