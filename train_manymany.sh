#!/bin/bash
CONFIGS=("Query/Query_typed_entity_marker_sentence4" "Query/Query_typed_entity_marker_sentence5" "Query/Query_typed_entity_marker_sentence6")

for (( i=0; i<3; i++ ))
do
    python3 train.py --config ${CONFIGS[$i]}
done