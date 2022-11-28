#!/bin/bash
CONFIGS=("LSTM/roberta_large_entity_marker_punct_tokens_LSTM(3)" "LSTM/roberta_large_entity_marker_tokens_LSTM(3)" "LSTM/roberta_large_entity_mask_tokens_LSTM(3)" "LSTM/roberta_large_typed_entity_marker_punct_tokens_LSTM(3)" "LSTM/roberta_large_typed_entity_marker_tokens_LSTM(3)" "LSTM/koelectra_entity_marker_punct_tokens_LSTM(3)" "LSTM/koelectra_entity_marker_tokens_LSTM(3)" "LSTM/koelectra_entity_mask_tokens_LSTM(3)" "LSTM/koelectra_typed_entity_marker_punct_tokens_LSTM(3)" "LSTM/koelectra_typed_entity_marker_tokens_LSTM(3)")

for (( i=0; i<10; i++ ))
do
    python3 train.py --config ${CONFIGS[$i]}
done