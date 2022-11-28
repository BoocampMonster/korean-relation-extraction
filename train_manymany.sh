#!/bin/bash
CONFIGS=("T5/T5_baseline" "T5/T5_baseline3e-5" "T5/T5_large_baseline" "T5/T5_large_baseline3e-5" "T5/T5_GRU" "T5/T5_GRU3e-5" "T5/T5_large_GRU" "T5/T5_large_GRU3e-5" "T5/T5_LSTM" "T5/T5_LSTM3e-5" "T5/T5_large_LSTM" "T5/T5_large_LSTM3e-5" "T5_Tokens" "T5_large_Tokens" "T5_Tokens3e-5" "T5_large_Tokens3e-5")

for (( i=0; i<10; i++ ))
do
    python3 train.py --config ${CONFIGS[$i]}
done