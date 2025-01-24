#!/bin/bash

datasets=("citeseer" "cora" "pubmed")

python experiments/abalation_test/run_abalation_tests.py --dataset "$ds" --nheads 10 --nrepeats 5
