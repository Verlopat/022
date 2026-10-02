#!/bin/bash
# usage: run_rl.sh <delta> <episodes> <seed> <tag>
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/claude/work
python3 rl.py train "$@" > "logs_rl_$4.txt" 2>&1
