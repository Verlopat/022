#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/claude/work
python3 run_dqn.py select "$1" "$2" > "logs_sel_$1.txt" 2>&1
