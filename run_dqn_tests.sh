#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/claude/work
python3 run_dqn.py test dqn_d10_s0 225 DQN-C P1 v1 > logs_test_dqnc.txt 2>&1
python3 run_dqn.py test dqno_d10_g8 200 DQN-O P1 > logs_test_dqno.txt 2>&1
python3 run_dqn.py series dqn_d10_s0:225:DQN-C:v1,dqno_d10_g8:200:DQN-O > logs_series_dqn.txt 2>&1
python3 run_dqn.py test dqn_d10_s0 225 DQN-C P0,P2,P3 v1 > logs_test_dqnc_px.txt 2>&1
echo done > logs_dqn_tests_done.txt
