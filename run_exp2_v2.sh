#!/usr/bin/env bash
# Experiment 2, v2: same as run_exp2.sh, but a timeout costs the attacker as much as being intercepted.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=.venv/Scripts/python
for s in 0 1; do
  $PY train_selfplay.py --method ippo  --sensing noisy --att-timeout -10 --seed $s --chunks 20 --chunk-iters 10 > logs/sp2_ippo_s$s.log 2>&1 &
  $PY train_selfplay.py --method sippo --sensing noisy --att-timeout -10 --seed $s --chunks 20 --chunk-iters 10 > logs/sp2_sippo_s$s.log 2>&1 &
done
wait
