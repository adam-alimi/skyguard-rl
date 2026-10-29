#!/usr/bin/env bash
# Experiment 2: competitive self-play under noisy defender sensing, IPPO vs asymmetric self-play (S-IPPO).
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=.venv/Scripts/python
for s in 0 1; do
  $PY train_selfplay.py --method ippo  --sensing noisy --seed $s --chunks 20 --chunk-iters 10 > logs/sp_ippo_s$s.log 2>&1 &
  $PY train_selfplay.py --method sippo --sensing noisy --seed $s --chunks 20 --chunk-iters 10 > logs/sp_sippo_s$s.log 2>&1 &
done
wait
