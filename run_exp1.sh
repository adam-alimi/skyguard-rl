#!/usr/bin/env bash
# Experiment 1: learned guidance under randomised sensing. 2 seeds x 4 configurations.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=.venv/Scripts/python
ITERS=${ITERS:-200}
for s in 0 1; do
  $PY train_guidance.py --mode direct   --dr full --critic sym  --seed $s --iters $ITERS > logs/direct_s$s.log 2>&1 &
  $PY train_guidance.py --mode residual --dr full --critic sym  --seed $s --iters $ITERS > logs/residual_s$s.log 2>&1 &
  $PY train_guidance.py --mode residual --dr full --critic priv --seed $s --iters $ITERS > logs/residual_priv_s$s.log 2>&1 &
  $PY train_guidance.py --mode residual --dr none --critic sym  --seed $s --iters $ITERS > logs/residual_nodr_s$s.log 2>&1 &
done
wait
