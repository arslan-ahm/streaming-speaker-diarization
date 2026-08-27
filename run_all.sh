set -e
cd "F:/projects/new ai projects/streaming-speaker-diarization"
export OMP_NUM_THREADS=2
PY=.venv/Scripts/python.exe

for s in 1 2; do
  if [ ! -f "checkpoints/noncausal_seed$s.pt" ]; then
    echo "=== train noncausal seed $s ==="
    $PY scripts/train.py --config configs/ablation_noncausal.yaml --seed $s --quiet \
      --checkpoint checkpoints/noncausal_seed$s.pt
  fi
done

echo "=== STAGE comparison ==="
$PY scripts/run_experiments.py --stage comparison --seeds 0,1,2

echo "=== STAGE latency sweep (chunked: one invocation per budget per seed) ==="
for s in 0 1 2; do
  for b in 0 125 250 500 1000 2000 4000 8000 16000; do
    $PY scripts/latency_sweep.py --budget-ms $b --seed $s
  done
  $PY scripts/latency_sweep.py --offline --seed $s
done
$PY scripts/latency_sweep.py --summarise

echo "=== STAGE ablations ==="
$PY scripts/run_experiments.py --stage ablations --seeds 0,1,2

echo "=== STAGE calibration ==="
$PY scripts/run_experiments.py --stage calibration --seeds 0,1,2

echo "=== STAGE efficiency ==="
$PY scripts/run_experiments.py --stage efficiency --seeds 0

echo "ALL EXPERIMENTS DONE"
ls -la results/tables/
