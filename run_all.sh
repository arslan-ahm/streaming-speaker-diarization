cd "F:/projects/new ai projects/streaming-speaker-diarization"
export OMP_NUM_THREADS=2
PY=.venv/Scripts/python.exe
RC=0

echo "=== STAGE comparison ==="
$PY scripts/run_experiments.py --stage comparison --seeds 0,1,2 || RC=1

echo "=== STAGE latency sweep (one invocation per budget per seed) ==="
for s in 0 1 2; do
  for b in 0 125 250 500 1000 2000 4000 8000 16000; do
    $PY scripts/latency_sweep.py --budget-ms $b --seed $s || RC=1
  done
  $PY scripts/latency_sweep.py --offline --seed $s || RC=1
done
$PY scripts/latency_sweep.py --summarise || RC=1

echo "=== STAGE ablations ==="
$PY scripts/run_experiments.py --stage ablations --seeds 0,1,2 || RC=1

echo "=== STAGE calibration ==="
$PY scripts/run_experiments.py --stage calibration --seeds 0,1,2 || RC=1

echo "=== STAGE efficiency ==="
$PY scripts/run_experiments.py --stage efficiency --seeds 0 || RC=1

echo "=== FIGURES ==="
$PY scripts/make_figures.py || RC=1

echo "ALL DONE rc=$RC"
exit $RC
