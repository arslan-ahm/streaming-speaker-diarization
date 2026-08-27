cd "F:/projects/new ai projects/streaming-speaker-diarization"
export OMP_NUM_THREADS=2
PY=.venv/Scripts/python.exe
CSV=results/tables/latency_sweep_momentum035.csv
for s in 0 1 2; do
  for b in 0 250 500 1000 2000 4000 8000; do
    $PY scripts/latency_sweep.py --budget-ms $b --seed $s --csv $CSV \
        --set diarizer.centroid_momentum=0.35
  done
done
echo "SENSITIVITY DONE"
