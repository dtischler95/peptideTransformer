#!/bin/bash
export PATH=/usr/local/bin:/usr/bin:/bin
cd /mnt/e/Test/peptideTransformer || { echo "repo dir not found"; exit 1; }
mkdir -p results
LOGFILE="results/sweep_$(date +%Y%m%d_%H%M%S).log"
# setsid + nohup + stdin from /dev/null fully detaches the run from the launching
# session, so closing the WSL invocation (or this terminal) does not stop it.
setsid nohup ~/peptide-cpu-venv/bin/python3 run_variance_sweep.py --seeds 42 1 2 3 4 --n-jobs 3 > "$LOGFILE" 2>&1 < /dev/null &
sleep 3
echo "PID=$(pgrep -f run_variance_sweep.py | head -1)"
echo "LOG=$LOGFILE"
