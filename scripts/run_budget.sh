#!/bin/bash
# Budget run on half the machine: B = N + EXTRA, GAMES per N for N = 4..8.
#
#   scripts/run_budget.sh EXTRA GAMES [AGENT ...] [--ident]
#   scripts/run_budget.sh 16 50 exact            # confirm B = N + 16
#   scripts/run_budget.sh 16 50 exact --ident    # ... plus identifiability checks
#
# Uses at most half the cores and (at 2 GB per process) half the RAM, one thread
# per process, low priority. Resumable: rerun the same command to continue.
# Output: results/budget<EXTRA>/{all.jsonl, ident.jsonl, summary.txt}, logs/budget<EXTRA>*.log
set -u
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
EXTRA=${1:?usage: scripts/run_budget.sh EXTRA GAMES [AGENT ...] [--ident]}
GAMES=${2:?usage: scripts/run_budget.sh EXTRA GAMES [AGENT ...] [--ident]}
shift 2
AGENTS=(); IDENT=0
for a in "$@"; do
  if [ "$a" = "--ident" ]; then IDENT=1; else AGENTS+=("$a"); fi
done
[ ${#AGENTS[@]} -eq 0 ] && AGENTS=(exact)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1

CORES=$(sysctl -n hw.ncpu 2>/dev/null || nproc)
MEM=$(sysctl -n hw.memsize 2>/dev/null || awk '/MemTotal/ {print $2 * 1024}' /proc/meminfo)
RAM_GB=$(( MEM / 1073741824 ))
P=$(( CORES / 2 ))
P_RAM=$(( RAM_GB / 2 / 2 ))
[ "$P_RAM" -lt "$P" ] && P=$P_RAM
[ "$P" -lt 1 ] && P=1
OUT=results/budget$EXTRA
mkdir -p "$OUT" logs
echo "$(date)  B=N+$EXTRA, $GAMES games/N, agents: ${AGENTS[*]}; cores=$CORES ram=${RAM_GB}GB -> $P processes"

nice -n 10 python scripts/evaluate.py --agents "${AGENTS[@]}" --cipher random \
    --n 4 5 6 7 8 --games "$GAMES" --budget-extra "$EXTRA" --resume --workers "$P" \
    --out "$OUT/all.jsonl" > "logs/budget$EXTRA.log" 2>&1
echo "$(date)  evaluate finished (exit $?)"

if [ "$IDENT" = 1 ]; then
  nice -n 10 python scripts/identifiability.py "$OUT/all.jsonl" \
      --at 8 10 12 14 16 18 20 22 end --seconds 120 \
      --out "$OUT/ident.jsonl" --resume > "logs/budget$EXTRA-ident.log" 2>&1
  echo "$(date)  identifiability finished (exit $?)"
fi

{
  python scripts/evaluate.py --summarize "$OUT/all.jsonl"
  [ -f "$OUT/ident.jsonl" ] && { echo; python scripts/identifiability.py --summarize "$OUT/ident.jsonl"; }
  echo; echo "games with caught CP-SAT errors: $(grep -c solver_errors "$OUT/all.jsonl")"
} > "$OUT/summary.txt" 2>&1
echo "$(date)  all done; summary in $OUT/summary.txt"
