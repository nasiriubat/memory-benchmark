#!/usr/bin/env bash
# Unattended multi-model campaign orchestration.
#
# Runs the remaining Gemini systems + the full Claude Sonnet 5 campaign, one
# system at a time, verifying each landed a clean 120 rows with the memory
# backend actually engaged (not local fallback). Clears + retries once on a bad
# result, then moves on (logging loudly) so one flaky system can't block the run.
# Finishes with aggregate + Excel export.
set -uo pipefail
cd "$(dirname "$0")/.."

log() { echo "[$(date +%H:%M:%S)] $*"; }

verify() { # rid sys -> prints "OK ..." / "BAD ..."
  RID="$1" SYS="$2" python3 - <<'PY'
import json, os, collections
rid=os.environ["RID"]; s=os.environ["SYS"]
f=f"results/by_memory_system/{s}/{rid}/results.jsonl"
try:
    rows=[json.loads(l) for l in open(f)]
except FileNotFoundError:
    print("BAD missing file"); raise SystemExit
n=len(rows)
fbT=sum(1 for r in rows if r.get('memory_uses_local_fallback') is True)
rm=sorted({str(r.get('resolved_model')) for r in rows})
ok = (n>=120) and (fbT<=3)
print(("OK" if ok else "BAD")+f" rows={n} fallbackTrue={fbT} resolved={rm}")
PY
}

runsys() { # rid models sys
  local rid="$1" models="$2" sys="$3" res
  for attempt in 1 2; do
    log "RUN rid=$rid models=$models sys=$sys attempt=$attempt"
    rm -rf "results/by_memory_system/$sys/$rid" "results/raw/$rid/$sys"
    RUN_ID="$rid" MODELS="$models" make "$sys" || log "make $sys exited non-zero (continuing to verify)"
    res=$(verify "$rid" "$sys")
    log "VERIFY $sys -> $res"
    [[ "$res" == OK* ]] && { log "OK $sys (rid=$rid)"; return 0; }
    log "$sys not clean; clearing + retrying"
  done
  log "WARNING: $sys STILL not clean after 2 attempts (rid=$rid) -- leaving as-is, review in the morning"
  return 0
}

log "=== OVERNIGHT CAMPAIGN START ==="

# --- Gemini 3.5 Flash: fix mem0 (2 fallback) + zep (all fallback) + acontext ---
runsys benchmark_gemini35flash google_fast mem0
runsys benchmark_gemini35flash google_fast zep
runsys benchmark_gemini35flash google_fast acontext

# --- Claude Sonnet 5: full sweep (stateless FIRST for gain-vs-stateless) ---
runsys benchmark_claudesonnet5 anthropic_sonnet stateless
runsys benchmark_claudesonnet5 anthropic_sonnet hindsight
runsys benchmark_claudesonnet5 anthropic_sonnet mem0
runsys benchmark_claudesonnet5 anthropic_sonnet zep
runsys benchmark_claudesonnet5 anthropic_sonnet acontext

log "=== AGGREGATE ==="
make aggregate || log "aggregate FAILED"

log "=== EXCEL EXPORT ==="
docker compose -p memory-agent-benchmark run --rm --no-deps -v "$(pwd):/app" \
  --entrypoint python stateless /app/scripts/export_excel.py || log "excel export FAILED"

log "=== OVERNIGHT CAMPAIGN DONE ==="