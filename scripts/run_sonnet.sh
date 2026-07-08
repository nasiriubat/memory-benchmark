#!/usr/bin/env bash
# Run the Claude Opus 4.8 campaign (OpenRouter, Bedrock avoided).
# Verification catches API errors / zero-token rows as well as row count and
# fallback. zep is excluded (broken backend).
set -uo pipefail
cd "$(dirname "$0")/.."

log() { echo "[$(date +%H:%M:%S)] $*"; }
RID=benchmark_claudeopus48
MODELS=anthropic_opus

verify() { # sys -> "OK ..." / "BAD ..."
  RID="$RID" SYS="$1" python3 - <<'PY'
import json, os
rid=os.environ["RID"]; s=os.environ["SYS"]
f=f"results/by_memory_system/{s}/{rid}/results.jsonl"
try:
    rows=[json.loads(l) for l in open(f)]
except FileNotFoundError:
    print("BAD missing file"); raise SystemExit
n=len(rows)
err=sum(1 for r in rows if r.get('error'))
fbT=sum(1 for r in rows if r.get('memory_uses_local_fallback') is True)
tok=sum(r.get('output_tokens',0) for r in rows)/max(n,1)
ok = (n>=120) and (err==0) and (tok>0) and (fbT<=3)
print(("OK" if ok else "BAD")+f" rows={n} errors={err} meanTok={tok:.0f} fallbackTrue={fbT}")
PY
}

runsys() { # sys
  local sys="$1" res
  for attempt in 1 2; do
    log "RUN $sys attempt=$attempt"
    rm -rf "results/by_memory_system/$sys/$RID" "results/raw/$RID/$sys"
    RUN_ID="$RID" MODELS="$MODELS" make "$sys" || log "make $sys exited non-zero"
    res=$(verify "$sys")
    log "VERIFY $sys -> $res"
    [[ "$res" == OK* ]] && { log "OK $sys"; return 0; }
  done
  log "WARNING: $sys STILL not clean after 2 attempts"
}

log "=== SONNET RE-RUN START ==="
runsys stateless
runsys hindsight
runsys mem0
runsys acontext
log "=== AGGREGATE ==="
make aggregate || log "aggregate FAILED"
log "=== EXCEL EXPORT ==="
docker compose -p memory-agent-benchmark run --rm --no-deps -v "$(pwd):/app" \
  --entrypoint python stateless /app/scripts/export_excel.py || log "excel FAILED"
log "=== SONNET RE-RUN DONE ==="
