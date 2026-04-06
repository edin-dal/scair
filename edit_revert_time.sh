#!/usr/bin/env bash
set -euo pipefail

# ---- CONFIG ----
ARITH_FILE="/Users/leogabayoyo/University/Year-4/Dissertation/scair/interpreter/src/Dialects/Arith.scala"
MILL_CMD=(./mill tools.runTool.nativeImage)
RUNNER="/Users/leogabayoyo/University/Year-4/Dissertation/scair/out/tools/runTool/launcher.dest/run"
MLIR="/Users/leogabayoyo/University/Year-4/Dissertation/scair/tests/filecheck/interpreter/full-programs/fib.mlir"
N=10
LOG_DIR="./bench_logs_$(date +%Y%m%d_%H%M%S)"
# --------------

if [[ ! -f "./mill" ]]; then
  echo "ERROR: Run from repo root (where ./mill exists)." >&2
  exit 1
fi
if [[ ! -f "$ARITH_FILE" ]]; then
  echo "ERROR: File not found: $ARITH_FILE" >&2
  exit 1
fi
if [[ ! -x "$RUNNER" ]]; then
  echo "ERROR: Runner not executable: $RUNNER" >&2
  exit 1
fi
if [[ ! -f "$MLIR" ]]; then
  echo "ERROR: MLIR not found: $MLIR" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
ALL_LOG="$LOG_DIR/all_output.log"
CSV="$LOG_DIR/timings.csv"

tmp_backup="$(mktemp)"
cleanup() {
  [[ -f "$tmp_backup" ]] && cp "$tmp_backup" "$ARITH_FILE" || true
  rm -f "$tmp_backup" >/dev/null 2>&1 || true
}
trap cleanup EXIT
cp "$ARITH_FILE" "$tmp_backup"

now_ns() { python3 - <<'PY'
import time
print(time.time_ns())
PY
}

make_file_unique() {
  python3 - "$ARITH_FILE" <<'PY'
import re, time, sys
from pathlib import Path

path = Path(sys.argv[1])
s = path.read_text(encoding="utf-8")

prefix = "private val __bench_touch: Long = "
line = prefix + f"{time.time_ns()}L"

# Put it right after package line to keep stable location.
pat = r'^\s*private val __bench_touch: Long = \d+L\s*$'

if re.search(pat, s, flags=re.M):
    s2, n = re.subn(pat, line, s, flags=re.M)
    if n != 1:
        print(f"ERROR: expected 1 __bench_touch line, replaced {n}", file=sys.stderr)
        sys.exit(2)
else:
    # insert after package line
    s2, n = re.subn(r'^(package[^\n]*\n)',
                    r'\1\n' + line + '\n',
                    s,
                    count=1,
                    flags=re.M)
    if n != 1:
        s2 = s + "\n" + line + "\n"

path.write_text(s2, encoding="utf-8")
print(line)
PY
}

echo "Logs: $LOG_DIR"
echo "Combined log: $ALL_LOG"
echo "iter,compile_s,run_s,total_s" > "$CSV"

{
  echo "=== bench start: $(date) ==="
  echo "ARITH_FILE=$ARITH_FILE"
  echo "MILL_CMD=${MILL_CMD[*]}"
  echo "RUNNER=$RUNNER"
  echo "MLIR=$MLIR"
  echo "N=$N"
  echo

  for i in $(seq 1 "$N"); do
    echo "=== Iteration $i/$N ==="

    echo "--- make file unique (comment marker) ---"
    marker="$(make_file_unique)"
    echo "marker: $marker"

    echo "--- compile ---"
    t0="$(now_ns)"
    "${MILL_CMD[@]}"
    t1="$(now_ns)"

    echo "--- run ---"
    t2="$(now_ns)"
    "$RUNNER" "$MLIR"
    t3="$(now_ns)"

    python3 - <<PY | tee -a "$CSV"
def s(a,b): return (int(b)-int(a))/1e9
compile_s = s($t0,$t1)
run_s = s($t2,$t3)
total_s = s($t0,$t3)
print(f"$i,{compile_s:.6f},{run_s:.6f},{total_s:.6f}")
PY

    echo "--- restore original file (safety) ---"
    cp "$tmp_backup" "$ARITH_FILE"
    echo
  done

  echo "=== bench end: $(date) ==="
  echo "CSV: $CSV"
} >>"$ALL_LOG" 2>&1

echo "Done. See:"
echo "  $ALL_LOG"
echo "  $CSV"