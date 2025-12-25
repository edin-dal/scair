#!/bin/sh

IN_DIR="./tests/filecheck/dialects/sdql/tpch-gen"
failed=""

for i in $(seq 1 22); do
  q="q$i"
  f="$IN_DIR/$q.mlir"

  echo "Running $q..."
  scair-opt "$f" >/dev/null 2>&1

  if [ $? -ne 0 ]; then
    echo "  FAILED: $q"
    failed="$failed $q"
  fi
done

if [ -n "$failed" ]; then
  echo "Done, failures:$failed"
  exit 1
else
  echo "Done. All good."
fi
