#!/usr/bin/env python3

import os
from pathlib import Path

def get_target_dir():
    script_dir = Path(__file__).resolve().parent

    current = script_dir

    while current != current.parent:
        candidate = current / "tests/filecheck/interpreter/full-programs"
        if candidate.exists():
            return candidate
        current = current.parent

    raise RuntimeError("Could not find full-programs directory")


def list_existing_mlir_files(target_dir: Path):
    return list(target_dir.glob("*.mlir"))


def generate_arith_chain(n_ops: int) -> str:
    lines = []
    lines.append("// AUTO-GENERATED BENCHMARK")
    lines.append("builtin.module {")
    lines.append("  func.func @main() -> (i32) {")
    lines.append("")
    lines.append('    %c0 = "arith.constant"() <{value = 0 : i32}> : () -> i32')
    lines.append('    %c1 = "arith.constant"() <{value = 1 : i32}> : () -> i32')
    lines.append("")

    prev = "%c0"

    for i in range(1, n_ops + 1):
        cur = f"%v{i}"
        lines.append(
            f'    {cur} = "arith.addi"({prev}, %c1) '
            f'<{{overflowFlags = #arith.overflow<none>}}> : (i32, i32) -> i32'
        )
        prev = cur

    lines.append("")
    lines.append(f"    func.return {prev} : i32")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ops", type=int, default=100)
    parser.add_argument("--out", type=str, default="large_prog.mlir")
    args = parser.parse_args()

    target_dir = get_target_dir()

    # optional: show existing files
    existing = list_existing_mlir_files(target_dir)
    print(f"[INFO] Found {len(existing)} MLIR files in benchmark directory")

    out_path = target_dir / args.out

    mlir = generate_arith_chain(args.ops)

    with open(out_path, "w") as f:
        f.write(mlir)

    print(f"[OK] Wrote benchmark to: {out_path}")