#!/bin/bash
# Rake compiler pipeline: .rk -> MLIR -> LLVM IR -> Object -> Executable

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <file.rk> [output]"
    exit 1
fi

INPUT="$1"
BASENAME=$(basename "$INPUT" .rk)
OUTPUT="${2:-$BASENAME}"
TMPDIR=$(mktemp -d)

echo "=== Rake Compiler Pipeline ==="
echo "Input:  $INPUT"
echo "Output: $OUTPUT"
echo ""

# Step 1: Parse and type-check, emit MLIR
echo "[1/4] Parsing and emitting MLIR..."
dune exec -- rake --emit-mlir "$INPUT" > "$TMPDIR/$BASENAME.mlir"

# Step 2: Optimize and lower MLIR to LLVM dialect
echo "[2/4] Optimizing and lowering to LLVM dialect..."
mlir-opt "$TMPDIR/$BASENAME.mlir" \
    --inline \
    --cse \
    --canonicalize \
    --convert-vector-to-llvm \
    --convert-func-to-llvm \
    --convert-arith-to-llvm \
    --convert-math-to-llvm \
    --reconcile-unrealized-casts \
    -o "$TMPDIR/$BASENAME.llvm.mlir"

# Step 3: Translate to LLVM IR
echo "[3/4] Translating to LLVM IR..."
mlir-translate --mlir-to-llvmir "$TMPDIR/$BASENAME.llvm.mlir" -o "$TMPDIR/$BASENAME.ll"

# Step 4: Compile to object file
echo "[4/4] Compiling to object file..."
llc -O3 -march=x86-64 -mcpu=haswell -filetype=obj "$TMPDIR/$BASENAME.ll" -o "$OUTPUT.o"

echo ""
echo "=== Done! ==="
echo "Object file: $OUTPUT.o"
echo ""
echo "To link with a C program:"
echo "  gcc -O3 -mavx2 your_main.c $OUTPUT.o -lm -o your_program"

# Cleanup
rm -rf "$TMPDIR"
