#!/usr/bin/env bash
# Build script for Rake programs
# Usage: ./scripts/build_rake.sh <source.rk> [output_name]

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <source.rk> [output_name]"
    echo ""
    echo "Compiles a Rake source file through the full pipeline:"
    echo "  .rk -> MLIR -> LLVM IR -> Object file"
    exit 1
fi

SOURCE="$1"
BASENAME=$(basename "$SOURCE" .rk)
OUTPUT="${2:-$BASENAME}"

# Directory setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_DIR}/_build/default/bin"
TMP_DIR="/tmp/rake_build_$$"

mkdir -p "$TMP_DIR"
trap "rm -rf $TMP_DIR" EXIT

echo "=== Rake Compiler Pipeline ==="
echo "Source: $SOURCE"
echo ""

# Step 1: Rake -> MLIR
echo "[1/4] Compiling Rake to MLIR..."
"$BUILD_DIR/main.exe" --emit-mlir "$SOURCE" > "$TMP_DIR/$BASENAME.mlir"
echo "      Generated: $BASENAME.mlir"

# Step 2: MLIR validation
echo "[2/4] Validating MLIR..."
mlir-opt "$TMP_DIR/$BASENAME.mlir" > /dev/null
echo "      MLIR is valid"

# Step 3: MLIR -> LLVM dialect -> LLVM IR
echo "[3/4] Lowering to LLVM IR..."
mlir-opt "$TMP_DIR/$BASENAME.mlir" \
    --convert-vector-to-scf \
    --convert-scf-to-cf \
    --convert-vector-to-llvm \
    --convert-math-to-llvm \
    --convert-arith-to-llvm \
    --convert-index-to-llvm \
    --convert-func-to-llvm \
    --convert-cf-to-llvm \
    --finalize-memref-to-llvm \
    --reconcile-unrealized-casts \
    -o "$TMP_DIR/$BASENAME.llvm.mlir"

mlir-translate --mlir-to-llvmir "$TMP_DIR/$BASENAME.llvm.mlir" -o "$TMP_DIR/$BASENAME.ll"
echo "      Generated: $BASENAME.ll"

# Step 4: LLVM IR -> Object file
echo "[4/4] Compiling to native code..."
llc -O3 -filetype=obj -march=x86-64 -mattr=+avx2 "$TMP_DIR/$BASENAME.ll" -o "$OUTPUT.o"
echo "      Generated: $OUTPUT.o"

echo ""
echo "=== Build Complete ==="
echo "Output: $OUTPUT.o"
echo ""
echo "To link with a C harness:"
echo "  clang -O3 -mavx2 harness.c $OUTPUT.o -o $OUTPUT -lm"
