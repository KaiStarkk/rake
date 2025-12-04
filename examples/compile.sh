#!/usr/bin/env bash
# Rake compilation pipeline: .rk -> MLIR -> LLVM -> native
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$SCRIPT_DIR/build"

# Check for required tools
check_tools() {
    local missing=""
    command -v mlir-opt >/dev/null 2>&1 || missing="$missing mlir-opt"
    command -v mlir-translate >/dev/null 2>&1 || missing="$missing mlir-translate"
    command -v llc >/dev/null 2>&1 || missing="$missing llc"
    command -v clang >/dev/null 2>&1 || missing="$missing clang"

    if [ -n "$missing" ]; then
        echo "Missing tools:$missing"
        echo "Run with: nix-shell -p llvmPackages.mlir llvmPackages.llvm llvmPackages.clang"
        exit 1
    fi
}

compile_rake() {
    local input="$1"
    local output="${2:-${input%.rk}}"
    local basename="$(basename "$input" .rk)"

    echo "=== Compiling $basename ==="

    # Step 1: Rake -> MLIR
    echo "[1/5] Rake -> MLIR"
    (cd "$PROJECT_DIR" && nix develop --command dune exec -- rakec "$input" -o "$BUILD_DIR/$basename.mlir")

    # Step 2: MLIR optimization and lowering
    echo "[2/5] MLIR -> LLVM dialect"
    mlir-opt "$BUILD_DIR/$basename.mlir" \
        --convert-vector-to-llvm \
        --convert-math-to-llvm \
        --convert-arith-to-llvm \
        --convert-func-to-llvm \
        --reconcile-unrealized-casts \
        -o "$BUILD_DIR/$basename.llvm.mlir"

    # Step 3: MLIR LLVM dialect -> LLVM IR
    echo "[3/5] LLVM dialect -> LLVM IR"
    mlir-translate --mlir-to-llvmir "$BUILD_DIR/$basename.llvm.mlir" -o "$BUILD_DIR/$basename.ll"

    # Step 4: LLVM IR -> Object file
    echo "[4/5] LLVM IR -> Object ($BUILD_DIR/$basename.o)"
    llc -O3 -march=x86-64 -mcpu=native -filetype=obj \
        "$BUILD_DIR/$basename.ll" -o "$BUILD_DIR/$basename.o"

    # Step 5: Generate assembly for inspection
    echo "[5/5] Generating assembly ($BUILD_DIR/$basename.s)"
    llc -O3 -march=x86-64 -mcpu=native \
        "$BUILD_DIR/$basename.ll" -o "$BUILD_DIR/$basename.s"

    echo ""
    echo "=== Compilation complete ==="
    echo "Object file: $BUILD_DIR/$basename.o"
    echo "Assembly:    $BUILD_DIR/$basename.s"
    echo "LLVM IR:     $BUILD_DIR/$basename.ll"
    echo ""

    # Show function list
    echo "Exported functions:"
    nm "$BUILD_DIR/$basename.o" | grep -E "^[0-9a-f]+ T" | awk '{print "  " $3}'
}

# Main
mkdir -p "$BUILD_DIR"
check_tools

if [ $# -eq 0 ]; then
    # Default: compile particles.rk
    compile_rake "$SCRIPT_DIR/particles.rk"
else
    compile_rake "$1" "${2:-}"
fi
