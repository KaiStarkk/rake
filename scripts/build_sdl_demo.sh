#!/usr/bin/env bash
# Build script for Rake SDL raytracer demo
# Usage: ./scripts/build_sdl_demo.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_DIR}/_build/default/bin"
TMP_DIR="/tmp/rake_sdl_build"

mkdir -p "$TMP_DIR"
mkdir -p "$PROJECT_DIR/build"

echo "================================================================"
echo "      Building Rake SDL Raytracer Demo"
echo "================================================================"

# Step 1: Build Rake compiler
echo ""
echo "[1/5] Building Rake compiler..."
cd "$PROJECT_DIR"
dune build
echo "      Done"

# Step 2: Compile Rake raytracer to MLIR
echo ""
echo "[2/5] Compiling Rake to MLIR..."
"$BUILD_DIR/main.exe" --emit-mlir "$PROJECT_DIR/examples/standalone_raytracer.rk" > "$TMP_DIR/raytracer.mlir"
echo "      Generated: raytracer.mlir"

# Step 3: Lower MLIR to LLVM IR
echo ""
echo "[3/5] Lowering to LLVM IR..."
mlir-opt "$TMP_DIR/raytracer.mlir" \
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
    -o "$TMP_DIR/raytracer.llvm.mlir"

mlir-translate --mlir-to-llvmir "$TMP_DIR/raytracer.llvm.mlir" -o "$TMP_DIR/raytracer.ll"
echo "      Generated: raytracer.ll"

# Step 4: Add inlining attributes and compile
echo ""
echo "[4/5] Compiling to native code with LTO..."

cat > "$TMP_DIR/raytracer_inline.ll" << 'HEADER'
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

HEADER

sed 's/define <8 x float> @dot(\(.*\)) {/define <8 x float> @dot(\1) #99 {/' "$TMP_DIR/raytracer.ll" | \
sed 's/define <8 x float> @intersect_flat(\(.*\)) {/define <8 x float> @intersect_flat(\1) #99 {/' >> "$TMP_DIR/raytracer_inline.ll"

echo '' >> "$TMP_DIR/raytracer_inline.ll"
echo 'attributes #99 = { alwaysinline }' >> "$TMP_DIR/raytracer_inline.ll"

clang -O3 -flto -mavx2 -c "$TMP_DIR/raytracer_inline.ll" -o "$PROJECT_DIR/build/raytracer_rake_lto.o" 2>/dev/null || \
    llc -O3 -filetype=obj -march=x86-64 -mattr=+avx2 "$TMP_DIR/raytracer.ll" -o "$PROJECT_DIR/build/raytracer_rake_lto.o"
echo "      Generated: build/raytracer_rake_lto.o"

# Step 5: Compile demo
echo ""
echo "[5/5] Building SDL demo..."

SDL_CFLAGS=$(pkg-config --cflags sdl2 2>/dev/null || echo "-I/usr/include/SDL2")
SDL_LIBS=$(pkg-config --libs sdl2 2>/dev/null || echo "-lSDL2")
echo "      SDL2 flags: $SDL_CFLAGS $SDL_LIBS"

clang -O3 -flto -mavx2 -Wall $SDL_CFLAGS \
    "$PROJECT_DIR/examples/sdl_demo.c" \
    "$PROJECT_DIR/build/raytracer_rake_lto.o" \
    -o "$PROJECT_DIR/build/sdl_demo" \
    $SDL_LIBS -lm 2>/dev/null || \
clang -O3 -mavx2 -Wall $SDL_CFLAGS \
    "$PROJECT_DIR/examples/sdl_demo.c" \
    "$PROJECT_DIR/build/raytracer_rake_lto.o" \
    -o "$PROJECT_DIR/build/sdl_demo" \
    $SDL_LIBS -lm

# Also build the headless benchmark
echo "      Building headless benchmark..."
clang -O3 -flto -mavx2 \
    "$PROJECT_DIR/examples/benchmark.c" \
    "$PROJECT_DIR/build/raytracer_rake_lto.o" \
    -o "$PROJECT_DIR/build/benchmark" \
    -lm 2>/dev/null || \
clang -O3 -mavx2 \
    "$PROJECT_DIR/examples/benchmark.c" \
    "$PROJECT_DIR/build/raytracer_rake_lto.o" \
    -o "$PROJECT_DIR/build/benchmark" \
    -lm

echo ""
echo "================================================================"
echo "  Build complete!"
echo ""
echo "  Interactive demo:    ./build/sdl_demo"
echo "  Headless benchmark:  ./build/benchmark"
echo ""
echo "  Controls (sdl_demo):"
echo "    1 = C Scalar"
echo "    2 = C SIMD (AVX2)"
echo "    3 = Rake SIMD"
echo "    Q = Quit"
echo "================================================================"
