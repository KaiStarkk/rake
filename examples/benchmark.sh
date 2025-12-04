#!/usr/bin/env bash
# Benchmark script for Rake vs C vs Rust particle simulation
#
# This script:
# 1. Builds all versions with optimizations
# 2. Runs hyperfine benchmarks
# 3. Generates assembly comparison

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

# Configuration
NUM_PARTICLES="${1:-100000}"
NUM_ITERATIONS="${2:-100}"

echo "================================================"
echo "Rake Language Benchmark Suite"
echo "================================================"
echo "Particles: $NUM_PARTICLES"
echo "Iterations: $NUM_ITERATIONS"
echo ""

# Create build directory
mkdir -p "$BUILD_DIR"

# ============================================
# Build C version
# ============================================
echo "Building C version..."
gcc -O3 -march=native -ffast-math -fopenmp-simd \
    -o "$BUILD_DIR/particles_c" \
    "$SCRIPT_DIR/particles.c" -lm

# Also build with explicit AVX2 for comparison
gcc -O3 -mavx2 -mfma -ffast-math -fopenmp-simd \
    -o "$BUILD_DIR/particles_c_avx2" \
    "$SCRIPT_DIR/particles.c" -lm

echo "  Built: $BUILD_DIR/particles_c"
echo "  Built: $BUILD_DIR/particles_c_avx2"

# ============================================
# Build Rust version
# ============================================
echo ""
echo "Building Rust version..."
cd "$SCRIPT_DIR/particles_rust"
if command -v cargo &> /dev/null; then
    RUSTFLAGS="-C target-cpu=native" cargo build --release --quiet
    cp target/release/particles "$BUILD_DIR/particles_rust"
    echo "  Built: $BUILD_DIR/particles_rust"
    RUST_AVAILABLE=true
else
    echo "  cargo not found, skipping Rust build"
    echo "  Install Rust or run: nix-shell -p cargo rustc"
    RUST_AVAILABLE=false
fi
cd "$SCRIPT_DIR"

# ============================================
# Generate assembly for comparison
# ============================================
echo ""
echo "Generating assembly comparison..."

# C assembly (update_particles function)
gcc -O3 -march=native -ffast-math -fopenmp-simd -S \
    -o "$BUILD_DIR/particles_c.s" \
    "$SCRIPT_DIR/particles.c"

# C assembly with explicit AVX2
gcc -O3 -mavx2 -mfma -ffast-math -fopenmp-simd -S \
    -o "$BUILD_DIR/particles_c_avx2.s" \
    "$SCRIPT_DIR/particles.c"

echo "  Generated: $BUILD_DIR/particles_c.s"
echo "  Generated: $BUILD_DIR/particles_c_avx2.s"

# ============================================
# Run benchmarks
# ============================================
echo ""
echo "================================================"
echo "Running Benchmarks"
echo "================================================"
echo ""

# Check if hyperfine is available
if command -v hyperfine &> /dev/null; then
    if [ "${RUST_AVAILABLE:-true}" = true ]; then
        hyperfine --warmup 3 --runs 10 \
            --export-markdown "$BUILD_DIR/benchmark_results.md" \
            --export-json "$BUILD_DIR/benchmark_results.json" \
            -n "C (native)" "$BUILD_DIR/particles_c $NUM_PARTICLES $NUM_ITERATIONS" \
            -n "C (AVX2)" "$BUILD_DIR/particles_c_avx2 $NUM_PARTICLES $NUM_ITERATIONS" \
            -n "Rust" "$BUILD_DIR/particles_rust $NUM_PARTICLES $NUM_ITERATIONS"
    else
        hyperfine --warmup 3 --runs 10 \
            --export-markdown "$BUILD_DIR/benchmark_results.md" \
            --export-json "$BUILD_DIR/benchmark_results.json" \
            -n "C (native)" "$BUILD_DIR/particles_c $NUM_PARTICLES $NUM_ITERATIONS" \
            -n "C (AVX2)" "$BUILD_DIR/particles_c_avx2 $NUM_PARTICLES $NUM_ITERATIONS"
    fi

    echo ""
    echo "Results saved to:"
    echo "  $BUILD_DIR/benchmark_results.md"
    echo "  $BUILD_DIR/benchmark_results.json"
else
    echo "hyperfine not found, running simple benchmarks..."
    echo ""
    echo "C (native):"
    time "$BUILD_DIR/particles_c" "$NUM_PARTICLES" "$NUM_ITERATIONS"
    echo ""
    echo "C (AVX2):"
    time "$BUILD_DIR/particles_c_avx2" "$NUM_PARTICLES" "$NUM_ITERATIONS"
    if [ "${RUST_AVAILABLE:-true}" = true ]; then
        echo ""
        echo "Rust:"
        time "$BUILD_DIR/particles_rust" "$NUM_PARTICLES" "$NUM_ITERATIONS"
    fi
fi

# ============================================
# Generate educational assembly comparison
# ============================================
echo ""
echo "================================================"
echo "Assembly Comparison"
echo "================================================"

cat > "$BUILD_DIR/assembly_comparison.md" << 'EOF'
# Assembly Comparison: Auto-vectorization vs Explicit SIMD

This document compares how different compilers vectorize the particle simulation code.

## Key Observations

### C with Auto-vectorization

The C compiler attempts to auto-vectorize loops when it can prove:
1. No loop-carried dependencies
2. Memory accesses don't alias
3. The trip count is sufficient

Look for patterns like:
- `vmovups` / `vmovaps` - Vector load/store
- `vaddps` / `vmulps` - Vector arithmetic
- `vcmpps` - Vector comparison
- `vblendvps` - Vector select (for conditionals)

### Challenges with Auto-vectorization

1. **Conditional Code**: The `bounce_pos` and `bounce_vel` functions have branches.
   Compilers must convert these to predicated operations (blend/select).

2. **Function Calls**: Inline functions must be inlined for vectorization.
   Non-inlined calls break vectorization.

3. **Memory Layout**: SoA layout is crucial. AoS would prevent vectorization
   due to non-contiguous memory access patterns.

4. **Aliasing**: The compiler must prove that `p->x` and `p->vx` don't overlap.
   This is why we use `restrict` or separate arrays.

### Rake's Approach (MLIR Output)

Rake generates MLIR that explicitly uses vector types:
- `vector<8xf32>` - 8-wide float vectors (AVX2)
- `arith.addf` on vectors - Always vectorized
- `arith.select` - Rails become predicated operations

The key difference is that Rake:
1. **Starts with vectors** - no auto-vectorization needed
2. **Rails are first-class** - conditionals are always SIMD-friendly
3. **SoA is enforced** - `soa` keyword guarantees optimal layout

## Sample Assembly Patterns

### Scalar Code (not vectorized)
```asm
movss   xmm0, [rdi]      ; Load single float
addss   xmm0, xmm1       ; Add single float
movss   [rdi], xmm0      ; Store single float
```

### AVX2 Vectorized Code
```asm
vmovups ymm0, [rdi]      ; Load 8 floats
vaddps  ymm0, ymm0, ymm1 ; Add 8 floats
vmovups [rdi], ymm0      ; Store 8 floats
```

### Predicated Operation (for branches)
```asm
vcmpltps ymm2, ymm0, ymm3    ; Compare x < limit
vblendvps ymm0, ymm1, ymm0, ymm2  ; Select based on mask
```

## Performance Implications

| Approach | Pros | Cons |
|----------|------|------|
| C auto-vec | Portable, familiar | Fragile, compiler-dependent |
| Rust auto-vec | Safe, same LLVM backend | Same limitations as C |
| Rake explicit | Guaranteed vectorization | New language to learn |

## Viewing the Generated Assembly

To see the actual assembly:
```bash
# C assembly
less build/particles_c_avx2.s

# Look for the update_particles function
grep -A 200 "update_particles:" build/particles_c_avx2.s
```

EOF

echo "Generated: $BUILD_DIR/assembly_comparison.md"

# Extract key assembly sections
echo ""
echo "Key assembly from C (AVX2) update_particles:"
echo "----------------------------------------------"
if grep -q "update_particles" "$BUILD_DIR/particles_c_avx2.s"; then
    # Try to extract the function
    sed -n '/^update_particles:/,/^[a-z_]*:/p' "$BUILD_DIR/particles_c_avx2.s" | head -50
else
    echo "(Function may be inlined - check full assembly)"
fi

echo ""
echo "================================================"
echo "Benchmark Complete"
echo "================================================"
echo ""
echo "Files generated in $BUILD_DIR:"
ls -la "$BUILD_DIR"
