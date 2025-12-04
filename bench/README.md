# Rake Evaluation Arena

A continuous benchmarking system for comparing Rake against competitor languages.

## Directory Structure

```
eval/
├── arena/          # CLI orchestrator
│   └── arena.ml    # Main entry point
├── lib/            # Core library
│   ├── config.ml   # Configuration and discovery
│   ├── compiler.ml # Multi-language compilation
│   ├── runner.ml   # Benchmark execution
│   ├── metrics.ml  # Vectorization/memory analysis
│   └── report.ml   # Report generation
├── bench/          # Benchmark implementations
│   ├── particles/  # Particle simulation [READY for Rake, C, Rust]
│   ├── mandelbrot/ # Mandelbrot fractal
│   ├── dsp/        # Digital signal processing
│   ├── filters/    # Image convolution
│   ├── raytracing/ # Ray tracing (KILLER DEMO)
│   ├── nbody/      # N-body simulation
│   ├── inference/  # ML inference
│   └── physics/    # Physics simulation
└── results/        # Historical benchmark results (JSON)
```

## Building

```bash
# Enter the nix development shell (provides all dependencies)
nix develop

# Build the arena
cd eval && dune build

# Or build from project root
dune build @all
```

## Usage

```bash
# Check status of benchmarks
dune exec -- rake-arena status

# Run all ready benchmarks
dune exec -- rake-arena run

# Run specific apps/languages
dune exec -- rake-arena run --apps particles,mandelbrot --langs rake,c

# Generate markdown report
dune exec -- rake-arena run --format markdown --output results/report.md

# Generate JSON for CI/tracking
dune exec -- rake-arena run --format json --output results/latest.json

# Compare results over time
dune exec -- rake-arena compare results/old.json results/new.json
```

## Benchmark Status

Benchmarks are marked with status indicators:
- `[OK]` - Ready to run
- `[WIP]` - Work in progress
- `[---]` - Stub only
- `[ ]` - Not available

Status is auto-detected by examining source files:
- Files starting with `// STUB` or `# STUB` → Stub
- Files starting with `// WIP` or `# WIP` → WIP
- Otherwise → Ready

## Adding a New Benchmark

1. Create directory under `bench/<app>/<lang>/`
2. Add source file: `<app>.<ext>` (e.g., `particles.rk`)
3. Implement the benchmark following existing patterns
4. For Rake: add `harness.c` in the app directory for linking
5. Output throughput as: `X.XX M items/sec` for auto-detection

## Metrics Collected

- **Throughput**: Items processed per second
- **Vectorization**: % of SIMD instructions in assembly
- **Memory**: Peak memory usage (via /usr/bin/time)
- **Code Size**: Binary size comparison
- **Compile Time**: Build performance
- **Theoretical Efficiency**: % of theoretical best vectorization

## Languages

| Language | Extension | Compiler | Notes |
|----------|-----------|----------|-------|
| Rake     | .rk       | rake → MLIR → LLVM | SIMD-first |
| C        | .c        | gcc/clang -O3 | Auto-vectorization baseline |
| Rust     | .rs       | rustc/cargo | Auto-vectorization |
| Zig      | .zig      | zig build | Explicit SIMD via @Vector |
| Mojo     | .mojo     | mojo build | Python-like with SIMD |
| Bend     | .bend     | bend compile | Massively parallel (HVM) |
| Odin     | .odin     | odin build | C alternative with #simd |

## Applications

| App | Description | Why It Matters |
|-----|-------------|----------------|
| particles | Bouncing particle simulation | Tests branching vectorization |
| mandelbrot | Fractal computation | Tests divergent iteration |
| dsp | Audio filters (FIR/IIR) | Tests streaming SIMD |
| filters | Image convolution | Tests 2D data layout |
| **raytracing** | Ray-sphere intersection | **KILLER DEMO** - divergent rays |
| nbody | Gravitational simulation | Tests O(n^2) computation |
| inference | Neural network forward pass | Tests matrix operations |
| physics | Collision detection | Tests AABB/sphere overlap |

## CI Integration

Results are saved as JSON in `results/` with timestamps:
```
results/
├── results_20241201_143022.json
├── results_20241202_091534.json
└── ...
```

Use `rake-arena compare` to track performance regressions.
