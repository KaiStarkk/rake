# Rake Development Priorities

## Current Focus: Close the Performance Gap

The 5% performance gap versus auto-vectorized C must close before anything else matters.

---

## Immediate (This Week)

### 1. Pipeline Fusion
**File**: `lib/mlir.ml`
**Problem**: `a |> f |> g` emits separate function calls with intermediate values
**Solution**: Detect pipeline chains, inline operations, eliminate intermediates
**Expected gain**: 20-40% on pipeline-heavy code

### 2. Function Inlining
**File**: `lib/mlir.ml`
**Problem**: Every `crunch` emits as separate MLIR function
**Solution**: Inline small `crunch` functions at call sites before MLIR emission
**Expected gain**: 5-15%

### 3. Vec3 Built-ins
**Files**: `lib/mlir.ml`, `lib/types.ml`
**Need**: `dot`, `cross`, `length`, `normalize` for ray tracer
**Approach**: Emit inline MLIR, not function calls

---

## Short Term (This Month)

### 4. Ray Tracer Foundation
**File**: `bench/suites/raytracing/rake/raytracing.rk`
**Components**:
- [ ] Ray-sphere intersection with rails
- [ ] Ray-plane intersection
- [ ] Basic shading (diffuse)
- [ ] Camera ray generation

### 5. Benchmark Harness
**File**: `bench/run.sh`
**Need**: Automated comparison of Rake vs C on same algorithm
**Output**: Table with particles/sec or rays/sec

### 6. Error Messages
**Files**: `lib/check.ml`, `lib/parser.mly`
**Problem**: Errors don't show source location
**Solution**: Thread position through AST, format errors with line/column

---

## Medium Term (Next Month)

### 7. Complete Ray Tracer
- [ ] Multiple object types (spheres, planes, boxes)
- [ ] BVH acceleration structure
- [ ] Rail-based material selection
- [ ] Reflections
- [ ] Output to PPM/PNG

### 8. SCF Dialect Adoption
**File**: `lib/mlir.ml`
**Why**: Enables loop fusion across `sweep` iterations
**Work**: Emit `scf.for` instead of direct loop lowering

### 9. C FFI
**Files**: New `lib/ffi.ml`
**Need**: Call Rake functions from C, call C functions from Rake
**Approach**: Generate C-compatible function signatures

---

## Deferred (Later)

- Module system
- AVX-512 / NEON backends
- LSP server
- Generics
- GPU targeting

---

## Implementation Status

### Compiler Pipeline

| Stage | Status | Notes |
|-------|--------|-------|
| Lexer | Done | All tokens implemented |
| Parser | Done | Full grammar |
| Type checker | 70% | Missing: full inference, better errors |
| MLIR emission | 80% | Missing: fusion, inlining |
| MLIR → LLVM | Done | Via mlir-opt + mlir-translate |
| LLVM → native | Done | Via llc + clang |

### Language Features

| Feature | Parsed | Type-checked | Codegen |
|---------|--------|--------------|---------|
| `float rack` | Yes | Yes | Yes |
| `vec3 rack` | Yes | Partial | Partial |
| `stack` structs | Yes | Yes | Partial |
| `crunch` functions | Yes | Yes | Yes |
| `rake` functions | Yes | Yes | Yes |
| Rails `\|` | Yes | Yes | Yes |
| `\|>` pipeline | Yes | No | Partial |
| `sweep` | Yes | No | No |
| `spread` | Yes | No | No |
| `gather`/`scatter` | Yes | No | No |
| `compact` | Yes | No | No |

### Benchmarks

| Benchmark | Rake | C | Status |
|-----------|------|---|--------|
| Particles | 453.7M/s | 475.8M/s | 5% gap |
| Ray tracer | - | - | Not implemented |
| Mandelbrot | - | - | Not implemented |

---

## Decision Log

### Decided
- MLIR over direct LLVM IR (leverage existing optimizations)
- ML-inspired syntax (rails as pattern matching)
- `<name>` for scalars (visual distinction from racks)
- Ray tracer as killer demo (divergent workload)

### Open Questions
- Module system syntax (ML-style vs Rust-style?)
- Error handling model (masks? exceptions? both?)
- Standard library scope (minimal vs batteries-included?)

---

*Updated: December 2024*
