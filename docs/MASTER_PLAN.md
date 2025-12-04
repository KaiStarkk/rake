# Rake Language Master Plan

## Executive Summary

Rake is a SIMD-first programming language designed to make vectorized computation the default, not an optimization. Where C, Rust, and Zig treat SIMD as an advanced feature requiring intrinsics or hoping for auto-vectorization, Rake treats scalar operations as the special case.

**Core Thesis**: Most numerical code should be vectorized, but isn't, because:
1. Auto-vectorization is fragile and unpredictable
2. Hand-written intrinsics are unportable and unmaintainable
3. Conditional logic breaks vectorization in traditional languages

Rake solves these by making vectors (`rack`) the default type and providing first-class support for predicated execution (`rails`).

---

## 1. Current Implementation Status

### 1.1 What Works Today

| Component | Status | Description |
|-----------|--------|-------------|
| Lexer | Complete | OCamllex-based, handles all tokens |
| Parser | Complete | Menhir LR(1), full grammar support |
| Type Checker | Partial | Basic type inference, SoA/AoS types |
| MLIR Emitter | Complete | Vector/Arith/Math/Func dialects |
| MLIR → LLVM | Complete | Full lowering pipeline |
| LLVM → Native | Complete | Via `llc` and `clang` |

### 1.2 Language Features

#### Implemented
- **Rack types**: `float rack`, `int rack` - 8-wide vectors (AVX2)
- **Scalar types**: `<name>` syntax for scalar variables
- **Stack structs**: `stack Name = { field: type rack; ... }` (SoA layout)
- **AoS structs**: `aos Name = { field: type; ... }`
- **Rails**: Conditional execution with `| condition -> body`
- **Named rails**: `| name := predicate -> body`
- **Otherwise**: `| otherwise -> default`
- **Functions**: `crunch` (pure) and `rake` (rails-enabled) with ML-style params
- **Let bindings**: `let x = expr in body`
- **Binary operators**: `+ - * / %` on vectors
- **Comparison operators**: `< <= > >= == !=` producing masks
- **Built-in math**: `sqrt`, `sin`, `cos`, `exp`, `log`, `abs`, `floor`, `ceil`
- **Reductions**: `reduce(+, expr)`, `reduce(*, expr)`, etc.
- **Results clause**: `results <| expr` for explicit return values

#### Partially Implemented
- **Pipeline operator**: `|>` parsed but not fused
- **Field access**: `.field` on structs (type checking incomplete)
- **Pack types**: `Name pack` for SIMD-friendly collections
- **Iteration**: `sweep`, `spread`, `repeat` (parsing only)
- **Compaction**: `compact` operation for removing retired lanes

#### Not Yet Implemented
- **Gather/Scatter**: Memory operations with index vectors (parsed, not codegen)
- **Lane manipulation**: `shuffle`, `rotate`, `shift`, `compress`, `expand` (parsed)
- **Modules**: Import/export system
- **Generics**: Parameterized types
- **SIMD width selection**: Currently hardcoded to 8 (AVX2)

---

## 2. Performance Strategy

### 2.1 How to Beat Auto-Vectorization

Auto-vectorization fails in predictable ways. Rake exploits each failure mode:

| Auto-Vec Failure | Why It Fails | Rake Solution |
|------------------|--------------|----------------|
| **Conditionals** | Branches prevent vectorization | Rails → `select` (always vectorized) |
| **Data layout** | AoS memory patterns | `stack` keyword enforces SoA |
| **Aliasing** | Compiler can't prove non-overlap | Immutable values, explicit layout |
| **Function calls** | Can't inline across boundaries | Aggressive inlining + fusion |
| **Complex control** | Loops with early exit | Predicated execution throughout |
| **Reductions** | Horizontal ops are slow | First-class reduction operations |

### 2.2 Target Performance Model

```
Theoretical Peak (AVX2): 8 × clock × cores FLOPs
Auto-vectorized C:       ~30-60% of peak (varies wildly)
Hand-tuned intrinsics:   ~70-90% of peak
Rake target:            ~70-85% of peak (consistent)
```

The key insight: **consistency beats occasional peaks**. Rake code should always vectorize, while C/Rust code vectorizes only when stars align.

### 2.3 Current Benchmark Results

```
1M particles, 100 iterations:
  Rake (MLIR→LLVM):     453.7 M particles/sec
  C (auto-vectorized):   475.8 M particles/sec
  Gap:                   ~5% slower
```

**Why the gap exists**:
1. Function call overhead (each Rake op is a separate function)
2. No fusion of pipeline operations
3. No link-time optimization

**How we close it** (see Section 4: Roadmap):
1. Pipeline fusion eliminates intermediate values
2. Whole-program compilation enables inlining
3. LTO across Rake/C boundaries

### 2.4 Path to Performance Leadership

The 5% gap represents function call overhead and lack of fusion. Closing this gap—and then exceeding C—requires three phases:

#### Phase A: Eliminate Overhead (Target: Parity)

1. **Pipeline Fusion**: Transform `a |> f |> g |> h` into a single fused loop
   - Currently: Each `|>` produces intermediate vector, each function is a call
   - Target: Single loop body with inlined operations, no intermediates
   - Implementation: MLIR Transform dialect patterns for `|>` sequences
   - Expected gain: 15-25%

2. **Function Inlining**: Inline all `crunch` functions at call sites
   - Currently: Each function is emitted separately, called via LLVM
   - Target: Aggressive inlining controlled by size heuristics
   - Implementation: MLIR inliner pass with Rake-specific cost model
   - Expected gain: 5-15%

3. **Register Allocation Hints**: Help LLVM keep vectors in registers
   - Currently: LLVM decides register allocation without context
   - Target: Emit LLVM metadata suggesting register pressure
   - Implementation: Custom MLIR-to-LLVM lowering attributes
   - Expected gain: 5-10%

#### Phase B: Exploit Language Guarantees (Target: 20% Lead)

Auto-vectorizers work with limited information. Rake has **guaranteed** properties:

1. **No Aliasing**: All Rake values are immutable
   - C compilers must assume pointers may alias
   - Rake can always parallelize without checks
   - Implementation: Emit `noalias` on all pointers, use value semantics

2. **Predictable Memory Layout**: `stack` enforces structure-of-arrays
   - C compilers analyze layout; may give up on complex types
   - Rake knows at compile time: all fields are contiguous
   - Implementation: Generate optimal gather patterns statically

3. **Rails Are Masks**: Branch-free predicated execution
   - C compilers: branch prediction + speculative execution
   - Rake: always evaluate both paths, mask-select result
   - For divergent workloads (ray tracing), this is 2-3x faster

4. **Whole-Program View**: See all code at compile time
   - C compilers: limited to translation units or LTO
   - Rake: can analyze entire program's data flow
   - Enables global optimization decisions

#### Phase C: Advanced Optimizations (Target: Match Hand-Tuned)

1. **Rail Merging**: Combine overlapping predicates
   ```rake
   (* Before optimization *)         (* After optimization *)
   | x > 0 -> f(x)                   | x > 0 and x < 1 -> g(x)
   | x < 1 -> g(x)          →        | x >= 1 -> f(x)
   | otherwise -> h(x)               | otherwise -> h(x)
   ```
   Reduces mask operations by detecting overlap at compile time.

2. **SIMD Width Selection**: Choose optimal vector width per function
   - AVX-512 for large data sets
   - AVX2 for mixed workloads (less frequency throttling)
   - SSE for memory-bound operations
   - Implementation: Cost model based on operation mix

3. **Automatic Blocking**: Tile loops for cache hierarchy
   - L1: 32KB → 4K floats per block
   - L2: 256KB → 32K floats per block
   - L3: 8MB+ → buffer entire working set
   - Implementation: Affine dialect + tiling transform

4. **Memory Prefetching**: Predict access patterns from SoA layout
   - Rake knows field access patterns statically
   - Emit prefetch hints for predictable streams
   - Implementation: Custom lowering pass

### 2.5 Performance Leadership Timeline

```
Week 1-2:  Pipeline fusion prototype (basic |> chains)
Week 3-4:  Function inlining infrastructure
Week 5-6:  Ray tracer demo implementation
Week 7-8:  Benchmark + optimize ray tracer to beat C
Month 3:   Rail merging optimization
Month 4:   SIMD width selection
Month 5:   Affine tiling integration
Month 6:   GPU targeting proof-of-concept
```

---

## 3. Killer Demo Analysis

### 3.1 Candidate Applications

| Application | Auto-Vec Difficulty | Rake Advantage | Demo Potential |
|-------------|--------------------|-----------------| ---------------|
| **Mandelbrot** | Medium | Rails for divergence | High (visual) |
| **Particles** | Medium | SoA + rails for bounds | High (visual) |
| **Audio DSP** | Low | Consistent perf | Medium |
| **Image filters** | Low-Medium | Rails for edge cases | High (visual) |
| **Ray tracing** | High | Rails for hit/miss | Very High |
| **N-body** | High | SoA + complex conditions | Very High |
| **Neural net inference** | Medium | Matrix ops + activation | High |
| **Physics simulation** | High | Constraint solving | Very High |

### 3.2 Recommended Killer Demo: Ray Tracing

**Why ray tracing is ideal**:

1. **Divergent execution**: Each ray may hit different objects, creating branches that kill auto-vectorization. Rake rails handle this naturally.

2. **Visual impact**: Results are immediately compelling and shareable.

3. **Well-understood benchmark**: Compare against existing implementations.

4. **Multiple rail conditions**:
   ```rake
   rake trace_ray ray =
     | hit_sphere := intersect_sphere(ray) -> shade_sphere(ray)
     | hit_plane := intersect_plane(ray) -> shade_plane(ray)
     | hit_box := intersect_box(ray) -> shade_box(ray)
     | otherwise -> background_color
   ```

5. **SoA benefits**: Rays, hits, and colors all benefit from SoA layout.

### 3.3 Secondary Demos

1. **Mandelbrot zoom**: Shows rails handling iteration divergence
2. **Particle collision**: Shows SoA layout + boundary rails
3. **Audio synthesis**: Shows consistent real-time performance

---

## 4. Development Roadmap

### Phase 1: Performance Parity (Current)
- [x] MLIR emission
- [x] LLVM lowering
- [x] Native compilation
- [x] Basic benchmarking
- [ ] Pipeline fusion
- [ ] Function inlining

### Phase 2: Performance Leadership
- [ ] Whole-program optimization
- [ ] Cross-module inlining
- [ ] SIMD width selection (AVX-512, NEON)
- [ ] Gather/scatter operations
- [ ] Advanced rail optimization

### Phase 3: Ecosystem
- [ ] Module system
- [ ] Package manager
- [ ] IDE support (LSP)
- [ ] Debugging support
- [ ] Profiling integration

### Phase 4: Production Ready
- [ ] Comprehensive test suite
- [ ] Formal specification
- [ ] Multiple backend targets
- [ ] Stable ABI
- [ ] Documentation

---

## 5. MLIR Optimization Roadmap

The key to performance leadership lies in leveraging MLIR's dialect system. We currently use basic dialects; the path forward involves progressively adopting higher-level dialects that enable more aggressive optimization.

### 5.1 Current Dialect Usage

| Dialect | Usage | Performance Impact |
|---------|-------|-------------------|
| `func` | Function definitions | Baseline |
| `arith` | Arithmetic operations | Baseline |
| `vector` | SIMD operations | **Core value** |
| `math` | Transcendentals | Baseline |
| `llvm` | Lowering target | Baseline |

### 5.2 Near-Term Dialect Adoption

**SCF (Structured Control Flow)** — Priority: High
- Replace direct loop lowering with `scf.for`, `scf.while`
- Enables loop-invariant code motion
- Enables loop fusion across pipeline stages
- Maps directly to our `sweep` construct

**Transform Dialect** — Priority: High
- Define Rake-specific optimization patterns
- Fusion rules for pipeline operator
- Rail merging when predicates overlap
- Custom tiling strategies for SIMD

### 5.3 Medium-Term Dialect Adoption

**Affine Dialect** — Priority: Medium
- Polyhedral analysis for nested loops
- Automatic cache-aware tiling
- Memory access pattern optimization
- Enables auto-parallelization

**Linalg Dialect** — Priority: Medium
- High-level matrix operation representation
- Automatic fusion of matrix chains
- Tiling for register blocking
- Path to GPU offload

### 5.4 Future Dialect Adoption

**GPU Dialect** — Priority: Future
- Same Rake source → GPU compute shaders
- CUDA/ROCm/Vulkan backends via SPIRV
- Automatic host/device partitioning

**Bufferization** — Priority: Future
- Minimize memory allocations
- In-place operation conversion
- Alias analysis for optimization

### 5.5 Performance Unlocks by Dialect

```
Dialect         Optimization                    Expected Gain
─────────────────────────────────────────────────────────────
SCF             Loop fusion                     10-30%
Transform       Pipeline operator fusion        20-40%
Affine          Cache-aware tiling              15-25%
Linalg          Matrix operation fusion         30-50%
GPU             Parallel execution              10-100x
```

---

## 6. Pipeline Architecture

### 6.1 Current Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAKE COMPILER                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  Source  │───▶│  Lexer   │───▶│  Parser  │───▶│   AST    │  │
│  │ .rake   │    │(ocamllex)│    │ (Menhir) │    │          │  │
│  └──────────┘    └──────────┘    └──────────┘    └────┬─────┘  │
│                                                       │        │
│                        RAKE (OCaml)                   │        │
├───────────────────────────────────────────────────────┼────────┤
│                                                       ▼        │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │   Type   │◀───│   AST    │    │   MLIR   │◀───│  MLIR    │  │
│  │  Check   │    │          │───▶│  Emitter │    │ (textual)│  │
│  └──────────┘    └──────────┘    └──────────┘    └────┬─────┘  │
│                                                       │        │
│                        RAKE (OCaml)                   │        │
├───────────────────────────────────────────────────────┼────────┤
│                                                       ▼        │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ mlir-opt │───▶│  LLVM    │───▶│mlir-trans│───▶│  LLVM IR │  │
│  │ (passes) │    │ Dialect  │    │  -late   │    │   .ll    │  │
│  └──────────┘    └──────────┘    └──────────┘    └────┬─────┘  │
│                                                       │        │
│                        MLIR TOOLS                     │        │
├───────────────────────────────────────────────────────┼────────┤
│                                                       ▼        │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │   llc    │───▶│  Object  │───▶│  clang   │───▶│  Binary  │  │
│  │          │    │   .o     │    │ (link)   │    │          │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│                                                                 │
│                        LLVM TOOLS                               │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Component Ownership

| Stage | Tool | We Control | They Control |
|-------|------|------------|--------------|
| Lexing | ocamllex | Token definitions | Regex engine |
| Parsing | Menhir | Grammar, AST shape | Parse algorithm |
| Type check | Custom | Everything | - |
| MLIR gen | Custom | IR structure | - |
| MLIR opt | mlir-opt | Pass selection | Pass implementation |
| LLVM gen | mlir-translate | - | Translation |
| Codegen | llc | Flags only | Everything |
| Linking | clang | Flags only | Everything |

### 6.3 Optimization Opportunities by Stage

```
Stage           Current         Potential       Difficulty
─────────────────────────────────────────────────────────
Parsing         None            Incremental     Low
Type Check      Basic           Full inference  Medium
MLIR Gen        Direct emit     Optimize first  Medium
MLIR Opt        Basic passes    Custom passes   High
LLVM            Default         LTO, PGO        Low
```

---

## 7. Feature Coverage Matrix

### 7.1 MLIR Feature Utilization

| MLIR Capability | Current Use | Potential | Priority |
|-----------------|-------------|-----------|----------|
| Vector dialect ops | 60% | 95% | High |
| Math dialect | 80% | 95% | Medium |
| Arith dialect | 90% | 100% | Low |
| Func dialect | 100% | 100% | - |
| SCF dialect | 0% | 80% | High |
| Affine dialect | 0% | 70% | Medium |
| Linalg dialect | 0% | 60% | Low |
| GPU dialect | 0% | 90% | Future |
| Transform dialect | 0% | 50% | Medium |
| Pass pipeline | Basic | Custom | High |

### 7.2 Rake Language Coverage

| Language Feature | Spec | Impl | Tests | Docs |
|------------------|------|------|-------|------|
| Rack types | 100% | 100% | 50% | 20% |
| Scalar types | 100% | 80% | 30% | 10% |
| SoA structs | 100% | 90% | 40% | 20% |
| AoS structs | 100% | 90% | 20% | 10% |
| Rails | 100% | 95% | 60% | 30% |
| Functions | 100% | 95% | 50% | 30% |
| Pipeline op | 80% | 30% | 10% | 10% |
| Reductions | 100% | 80% | 30% | 20% |
| Iteration | 60% | 20% | 0% | 0% |
| Modules | 20% | 0% | 0% | 0% |

---

## 8. Risk Analysis

### 8.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| MLIR breaking changes | Low | High | Pin LLVM version |
| Performance ceiling | Medium | High | Custom MLIR passes |
| Complex type system | Medium | Medium | Incremental design |
| Memory management | High | Medium | Design carefully |

### 8.2 Adoption Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| No killer demo | Medium | High | Focus on ray tracer |
| Learning curve | High | Medium | Good docs, familiar syntax |
| Limited ecosystem | High | Medium | C FFI, gradual adoption |
| Competition | Low | Low | Unique positioning |

---

## 9. Success Metrics

### 9.1 Performance Milestones

1. **Parity** (Current): Match auto-vectorized C
2. **Lead**: 20% faster than auto-vectorized C on rail-heavy code
3. **Dominance**: Match hand-tuned intrinsics on target demos

### 9.2 Language Milestones

1. **Bootstrap**: Compile particles.rake to working binary ✓
2. **Demo**: Ray tracer that beats C implementation
3. **Usable**: Module system, error messages, debugging
4. **Production**: Stable ABI, package ecosystem

### 9.3 Adoption Milestones

1. **Interest**: Conference talk / blog post attention
2. **Trial**: External contributors try Rake
3. **Use**: Real project uses Rake for hot path
4. **Recommend**: Users advocate for Rake

---

## 10. Conclusion

Rake occupies a unique position: a language where SIMD is the default, not an optimization. The path to success requires:

1. **Demonstrating clear wins** on problems where auto-vectorization fails
2. **Minimizing friction** through familiar syntax and good tooling
3. **Leveraging existing infrastructure** (MLIR, LLVM) rather than reinventing

The immediate priorities are:
1. **Pipeline fusion** to close the performance gap
2. **Ray tracer demo** to prove the concept
3. **Polish** to make the language pleasant to use

The MLIR bet is correct - it provides the optimization infrastructure we need while allowing us to focus on language design. A custom backend would be a distraction with no clear benefit.

---

## Appendix A: File Structure

```
rake/
├── lib/                    # Compiler library
│   ├── ast.ml             # AST definition
│   ├── lexer.mll          # Lexer specification
│   ├── parser.mly         # Parser grammar
│   ├── types.ml           # Type definitions
│   ├── check.ml           # Type checker
│   ├── mlir.ml            # MLIR emitter
│   └── emit.ml            # Legacy LLVM emitter
├── bin/
│   └── main.ml            # Compiler driver
├── examples/
│   ├── particles.rake    # Main example
│   ├── particles.c        # C comparison
│   ├── particles_rust/    # Rust comparison
│   ├── compile.sh         # Compilation script
│   ├── benchmark.sh       # Benchmark script
│   └── build/             # Build artifacts
├── docs/
│   └── MASTER_PLAN.md     # This document
└── dune-project           # Build configuration
```

## Appendix B: Dialect Migration Path

```
Current:    Rake → vector/arith/math/func → llvm → native

Phase 2:    Rake → scf/vector/arith → vector/arith → llvm → native
                    (structured loops)  (loop opts)

Phase 3:    Rake → affine/linalg → scf/vector → llvm → native
                    (high-level)    (tiling)

Future:     Rake → linalg → gpu/spirv → GPU binary
                           ↘ llvm → CPU binary
```

## Appendix C: Comparison with Alternatives

| Language | SIMD Approach | Pros | Cons |
|----------|---------------|------|------|
| C | Intrinsics or auto-vec | Universal, fast | Fragile, unportable |
| Rust | std::simd (nightly) | Safe, explicit | Verbose, unstable |
| Zig | @Vector built-in | Simple, explicit | Manual, no rails |
| ISPC | Implicit vectorization | Automatic, proven | Separate language |
| **Rake** | Default vectors + rails | Consistent, elegant | New, unproven |

---

*Document version: 2.0*
*Last updated: December 2024*
*Focus: Performance Leadership via MLIR dialect adoption*
