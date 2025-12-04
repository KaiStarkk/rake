# Rake: Strategic Vision

## The Core Thesis

**Auto-vectorization fails on divergent code. Rails fix this.**

Every compiler vendor has spent decades on auto-vectorization. It remains fragile because the fundamental problem is unsolvable: you cannot infer vectorization intent from scalar code. When different loop iterations need different things—when there are branches—compilers give up.

Rake inverts the model. Vectors are primitive. Branches don't exist. Instead, **rails** express predicated execution: all paths compute, masks select results. This isn't an optimization; it's the semantic foundation.

The language succeeds or fails on one question: **Can rails deliver consistent performance where auto-vectorization cannot?**

---

## Honest Assessment

### What Rake Is

A domain-specific language for CPU SIMD computation where:
- Data parallelism dominates
- Control flow diverges (different elements need different treatment)
- Performance predictability matters more than peak theoretical throughput

### What Rake Is Not

- A general-purpose language (use Rust, Go, C++)
- A GPU programming model (use CUDA, HIP, SYCL)
- A replacement for NumPy/Julia in data science
- A systems language for OS/driver development

### The Niche

Rake targets the **10-20% of code that consumes 80-90% of cycles** in compute-bound applications—specifically, code where auto-vectorization fails due to conditional logic:

| Domain | Why Rails Win |
|--------|---------------|
| Ray tracing | Rays hit different objects; BVH traversal branches unpredictably |
| Particle systems | Boundary conditions, life/death, collision responses |
| Game physics | Constraint solving with conditional contact points |
| Audio DSP | Voice stealing, filter mode switching, envelope stages |
| ML inference | Activation functions with branches, attention masking |

### Competitive Position

| Language | Approach | Rake's Edge |
|----------|----------|-------------|
| **ISPC** | Implicit vectorization, `uniform`/`varying` | Complete language (not kernel-only), cleaner rail syntax |
| **Rust std::simd** | Explicit SIMD types, manual predication | First-class rails, less boilerplate |
| **Zig @Vector** | Built-in vector types | Higher-level abstraction for divergent code |
| **C intrinsics** | Manual everything | Readable, portable, maintainable |

ISPC is the closest competitor. Rake differentiates through:
1. ML-inspired syntax (rails as pattern matching)
2. Complete language (no C++ host required)
3. MLIR backend (leverage existing optimization infrastructure)

---

## Current State

### What Works

| Component | Status | Quality |
|-----------|--------|---------|
| Lexer | Complete | Production |
| Parser | Complete | Production |
| Type checker | Partial | Functional |
| MLIR emitter | Complete | Functional |
| LLVM lowering | Complete | Functional |
| Native codegen | Complete | Functional |

### Performance Gap

```
Benchmark: 1M particles, 100 iterations
─────────────────────────────────────────
Rake (MLIR→LLVM):     453.7 M particles/sec
C (auto-vectorized):  475.8 M particles/sec
─────────────────────────────────────────
Gap:                  ~5% slower
```

**This gap is unacceptable.** Nobody adopts a new language to be slower.

### Gap Analysis

The 5% comes from:

1. **Function call overhead** (3-4%): Each `crunch`/`rake` emits as a separate function. LLVM doesn't inline across the MLIR boundary without LTO.

2. **No pipeline fusion** (1-2%): `a |> f |> g |> h` creates intermediate values instead of fusing into one operation.

3. **Conservative MLIR lowering**: Default passes don't exploit Rake's semantic guarantees (no aliasing, known layouts).

---

## The Path Forward

### Phase 1: Prove the Concept

**Objective**: Demonstrate that rails beat auto-vectorization on divergent workloads.

**Deliverables**:
1. Close the 5% gap (pipeline fusion + inlining)
2. Implement ray tracer demo
3. Beat C by 20%+ on ray tracing

**Why Ray Tracing**: It's the perfect showcase because:
- Every ray diverges (hits different objects)
- BVH traversal has unpredictable branches
- Auto-vectorizers fail catastrophically here
- Visual output is immediately compelling
- Well-understood benchmark for comparison

**Technical Work**:

| Task | Impact | Complexity |
|------|--------|------------|
| Pipeline fusion (`\|>` chains) | 20-40% on hot paths | Medium |
| Function inlining (crunch → inline) | 5-15% | Low |
| Vec3 operations (dot, cross, normalize) | Enables ray tracer | Low |
| Ray-sphere/plane/box intersection | Core demo | Medium |
| BVH traversal with rail-based selection | Killer demo | High |

### Phase 2: Make It Usable

**Objective**: A developer can write real code without constant friction.

**Deliverables**:
1. Standard library (vec3, mat4, common math)
2. Useful error messages with source locations
3. Basic module system (import/export)
4. C FFI (embed Rake in existing projects)

**Technical Work**:

| Task | Impact | Complexity |
|------|--------|------------|
| Vec module (dot, cross, length, normalize) | High | Low |
| Mat module (mul, transform, inverse) | Medium | Medium |
| Error recovery in parser | Medium | Medium |
| Source location tracking | High | Low |
| Module system design | High | High |
| C calling convention | High | Medium |

### Phase 3: Make It Adoptable

**Objective**: External developers can evaluate and adopt Rake.

**Deliverables**:
1. Multiple SIMD targets (AVX-512, NEON)
2. LSP for IDE support
3. Documentation that doesn't require reading source
4. Benchmark suite with reproducible results

**Technical Work**:

| Task | Impact | Complexity |
|------|--------|------------|
| AVX-512 backend (16-wide) | Medium | Low |
| NEON backend (ARM) | Medium | Medium |
| LSP server | High | High |
| Tutorial documentation | High | Medium |
| Automated benchmark CI | Medium | Low |

---

## MLIR Strategy

The decision to target MLIR instead of emitting LLVM directly is correct. MLIR provides:

1. **Vector dialect**: First-class SIMD operations
2. **Transform dialect**: Pattern-based optimization
3. **Affine dialect**: Polyhedral loop optimization
4. **Multiple backends**: CPU, GPU, TPU paths

### Current Dialect Usage

```
Rake source
    ↓
func + arith + vector + math (current)
    ↓
llvm dialect
    ↓
LLVM IR → native
```

### Target Dialect Stack

```
Rake source
    ↓
scf + vector + arith + math (structured control flow)
    ↓ [loop fusion, tiling]
vector + arith
    ↓ [lowering]
llvm dialect
    ↓
LLVM IR → native
```

### Key Optimizations by Dialect

| Dialect | Optimization | Expected Gain |
|---------|--------------|---------------|
| SCF | Loop fusion across `sweep` | 10-30% |
| Transform | Pipeline operator fusion | 20-40% |
| Affine | Cache-aware tiling | 15-25% |
| Vector | Mask simplification | 5-10% |

---

## Language Guarantees That Enable Optimization

Rake code has properties that C/Rust cannot guarantee:

1. **No aliasing**: All values are semantically immutable within `rake`/`crunch`. The compiler can always parallelize without alias checks.

2. **Known memory layout**: `stack` enforces SoA. The compiler knows at compile time that all x-components are contiguous.

3. **Predication, not branching**: Rails guarantee all paths execute. No branch misprediction, no speculation.

4. **Whole-program visibility**: No separate compilation (initially). The compiler sees everything.

These enable optimizations that are unsafe or impossible in C:
- Aggressive reordering without alias analysis
- Static gather pattern generation
- Cross-function fusion without interprocedural analysis

---

## Risk Assessment

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Performance ceiling (can't beat C) | Medium | Fatal | Focus on divergent workloads where C fails |
| MLIR breaking changes | Low | High | Pin LLVM version, abstract emission layer |
| Complexity explosion in type system | Medium | Medium | Keep types simple; defer generics |

### Adoption Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| No killer demo | High | Fatal | Ray tracer is existential priority |
| Learning curve too steep | Medium | High | Familiar ML syntax; good docs |
| No ecosystem | High | Medium | C FFI for gradual adoption |

---

## Success Criteria

### Phase 1 Success (Proof of Concept)

- [ ] Ray tracer compiles and runs
- [ ] Ray tracer beats auto-vectorized C by 20%+
- [ ] Performance gap on particles benchmark < 2%

### Phase 2 Success (Usable)

- [ ] 3 non-trivial programs written by non-authors
- [ ] Error messages point to correct source locations
- [ ] C FFI works for embedding Rake functions

### Phase 3 Success (Adoptable)

- [ ] Someone outside the project writes a blog post
- [ ] Conference talk or paper accepted
- [ ] One production use case (even if small)

---

## What We're Not Doing

Explicit non-goals to maintain focus:

1. **GPU targeting** — CPU SIMD is hard enough. GPU comes later (if ever).
2. **Generics** — Monomorphic code is simpler. Add generics when there's pain.
3. **Formal verification** — Interesting but not the bottleneck.
4. **Distributed execution** — Out of scope entirely.
5. **IDE plugins** — LSP first; editor-specific plugins never.
6. **Package manager** — Premature. Single-file programs are fine for now.

---

## The Vocabulary Is the Product

The terms `rake`, `rail`, `rack`, `sweep`, `spread`, `crunch`, `retire`, `compact` aren't whimsy. They're cognitive tools that reinforce the parallel mental model.

Traditional keywords carry scalar baggage:
- `if` implies one path executes
- `for` implies sequential iteration
- `array` implies indexed access

Rake's vocabulary forces parallel thinking:
- `rail` implies all paths execute, masks select
- `sweep` implies parallel processing of chunks
- `rack` implies SIMD register, not array

This vocabulary IS the product. A Rake programmer thinks differently than a C programmer, and the language enforces this through every keyword.

---

## Conclusion

Rake exists because **divergent control flow kills auto-vectorization**, and rails are the answer.

The path to success:
1. **Prove it works**: Ray tracer that beats C
2. **Make it usable**: Vec3/mat4, error messages, modules
3. **Make it adoptable**: Multiple targets, LSP, docs

Everything else is distraction.

---

*Version 3.0 — December 2024*
*Focus: Existential proof via ray tracer demo*
