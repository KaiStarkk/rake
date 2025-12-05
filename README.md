# Rake

**A vector-first language for CPU SIMD with divergent control flow.**

Rake achieves **3.26x speedup** over auto-vectorized C code by making SIMD explicit in the language design, not an afterthought.

```rake
~~ Ray-sphere intersection: clean code, no intrinsics, full SIMD performance
rake intersect ray_ox ray_oy ray_oz ray_dx ray_dy ray_dz
  <sphere_cx> <sphere_cy> <sphere_cz> <sphere_r>
  -> t_result:

  let disc = b * b - <4.0> * a * c

  | #miss := (disc < <0.0>)
  | #hit  := (!#miss)

  through #hit:
    let sqrt_disc = sqrt(disc)
    (- b - sqrt_disc) / (<2.0> * a)
  -> t_value

  through #miss:
    <-1.0>
  -> miss_value

  sweep:
    | #miss -> miss_value
    | #hit  -> t_value
  -> t_result
```

## Why Rake?

Modern CPUs have 256-512 bit SIMD registers. A single instruction can process 8 floats at once. But existing languages treat SIMD as an optimization target, not a first-class concept:

| Approach | Problem |
|----------|---------|
| Auto-vectorization | Fragile. Minor code changes break it. Can't handle divergent control flow. |
| Intrinsics | Unreadable. Ties code to specific instruction sets. |
| ISPC | Better, but still C-like. Implicit divergence handling. |

**Rake makes SIMD explicit.** Every value is either a *rack* (vector across lanes) or a *scalar* (uniform). Control flow divergence is expressed through *tines*—named lane masks that partition your data.

## Key Concepts

### Racks and Scalars

```rake
~~ Rack: 8 floats, one per SIMD lane
let positions = ray_ox

~~ Scalar: broadcast to all lanes
let radius = <sphere_r>
```

### Tines: Named Lane Partitions

```rake
| #positive := (x >= <0.0>)
| #negative := (!#positive)
```

Tines are compile-time masks. No runtime branching—just masked operations.

### Through Blocks: Divergent Computation

```rake
through #positive:
  sqrt(x)
-> positive_result

through #negative:
  <0.0>
-> negative_result
```

Each `through` block executes under its tine's mask. All lanes compute in parallel; masked lanes get identity values.

### Sweep: Collect Results

```rake
sweep:
  | #positive -> positive_result
  | #negative -> negative_result
-> final_result
```

`sweep` combines results from different tines using masked selection.

## Performance

Ray-sphere intersection benchmark (8M rays, 50% hit rate):

| Implementation | Throughput | Speedup |
|---------------|------------|---------|
| C (auto-vectorized, `-O3 -march=native`) | 207.89 M rays/sec | 1.00x |
| **Rake** | 677.89 M rays/sec | **3.26x** |

Rake generates clean AVX2 assembly with:
- Zero function call overhead (inlined crunches)
- Optimal mask handling via `vblendvps`
- No unnecessary memory traffic

## Installation

### Prerequisites
- OCaml 5.0+ with dune
- LLVM 17+ (for `mlir-opt`, `mlir-translate`, `llc`)

### Build
```bash
git clone https://github.com/KaiStarkk/rake-lang
cd rake-lang
dune build
```

### Compile a Rake program
```bash
./scripts/compile.sh examples/intersect_flat.rk
# Output: intersect_flat.o (object file with AVX2 code)
```

## Function Types

- **crunch**: Pure vector computation. Always gets inlined.
- **rake**: Divergent computation with tines/through/sweep.
- **run**: Sequential orchestration (for control flow that can't be vectorized).

## Syntax Highlighting

Install the VS Code extension:
```bash
code --install-extension rake-lang-0.2.0.vsix
```

Or use the [tree-sitter grammar](https://github.com/KaiStarkk/tree-sitter-rake) for other editors.

## Status

Rake 0.2.0 is a working prototype. The compiler pipeline:

```
.rk source → Parser (Menhir) → Type Checker → MLIR → LLVM IR → Native Code
```

Current limitations:
- Single translation unit
- No standard library yet
- AVX2 only (AVX-512 planned)

## License

MIT

## Acknowledgments

Built with OCaml, MLIR, and LLVM. Inspired by ISPC, APL, and the dream of making SIMD accessible.
