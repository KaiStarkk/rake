# Rake

**A vector-first language for CPU SIMD with divergent control flow.**

Rake achieves **2x+ speedup** over scalar C code, matching hand-written AVX2 intrinsics—with clean, readable syntax.

Other languages can do this too with intrinsics or handrolling assembly, but Rake's purpose is to help the programmer write vectorized code without the migraine.

```rake
~~ Ray-sphere intersection: clean code, no intrinsics, full SIMD performance
rake intersect_flat ray_ox ray_oy ray_oz ray_dx ray_dy ray_dz
  <sphere_cx> <sphere_cy> <sphere_cz> <sphere_r>
  -> t_result:

  ~~ Setup: compute quadratic coefficients
  let oc_x = ray_ox - <sphere_cx>
  let oc_y = ray_oy - <sphere_cy>
  let oc_z = ray_oz - <sphere_cz>

  let a = dot(ray_dx, ray_dy, ray_dz, ray_dx, ray_dy, ray_dz)
  let b = <2.0> * dot(oc_x, oc_y, oc_z, ray_dx, ray_dy, ray_dz)
  let c = dot(oc_x, oc_y, oc_z, oc_x, oc_y, oc_z) - <sphere_r> * <sphere_r>
  let disc = b * b - <4.0> * a * c

  ~~ Tines: partition lanes by hit/miss
  | #miss := (disc < <0.0>)
  | #hit  := (!#miss)

  ~~ Through: compute only for hit lanes
  through #hit:
    let sqrt_disc = sqrt(disc)
    (neg(b) - sqrt_disc) / (<2.0> * a)
  -> t_value

  through #miss:
    <-1.0>
  -> miss_value

  ~~ Sweep: collect results from all tines
  sweep:
    | #hit  -> t_value
    | #miss -> miss_value
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

### Over: Iterate Over Packs

```rake
~~ Process all rays in a pack (automatic SIMD chunking)
run render_all (rays : Ray pack) (<count> : int64)
               (<sphere_cx> : float) (<sphere_cy> : float)
               (<sphere_cz> : float) (<sphere_r> : float)
               -> result:
  over rays, <count> |> ray:
    let t = intersect_flat(ray.ox, ray.oy, ray.oz,
                           ray.dx, ray.dy, ray.dz,
                           <sphere_cx>, <sphere_cy>,
                           <sphere_cz>, <sphere_r>)
    t
```

`over` iterates over pack data in SIMD-width chunks, automatically handling tail masking. This is the bridge between scalar control flow and vectorized computation.

## Performance

Raytracer benchmark (1920×1080, 10 spheres, 100 iterations):

| Implementation | Time/Frame | FPS | Speedup |
|---------------|------------|-----|---------|
| C Scalar | 60.90 ms | 16.4 | 1.00x |
| C SIMD (hand-written AVX2) | 29.82 ms | 33.5 | 2.04x |
| **Rake SIMD** | **29.85 ms** | **33.5** | **2.04x** |

Rake matches hand-written AVX2 intrinsics—with clean, readable syntax.

The compiler generates optimal AVX2 assembly with:
- Link-time inlining (zero function call overhead)
- Optimal mask handling via `vblendvps`
- Masked stores for correct tail handling

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
- **run**: Entry point with pack iteration via `over`.

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
