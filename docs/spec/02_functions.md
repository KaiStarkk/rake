# Functions

## Crunch: Pure Vector Functions

A `crunch` is a pure function where all lanes execute identical logic.
No divergence, no tines needed.

```rake
crunch dot ax ay az bx by bz -> d:
  d <- ax * bx + ay * by + az * bz

crunch magnitude x y z -> m:
  m <- sqrt(x * x + y * y + z * z)

crunch normalize x y z -> (nx, ny, nz):
  m <- magnitude x y z
  inv <- <1.0> / m
  nx <- x * inv
  ny <- y * inv
  nz <- z * inv
```

**Properties**:
- Always inlined at call sites
- No runtime overhead
- Parameters bind directly (no `let` needed)
- Result bound to name after `->`, yielded at end

**Syntax**:
```
crunch <name> <params> -> <result>:
  <body>
```

## Rake: Divergent Vector Functions

A `rake` is a function where lanes may diverge. Tines define masks,
`through` blocks execute under masks, `sweep` collects results.

```rake
rake safe_sqrt x -> result:
  | tine valid := x >= <0.0>
  | tine invalid := !valid

  through valid:
    sqrt x
  -> valid_result

  sweep:
    | valid -> valid_result
    | invalid -> <0.0>
  -> result
```

**Structure**:
```
rake <name> <params> -> <result>:
  | tine <name> := <predicate>    -- declare masks
  ...

  through <tine>:                  -- masked computation
    <body>
  -> <binding>

  sweep:                           -- collect results
    | <tine> -> <value>
    ...
  -> <result>
```

## Run: Pack Iteration Entry Point

A `run` function is the entry point for processing packs of data. The `over` construct
iterates over pack data in SIMD-width chunks with automatic tail masking.

```rake
~~ Process rays through a raytracer
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

**The `over` construct**:
- `over pack, count |> binding:` — iterate over pack in SIMD-width chunks
- Automatically handles tail masking for non-multiple-of-8 counts
- Pack fields become vector loads from the corresponding memrefs
- Results are stored via masked stores to output buffer

**Generated MLIR for `over`**:
```mlir
scf.for %i = %zero to %num_iters step %one {
  %offset = arith.muli %i, %lanes : index
  %remaining = arith.subi %count, %offset : index
  %mask = vector.create_mask %remaining : vector<8xi1>

  // Load pack fields
  %ox = vector.load %rays_ox[%offset] : memref<?xf32>, vector<8xf32>
  %oy = vector.load %rays_oy[%offset] : memref<?xf32>, vector<8xf32>
  // ... other fields

  // Compute
  %result = call @intersect_flat(...) : (...) -> vector<8xf32>

  // Masked store
  vector.maskedstore %output[%offset], %mask, %result
}
```

**Pack type expansion**:
When a pack type is used as a parameter, it expands to separate memrefs for each field:
- `(rays : Ray pack)` → `memref<?xf32>` for each field (ox, oy, oz, dx, dy, dz)

## Parameter Syntax

**Rack parameters** (default):
```rake
crunch add x y -> z:    -- x, y are racks
```

**Scalar parameters** (angle brackets):
```rake
crunch scale x <factor> -> z:   -- factor is scalar
  z <- x * <factor>
```

**Typed parameters**:
```rake
crunch process (x : float rack) (y : int rack) -> z:
```

## Result Syntax

**Single result**:
```rake
crunch square x -> y:
  y <- x * x
```

**Multiple results (tuple)**:
```rake
crunch split x -> (lo, hi):
  lo <- x - <0.5>
  hi <- x + <0.5>
```

**Typed result**:
```rake
crunch compute x -> (y : float rack):
  y <- x * <2.0>
```
