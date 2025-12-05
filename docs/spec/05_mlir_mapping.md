# MLIR Mapping

## Compilation Pipeline

```
Rake Source (.rk)
       ↓
   [Lexer + Parser]
       ↓
   Rake AST
       ↓
   [Type Checker]
       ↓
   Typed AST
       ↓
   [MLIR Emitter]
       ↓
   MLIR (func + arith + vector + math + scf)
       ↓
   [mlir-opt passes]
       ↓
   MLIR (llvm dialect)
       ↓
   [mlir-translate]
       ↓
   LLVM IR
       ↓
   [llc]
       ↓
   Native Code (AVX2/AVX-512/NEON)
```

## Dialect Usage

| MLIR Dialect | Purpose in Rake |
|--------------|-----------------|
| `func` | Function definitions, calls |
| `arith` | Arithmetic, comparisons, selects |
| `vector` | SIMD operations, masks, reductions |
| `math` | Transcendental functions |
| `scf` | Structured control flow (loops) |
| `llvm` | Final lowering target |

## Vector Width

Default: 8 (AVX2: 256-bit / 32-bit elements)

Configurable for:
- AVX-512: 16
- NEON: 4
- Scalable vectors: runtime-determined

## Type Mapping

| Rake Type | MLIR Type |
|-----------|-----------|
| `float rack` | `vector<8xf32>` |
| `int rack` | `vector<8xi32>` |
| `bool rack` / mask | `vector<8xi1>` |
| `<scalar>` (float) | `f32` |
| `vec3 rack` | `!rake.vec3rack<8>` (custom) |
| `Particle stack` | `!rake.soa_Particle` (custom) |

## Key Operation Mappings

### Tine → Mask Creation

```rake
| tine valid := x > <0.0>
```

```mlir
%threshold = arith.constant 0.0 : f32
%threshold_vec = vector.broadcast %threshold : f32 to vector<8xf32>
%valid = arith.cmpf ogt, %x, %threshold_vec : vector<8xf32>
```

### Through → vector.mask

```rake
through valid else <0.0>:
  sqrt x
-> result
```

```mlir
%zero = arith.constant 0.0 : f32
%zero_vec = vector.broadcast %zero : f32 to vector<8xf32>
%result = vector.mask %valid, %zero_vec {
  %sqrt = math.sqrt %x : vector<8xf32>
  vector.yield %sqrt : vector<8xf32>
} : vector<8xi1> -> vector<8xf32>
```

### Sweep → arith.select chain

```rake
sweep:
  | positive -> pos_val
  | negative -> neg_val
  | zero -> zero_val
-> result
```

```mlir
// Build from last to first
%tmp1 = arith.select %zero, %zero_val, %undef : vector<8xi1>, vector<8xf32>
%tmp2 = arith.select %negative, %neg_val, %tmp1 : vector<8xi1>, vector<8xf32>
%result = arith.select %positive, %pos_val, %tmp2 : vector<8xi1>, vector<8xf32>
```

### Reduction

```rake
sum <- values \+/
```

```mlir
%sum = vector.reduction <add>, %values : vector<8xf32> into f32
```

### Scan

```rake
prefix <- values \+\
```

```mlir
%init = arith.constant 0.0 : f32
%prefix = vector.scan <add>, %init, %values
    {inclusive = true} : vector<8xf32>
```

### Gather

```rake
values <- base[offsets]
```

```mlir
%values = vector.gather %base[%offsets]
    : memref<?xf32>, vector<8xi32>, vector<8xi1>, vector<8xf32>
```

### Scatter

```rake
base[offsets] <- values through mask
```

```mlir
vector.scatter %base[%offsets], %mask, %values
    : memref<?xf32>, vector<8xi32>, vector<8xi1>, vector<8xf32>
```

### Compress Store

```rake
base[ptr] <-| values through mask
```

```mlir
vector.compressstore %base[%ptr], %mask, %values
    : memref<?xf32>, vector<8xi1>, vector<8xf32>
```

### Shuffle

```rake
shuffled <- v ~> [3, 2, 1, 0, 7, 6, 5, 4]
```

```mlir
%shuffled = vector.shuffle %v, %v [3, 2, 1, 0, 7, 6, 5, 4]
    : vector<8xf32>, vector<8xf32>
```

### Over → scf.for with masked stores

```rake
over rays, <count> |> ray:
  compute(ray.ox, ray.oy, ...)
```

```mlir
%lanes = arith.constant 8 : index
%zero = arith.constant 0 : index
%one = arith.constant 1 : index
%count_idx = arith.index_cast %count : i64 to index
%lanes_m1 = arith.constant 7 : index
%count_plus = arith.addi %count_idx, %lanes_m1 : index
%niters = arith.divui %count_plus, %lanes : index

scf.for %i = %zero to %niters step %one {
  %offset = arith.muli %i, %lanes : index
  %remaining = arith.subi %count_idx, %offset : index
  %mask = vector.create_mask %remaining : vector<8xi1>

  // Load fields from pack memrefs
  %ox = vector.load %rays_ox[%offset] : memref<?xf32>, vector<8xf32>
  %oy = vector.load %rays_oy[%offset] : memref<?xf32>, vector<8xf32>
  // ...

  // Call computation
  %result = func.call @compute(%ox, %oy, ...) : (...) -> vector<8xf32>

  // Masked store to output
  vector.maskedstore %output[%offset], %mask, %result
      : memref<?xf32>, vector<8xi1>, vector<8xf32>
}
```

### FMA

```rake
result <- fma(a, b, c)
```

```mlir
%result = vector.fma %a, %b, %c : vector<8xf32>
```

## Optimization Opportunities

### Pipeline Fusion

```rake
x |> f |> g |> h
```

Instead of:
```mlir
%t1 = call @f(%x)
%t2 = call @g(%t1)
%t3 = call @h(%t2)
```

Inline and fuse:
```mlir
// All operations inline, no intermediate allocations
```

### Tine Sharing

Multiple `through` blocks using the same tine don't recompute the mask:

```rake
| tine active := condition

through active:
  op1
-> r1

through active:
  op2
-> r2
```

The `active` mask is computed once and reused.

### SoA Layout Exploitation

Stack types guarantee contiguous field layout:

```rake
stack Particle { pos: vec3 rack, vel: vec3 rack }
```

All `pos.x` values are contiguous, enabling:
- Aligned vector loads
- Prefetching
- Cache-efficient access

### Known Aliasing

Rake's type system prevents aliasing:
- Stacks don't alias other stacks
- Scalars don't alias racks
- Enables aggressive optimization without alias analysis

## Custom Rake Dialect (Future)

For advanced optimizations, a custom MLIR dialect could provide:

```mlir
// Tine as first-class value
%valid = rake.tine %x > %threshold : vector<8xf32> -> !rake.mask

// Through with explicit passthru
%result = rake.through %valid, %passthru {
  %sqrt = math.sqrt %x : vector<8xf32>
  rake.yield %sqrt
} : !rake.mask, vector<8xf32> -> vector<8xf32>

// Sweep with pattern matching
%final = rake.sweep {
  ^positive(%pos: vector<8xf32>):
    rake.yield %pos_val
  ^negative(%neg: vector<8xf32>):
    rake.yield %neg_val
} when [%positive, %negative] : vector<8xf32>
```

This would enable:
- Custom lowering strategies
- Rake-specific optimizations
- Better error messages
