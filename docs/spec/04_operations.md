# Operations

## Arithmetic

Standard operators work on racks (element-wise):

```rake
a + b       -- addition
a - b       -- subtraction
a * b       -- multiplication
a / b       -- division
a % b       -- modulo
-a          -- negation
```

## Comparison (Mask Creation)

Comparisons produce masks:

```rake
a > b       -- greater than
a >= b      -- greater or equal
a < b       -- less than
a <= b      -- less or equal
a = b       -- equal
a != b      -- not equal
```

## Logical (Mask Composition)

```rake
a && b      -- AND
a || b      -- OR
!a          -- NOT
```

## Reductions (Lanes → Scalar)

Reductions collapse all lanes to a single value:

```rake
x \+/       -- sum: x₀ + x₁ + ... + x₇
x \*/       -- product: x₀ * x₁ * ... * x₇
x \min/     -- minimum: min(x₀, x₁, ..., x₇)
x \max/     -- maximum: max(x₀, x₁, ..., x₇)
mask \|/    -- any: true if any lane is true
mask \&/    -- all: true if all lanes are true
```

The `\ /` ligature suggests convergence — lanes folding inward.

**MLIR mapping**: `vector.reduction <op>`

## Scans (Prefix Operations)

Scans compute running totals across lanes:

```rake
x \+\       -- prefix sum: [x₀, x₀+x₁, x₀+x₁+x₂, ...]
x \*\       -- prefix product
x \min\     -- prefix min
x \max\     -- prefix max
```

The `\ \` ligature suggests a wave moving rightward.

**MLIR mapping**: `vector.scan <op>`

## Lane Access

```rake
@           -- lane indices: [0, 1, 2, 3, 4, 5, 6, 7]
lanes       -- vector width (scalar): 8

v@3         -- extract lane 3 (scalar)
v@3 := x    -- insert x at lane 3

v@(0..4)    -- extract lanes 0-3 (half-width vector)
```

**MLIR mapping**: `vector.extract`, `vector.insert`, `vector.step`

## Shuffle Operations

```rake
v ~> [3,2,1,0,7,6,5,4]   -- permute by index pattern
v >> 2                    -- shift right by 2 (zeros enter left)
v << 2                    -- shift left by 2 (zeros enter right)
v >>> 2                   -- rotate right by 2
v <<< 2                   -- rotate left by 2
```

**MLIR mapping**: `vector.shuffle`

## Interleave / Deinterleave

```rake
a >< b              -- interleave: [a₀,b₀,a₁,b₁,a₂,b₂,...]
(evens, odds) <- v ><   -- deinterleave
```

**MLIR mapping**: `vector.interleave`, `vector.deinterleave`

## Memory Operations

### Gather (Indexed Load)

```rake
values <- base[offsets]              -- gather from base + offsets
values <- base[offsets] through mask -- masked gather
```

**MLIR mapping**: `vector.gather`

### Scatter (Indexed Store)

```rake
base[offsets] <- values              -- scatter to base + offsets
base[offsets] <- values through mask -- masked scatter
```

**MLIR mapping**: `vector.scatter`

### Compress / Expand

```rake
-- Compress: pack active lanes contiguously
packed <- values |> compress through mask
base[ptr] <-| values through mask    -- compress store

-- Expand: unpack contiguous into active lanes
expanded <- |-> base[ptr] through mask else passthru
```

**MLIR mapping**: `vector.compressstore`, `vector.expandload`

## Fused Multiply-Add

```rake
fma(a, b, c)    -- a * b + c (fused, single rounding)
```

**MLIR mapping**: `vector.fma`

## Outer Product

```rake
a outer b       -- outer product (for matrix ops)
```

**MLIR mapping**: `vector.outerproduct`

## Math Functions

Built-in functions (vectorized):

```rake
sqrt x      -- square root
exp x       -- exponential
log x       -- natural log
sin x       -- sine
cos x       -- cosine
tan x       -- tangent
abs x       -- absolute value
floor x     -- floor
ceil x      -- ceiling
min a b     -- element-wise minimum
max a b     -- element-wise maximum
pow a b     -- power
```

**MLIR mapping**: `math.*` operations

## Pipeline Operator

Chain operations left-to-right:

```rake
x |> normalize |> clamp |> scale
-- equivalent to: scale(clamp(normalize(x)))
```

Pipelines are fused — no intermediate allocations.

## Record Operations

### Field Access

```rake
particle.pos    -- access pos field of particle
particle.vel.x  -- nested access
```

### Record Update

```rake
{ particle with pos := new_pos }
{ particle with pos := new_pos, vel := new_vel }
```

### Record Construction

```rake
Particle { pos := p, vel := v, life := l, alive := true }
```

## Summary: Operation → MLIR Mapping

| Rake Syntax | MLIR Operation |
|-------------|----------------|
| `<scalar>` | `vector.broadcast` |
| `\| tine := pred` | `arith.cmp*` |
| `through tine:` | `vector.mask` |
| `sweep:` | `arith.select` chain |
| `x \+/` | `vector.reduction <add>` |
| `x \+\` | `vector.scan <add>` |
| `base[idx]` | `vector.gather` |
| `base[idx] <- v` | `vector.scatter` |
| `v \|> compress` | `vector.compressstore` |
| `v ~> [...]` | `vector.shuffle` |
| `v@i` | `vector.extract` |
| `v@i := x` | `vector.insert` |
| `a >< b` | `vector.interleave` |
| `fma(a,b,c)` | `vector.fma` |
| `a outer b` | `vector.outerproduct` |
