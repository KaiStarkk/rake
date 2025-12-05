# Types

## Primitive Types

| Type | Description | MLIR Type |
|------|-------------|-----------|
| `float` | 32-bit floating point | `f32` |
| `double` | 64-bit floating point | `f64` |
| `int` | 32-bit signed integer | `i32` |
| `int8` | 8-bit signed integer | `i8` |
| `int16` | 16-bit signed integer | `i16` |
| `int64` | 64-bit signed integer | `i64` |
| `uint` | 32-bit unsigned integer | `i32` |
| `bool` | Boolean (1-bit) | `i1` |

## Compound Types

| Type | Description | MLIR Type |
|------|-------------|-----------|
| `vec2` | 2-component vector | `vector<2xf32>` |
| `vec3` | 3-component vector | `vector<3xf32>` |
| `vec4` | 4-component vector | `vector<4xf32>` |
| `mat3` | 3x3 matrix | `vector<9xf32>` |
| `mat4` | 4x4 matrix | `vector<16xf32>` |

## Rack Types (Parallel)

A `rack` is a vector of values, one per SIMD lane:

```rake
x : float rack    -- 8 floats (AVX2), one per lane
v : vec3 rack     -- 8 vec3s, one per lane (24 floats total)
alive : bool rack -- 8 booleans, one per lane (mask)
```

**MLIR mapping**: `float rack` → `vector<8xf32>` (width is configurable)

## Scalar Types (Uniform)

A scalar is a single value, uniform across all lanes:

```rake
<threshold> : float    -- one float, same for all lanes
<config> : Config      -- one Config struct
```

**Usage**: Scalars must be wrapped in `<>` when used in expressions.
This prevents accidental broadcasting without visual indication.

```rake
result <- x * <scale>   -- scale broadcasts to all lanes
result <- x * scale     -- ERROR: scale is ambiguous
```

## Stack Types (Structure-of-Arrays)

A `stack` defines parallel data with SoA layout:

```rake
stack Particle {
  pos  : vec3 rack     -- 8 positions
  vel  : vec3 rack     -- 8 velocities
  life : float rack    -- 8 lifetimes
  alive: bool rack     -- 8 alive flags
}
```

Each field is a `rack` — you have 8 particles in parallel, with their
positions contiguous, then velocities contiguous, etc.

**MLIR mapping**: Custom struct type with vector fields

## Single Types (All Scalars)

A `single` defines configuration/constants with scalar fields:

```rake
single Config {
  dt      : float
  gravity : float
  damping : float
}
```

All fields are scalars. When used in expressions, the whole struct
is accessed with `<>`:

```rake
new_vel <- vel + <cfg.gravity> * <cfg.dt>
```

## Pack Types (Collections)

A `pack` is a collection of stack chunks:

```rake
particles : Particle pack   -- many groups of 8 particles
```

You iterate over packs with `over`:

```rake
over particles -> p:
  p <- update p
```

## Mask Type

A mask is a boolean rack — the result of comparisons:

```rake
active <- x > <0.0>    -- mask: which lanes are positive
```

Masks are used with tines and `through` blocks.
