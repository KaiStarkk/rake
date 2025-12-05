# Tines and Through Blocks

## The Core Abstraction

Tines and `through` blocks are Rake's answer to divergent control flow.
Instead of branches (which break vectorization), Rake uses:

1. **Tine declarations** — compute masks once, give them names
2. **Through blocks** — execute under a mask, inactive lanes get passthru
3. **Sweep blocks** — select final results based on which tines matched

## Tine Declarations

A tine declares a named mask:

```rake
| tine positive := x > <0.0>
| tine negative := x < <0.0>
| tine zero := x = <0.0>
```

**Syntax**: `| tine <name> := <predicate>`

The `|` character represents a horizontal bar — the teeth of the rake.
Multiple tines stack vertically, defining the filter pattern.

**MLIR mapping**: Each tine becomes a comparison producing `vector<Nxi1>`

### Tine Composition

Tines can reference other tines:

```rake
| tine a := x > <0.0>
| tine b := y > <0.0>
| tine both := a && b       -- intersection
| tine either := a || b     -- union
| tine only_a := a && !b    -- difference
```

### Predicate Syntax

| Syntax | Meaning |
|--------|---------|
| `x > y` | Greater than |
| `x >= y` | Greater or equal |
| `x < y` | Less than |
| `x <= y` | Less or equal |
| `x = y` | Equal |
| `x != y` | Not equal |
| `a && b` | Logical AND |
| `a \|\| b` | Logical OR |
| `!a` | Logical NOT |
| `x is <val>` | Identity comparison |
| `x is not <val>` | Negated identity |

## Through Blocks

A `through` block executes under a mask:

```rake
through positive:
  sqrt x
-> sqrt_result
```

**Semantics**:
- All lanes execute the computation (SIMD is SPMD)
- Only active lanes (where mask is true) contribute to result
- Inactive lanes receive passthru value

### With Passthru

Specify what inactive lanes receive:

```rake
through positive else <0.0>:
  sqrt x
-> result
```

If omitted, inactive lanes hold undefined values (compiler may optimize).

### Multi-Statement Through

```rake
through hit else no_hit:
  point <- ray.origin + t * ray.dir
  normal <- normalize (point - sphere.center)
  Hit { t := t, point := point, normal := normal }
-> hit_result
```

### Composed Tines

Apply multiple tines inline:

```rake
through (a && b):
  expensive_op x
-> result
```

### MLIR Mapping

```rake
through mask else passthru:
  computation
-> result
```

Becomes:

```mlir
%result = vector.mask %mask, %passthru {
  // computation
  vector.yield %value
} : vector<8xi1> -> result_type
```

## Sweep Blocks

A `sweep` block collects results from different tines:

```rake
sweep:
  | positive -> pos_result
  | negative -> neg_result
  | zero -> zero_result
-> final
```

**Semantics**: For each lane, select the result from the first matching tine.

### Exhaustiveness

Tines in a sweep should be exhaustive (cover all lanes). If not, unmatched
lanes hold undefined values.

Add a catch-all with `_`:

```rake
sweep:
  | positive -> pos_result
  | negative -> neg_result
  | _ -> default_result
-> final
```

### MLIR Mapping

```rake
sweep:
  | a -> va
  | b -> vb
  | c -> vc
-> result
```

Becomes nested `arith.select`:

```mlir
%tmp1 = arith.select %c, %vc, %undef : vector<8xi1>, type
%tmp2 = arith.select %b, %vb, %tmp1 : vector<8xi1>, type
%result = arith.select %a, %va, %tmp2 : vector<8xi1>, type
```

## Complete Example

```rake
rake classify x -> category:
  -- ═══════════════════════════════════════════════
  -- TINE DECLARATIONS (compute masks)
  --
  --   lanes:    0   1   2   3   4   5   6   7
  --   ─────────────────────────────────────────
  --   positive: █   ░   █   █   ░   ░   █   ░
  --   negative: ░   █   ░   ░   █   █   ░   ░
  --   zero:     ░   ░   ░   ░   ░   ░   ░   █
  --
  -- ═══════════════════════════════════════════════
  | tine positive := x > <0.0>
  | tine negative := x < <0.0>
  | tine zero := x = <0.0>

  -- ═══════════════════════════════════════════════
  -- THROUGH BLOCKS (masked computation)
  --
  --       ↓↓↓↓↓↓↓↓  (data enters)
  --   ════════════════════
  --       ↓ ↓↓  ↓    (positive lanes compute)
  --   ════════════════════
  --
  -- ═══════════════════════════════════════════════
  through positive:
    <1>
  -> pos_val

  through negative:
    <-1>
  -> neg_val

  through zero:
    <0>
  -> zero_val

  -- ═══════════════════════════════════════════════
  -- SWEEP (collect results)
  -- ═══════════════════════════════════════════════
  sweep:
    | positive -> pos_val
    | negative -> neg_val
    | zero -> zero_val
  -> category
```

## Why This Design?

Traditional auto-vectorizers fail on:

```c
for (int i = 0; i < n; i++) {
    if (x[i] > 0) result[i] = sqrt(x[i]);
    else result[i] = 0;
}
```

The branch prevents vectorization. Compilers either:
- Give up and use scalar code
- Generate expensive predicated code with poor performance

Rake makes predication explicit and first-class:

```rake
rake safe_sqrt x -> result:
  | tine valid := x > <0.0>

  through valid else <0.0>:
    sqrt x
  -> result
```

All lanes compute `sqrt`. The mask selects which results to keep.
No branches. No speculation. Clean SIMD.
