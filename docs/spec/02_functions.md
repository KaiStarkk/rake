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

## Run: Sequential Orchestration

A `run` block handles sequential composition over time:

```rake
run simulate particles <cfg> <frames> -> particles':
  repeat <frames> times:
    over particles -> p:
      p <- update_particle p <cfg>

  results in particles
```

**Constructs available in `run`**:
- `over pack -> binding:` — iterate over pack chunks
- `repeat <n> times:` — temporal iteration
- `repeat until <cond>:` — conditional iteration
- Mutable bindings with `<-`

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
