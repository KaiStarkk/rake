# Rake Language Specification v0.2.0

## The Raking Metaphor

Rake is named for its execution model: **data rakes through tine patterns**.

```
                ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓   (8 lanes of data enter)
    ════════════════════════════════════
    |  tine A |███░░░███░░░███░░░██████| (some lanes blocked)
    ════════════════════════════════════
                   ↓     ↓     ↓         (survivors continue)
    ═══════════════════════════════════
    |  tine B |░░░███░░░░░░███░░░░░░███| (different pattern)
    ═══════════════════════════════════
                         ↓     ↓         (fewer survivors)
        sweep:           <     <         (collect results)
```

Each **tine** is a horizontal barrier — the teeth of a rake — that filters lanes.
Data flows **downward** through tine declarations.
Results are **swept up** at the end.

## Core Thesis

Auto-vectorization fails on divergent code. Traditional scalar `if/else` branches
cannot map efficiently to SIMD execution because different lanes need different
code paths.

Rake inverts the model:
1. **Tines declare masks** — boolean vectors that define which lanes are active
2. **`through` blocks execute under masks** — all lanes compute, mask selects
3. **`sweep` collects results** — lanes merge based on which tines they passed

This is SIMD semantics made explicit in the language.

## Design Principles

1. **Vectors are primitive** — A `rack` is one SIMD register, not an array
2. **Scalars are marked** — `<name>` broadcasts, preventing accidental confusion.
3.   `<>` are used to visually represnt SIMD broadcast (spreads a scalar out across the whole lane, like a flood fill)
4. **Control flow is predication** — No branches; tines create masks, `through` applies them
5. **Data layout is explicit** — `stack` (SoA) vs `single` (scalars) is visible
6. **Vocabulary matches semantics** — `tine`, `rake`, `through`, `sweep` reinforce parallel thinking
7. **Vertical = parallel time** — Same-indentation operations are conceptually simultaneous
8. **Syntax serves cognition** — Visual structure implies execution model

The problem isn't that we can't write vectorized code. It's that the ergonomics in most languages for intrinsics are atrocious.
The reason for this is that vectors are a bolt-on to a fundamentally scalar mental model for legacy reasons.
Rake is a fresh start where every design decision has been made to encourage programmers to slip more naturally into a
parallel mental model.

## Vocabulary

| Term | Meaning | SIMD Analogy |
|------|---------|--------------|
| `rack` | Vector value (one per lane) | SIMD register |
| `tine` | Named mask declaration | `vector.create_mask` |
| `through` | Masked computation region | `vector.mask` |
| `sweep` | Collect results from tines | `arith.select` chain |
| `crunch` | Pure function (all lanes same) | Inlinable helper |
| `rake` | Divergent function (lanes differ) | Main computation |
| `over` | Iterate over pack chunks | Outer loop |
| `stack` | Structure-of-Arrays type | SoA layout |
| `single` | All-scalar configuration | Broadcast constants |

## File Extension

`.rk` — Rake source files
