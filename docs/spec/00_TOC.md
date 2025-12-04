# Rake Language Specification

## Table of Contents

**Version 0.3 — Focused Draft**

---

### Part I: Foundation

1. **Philosophy**
   - 1.1 The Problem: Auto-vectorization Fails on Divergent Code
   - 1.2 The Solution: Rails as First-Class Predication
   - 1.3 Design Principles (7 core tenets)
   - 1.4 Target Domains and Non-Goals

2. **Execution Model**
   - 2.1 Racks, Lanes, and the SPMD Model
   - 2.2 Rails: Predication, Not Branching
   - 2.3 Stacks: Structure-of-Arrays by Default
   - 2.4 Packs: Collections for Large Datasets

### Part II: Core Language

3. **Lexical Structure**
   - 3.1 Tokens and Keywords
   - 3.2 The `<name>` Scalar Convention
   - 3.3 Ligatures: `->`, `<-`, `|>`, `<|`, `:=`
   - 3.4 Comments (ML-style)

4. **Types**
   - 4.1 Rack Types (`float rack`, `int rack`, `bool rack`)
   - 4.2 Compound Racks (`vec3 rack`, `mat4 rack`)
   - 4.3 Scalar Types (marked with `<>`)
   - 4.4 Structure Types (`stack`, `aos`, `single`)
   - 4.5 Collection Types (`pack`, `array`)
   - 4.6 Mask Type (predicate results)

5. **Expressions**
   - 5.1 Literals and Identifiers
   - 5.2 Binary and Unary Operators
   - 5.3 Let Bindings (`let x = e in body`)
   - 5.4 Field Access and Functional Update
   - 5.5 Pipeline Operator (`|>`)
   - 5.6 Function Calls

6. **Rails**
   - 6.1 Rail Syntax (`| pred -> expr`)
   - 6.2 Named Rails (`| name := pred -> expr`)
   - 6.3 Predicates (`is`, `is not`, `and`, `or`, `not`)
   - 6.4 The `otherwise` Catch-All
   - 6.5 Mutual Exclusivity and Exhaustiveness

7. **Definitions**
   - 7.1 `stack` — Parallel Data (SoA with racks)
   - 7.2 `aos` — Interop Data (AoS)
   - 7.3 `single` — Configuration (all scalars)
   - 7.4 `crunch` — Pure Single-Rail Functions
   - 7.5 `rake` — Multi-Rail Parallel Functions
   - 7.6 `run` — Sequential Composition

### Part III: Iteration and Control

8. **Iteration Constructs**
   - 8.1 `sweep pack -> item: body`
   - 8.2 `repeat <n> times: body`
   - 8.3 `repeat until <cond>: body`
   - 8.4 `spread pack across cores -> chunk: body`

9. **Lane Control**
   - 9.1 `retire` — Deactivate Lane
   - 9.2 `halt` — Stop All Lanes
   - 9.3 `compact pack` — Remove Retired Lanes

10. **Lane Operations**
    - 10.1 Queries: `lanes()`, `lane_index()`, `lead()`
    - 10.2 Predicates: `any()`, `all()`, `none()`, `tally()`
    - 10.3 Reductions: `reduce(op, expr)`
    - 10.4 Shuffles: `shuffle`, `rotate`, `shift`, `broadcast`
    - 10.5 Memory: `gather`, `scatter`
    - 10.6 Compression: `compress`, `expand`

### Part IV: Standard Library

11. **Math Module**
    - 11.1 Transcendentals (`sin`, `cos`, `exp`, `log`, etc.)
    - 11.2 Utility (`abs`, `floor`, `ceil`, `clamp`, `min`, `max`)
    - 11.3 Fast Approximations

12. **Vec Module**
    - 12.1 `dot`, `cross`, `length`, `normalize`
    - 12.2 `reflect`, `refract`
    - 12.3 Component-wise operations

13. **Mat Module**
    - 13.1 Matrix multiplication
    - 13.2 Vector transformation
    - 13.3 Construction functions

### Part V: System Interface

14. **Foreign Function Interface**
    - 14.1 C Interoperability
    - 14.2 Calling Conventions
    - 14.3 Type Mapping

15. **Compilation Model**
    - 15.1 MLIR Emission
    - 15.2 Target Architectures (AVX2, AVX-512, NEON)
    - 15.3 Optimization Passes

### Appendices

**A. Keyword Reference** — Alphabetical listing with brief descriptions

**B. Operator Precedence** — Complete precedence table

**C. Glossary** — Definitions: rack, rail, lane, mask, broadcast, etc.

**D. Example Programs**
- D.1 Particle System
- D.2 Ray Tracer (the killer demo)
- D.3 Mandelbrot Renderer

**E. Design Rationale** — Why these choices were made

---

## Document Status

| Part | Status | Priority |
|------|--------|----------|
| Part I: Foundation | 70% | Critical |
| Part II: Core Language | 40% | Critical |
| Part III: Iteration | 20% | High |
| Part IV: Standard Library | 5% | Medium |
| Part V: System Interface | 10% | Medium |
| Appendices | 10% | Low |

---

*Focus principle: Specify what's implemented. Defer what's aspirational.*
