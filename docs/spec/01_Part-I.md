# The Rake Programming Language Specification

**Version 0.3 Draft**

---

# Part I: Overview

---

## Chapter 1: Introduction

### 1.1 Purpose and Scope

This document specifies the Rake programming language, a systems programming language designed for compute-intensive applications where parallel data processing is the dominant workload.

Rake differs from conventional languages in a fundamental way: the vector is the primitive unit of computation. Where other languages treat parallelism as an optimization to be discovered or requested, Rake treats it as the semantic foundation. Scalar values exist but are marked as exceptional.

The name "Rake" evokes the core metaphor: like a rake's tines working in parallel across soil, Rake programs operate on racks of values simultaneously. Each tine is a lane; the rake is the program processing them all at once.

This specification defines:

- The lexical structure and grammar of Rake programs
- The type system, including racks, scalars, stacks, and packs
- The execution model for parallel and sequential constructs
- The standard library of mathematical and I/O operations
- The foreign function interface for C interoperability
- The compilation model and target architectures

This specification does not define:

- Implementation strategies (though recommendations are provided)
- Debugging protocols (though requirements are stated)
- Editor or IDE behavior (though LSP support is expected)
- Performance guarantees (though the cost model is explicit)

### 1.2 Design Philosophy

Rake is built on seven principles:

**Vectors are primitive.**

A `float rack` is not a collection of floats. It is a single value that happens to contain multiple lanes. Operations on racks are indivisible parallel operations, not loops over elements. The programmer's mental model begins with parallelism, not with scalars that might be parallelized.

**Scalars are marked.**

Scalar identifiers are wrapped in angle brackets: `<threshold>`. When reading `values * <threshold>`, you know instantly that `<threshold>` is uniform across lanes — a single value broadcast to all lanes. The angle brackets visually suggest the value "spreading outward" to fill all lanes. There is no hidden broadcasting, no implicit scalar-to-vector conversion that might surprise you.

**Control flow is predication.**

There are no branches inside parallel operations. When different lanes require different computations, all computations execute and results merge according to masks. The rail construct — written with `|` like ML pattern matching — makes this explicit. You define which lanes go where, not which branch to take. All rails execute; masks determine which results each lane receives.

**Data layout is explicit.**

The `stack` keyword declares Structure-of-Arrays data containing racks, optimal for SIMD. The `aos` keyword declares Array-of-Structures data for interoperability. The `single` keyword declares all-scalar configuration structures. A `pack` is a collection of stacks. You always know how your data is laid out in memory.

**Vocabulary matches semantics.**

Keywords are chosen to reinforce the parallel mental model. Traditional terms like `if`, `else`, `for`, and `while` carry scalar connotations accumulated over decades. Rake uses `rail`, `rake`, `crunch`, `run`, `stack`, `pack`, `sweep`, `spread`, and `retire` — words that describe operations on collections of lanes rather than sequential control flow.

**Hardware independence with explicit cost.**

Source code does not encode vector widths. The same program compiles to 4-wide NEON, 8-wide AVX2, or 16-wide AVX-512. But operations with significant performance implications — `gather`, `scatter`, `pack` (compaction), cross-lane shuffles — are syntactically visible. The programmer sees what is cheap and what is expensive.

**Syntax serves cognition.**

Rake adopts an ML-inspired syntax: `|` for rails mirrors pattern matching, `let ... in` for bindings, `{ ... with }` for functional record updates. Parameters bind directly to `crunch` and `rake` definitions without intermediate syntax. This syntax is not merely aesthetic — it leverages decades of ergonomic refinement for expressing transformations on structured data.

### 1.3 Target Domains

Rake is designed for domains where parallel computation on regular data structures dominates the workload:

**Real-time simulation.** Particle systems, physics engines, cloth and fluid dynamics, rigid body solvers. These workloads process thousands to millions of entities per frame, applying the same operations to each. SIMD parallelism maps directly to entity parallelism.

**Signal processing.** Audio effects, software-defined radio, telecommunications. These workloads process streams of samples through filter chains. Multiple channels (audio) or symbols (communications) can occupy parallel lanes.

**Image and video processing.** Convolution, color space conversion, compositing, encoding, decoding. Pixels are independent; SIMD lanes map to pixels or pixel blocks.

**Scientific computing.** N-body simulation, finite element methods, Monte Carlo methods, molecular dynamics. These workloads have regular structure and high arithmetic intensity.

**Machine learning inference.** Neural network evaluation, particularly for small batch sizes where GPU kernel launch overhead dominates. SIMD lanes map to batch elements or feature dimensions.

**Game engine internals.** Animation blending, skeletal transforms, culling, LOD selection, spatial queries. These are the compute-bound portions that currently require hand-written SIMD.

Rake is not designed for:

**Text processing.** Strings are variable-length and character-oriented. They do not vectorize well.

**Graph algorithms.** Pointer-chasing through irregular structures defeats SIMD parallelism.

**I/O-bound workloads.** Web servers, databases, file processors. These are limited by I/O latency, not compute throughput.

**GUI applications.** Event-driven, irregular control flow, modest compute requirements.

For these domains, conventional languages remain appropriate. Rake targets the 10-20% of code that consumes 80-90% of cycles in compute-bound applications.

### 1.4 Notation Conventions

This specification uses the following conventions:

**Code examples** appear in monospace font:

```
crunch magnitude v =
  sqrt (v.x * v.x + v.y * v.y + v.z * v.z)
```

**Grammar rules** use Extended Backus-Naur Form (EBNF):

```ebnf
crunch_def = 'crunch' IDENT params '=' expr ;
```

In EBNF:
- `'keyword'` denotes a literal keyword
- `IDENT` denotes a token class (identifier)
- `|` denotes alternation
- `*` denotes zero or more repetitions
- `+` denotes one or more repetitions
- `?` denotes optional elements
- `( )` groups elements

**Semantic descriptions** use the following terms:
- "shall" indicates a requirement on conforming implementations
- "shall not" indicates a prohibition
- "may" indicates permission
- "should" indicates a recommendation
- "undefined behavior" indicates that the specification imposes no requirements

**Lane indices** are zero-based. In an 8-wide vector, lanes are numbered 0 through 7.

**Target width** refers to the number of lanes in a rack for a given compilation target. This is a compile-time constant.

**Comments** in code examples use ML-style notation:

```
(* This is a comment *)

-- This is also a comment (single-line)
```

### 1.5 Document Organization

This specification is organized into sixteen parts:

**Part I (Overview)** introduces the language and its design philosophy.

**Part II (Lexical Structure)** defines the character set, tokens, and lexical conventions including the ML-inspired syntax.

**Part III (Types)** defines the type system: racks, scalars, stacks, packs, and compounds.

**Part IV (Declarations and Scope)** defines how names are introduced and resolved.

**Part V (Expressions)** defines operators, lane operations, and memory access.

**Part VI (Statements)** defines assignments, rails, and control flow.

**Part VII (Definitions)** defines stack, crunch, rake, run, and spread.

**Part VIII (Modules and Programs)** defines program structure and module organization.

**Part IX (Memory Model)** defines ownership, aliasing, and synchronization.

**Part X (Error Handling)** defines behavior for numeric errors and runtime failures.

**Part XI (Standard Library)** defines the built-in modules for math, I/O, and utilities.

**Part XII (Foreign Function Interface)** defines C interoperability.

**Part XIII (Compilation Model)** defines targets, optimization, and linking.

**Part XIV (Tooling)** defines compiler interface, diagnostics, and debugger support.

**Part XV (Formal Grammar)** provides the complete grammar in EBNF.

**Part XVI (Appendices)** provides reference material, examples, and rationale.

---

## Chapter 2: Language Overview

### 2.1 The Vector-First Model

Rake's execution model is based on SPMD (Single Program, Multiple Data) execution on SIMD hardware. Understanding this model is essential to writing effective Rake programs.

**Lanes and Racks**

A modern CPU's SIMD unit operates on wide registers. An AVX2 register holds 8 single-precision floats. An AVX-512 register holds 16. These individual positions within a register are called **lanes**.

A **rack** is a value that occupies one such register. When you declare:

```
x : float rack
```

You are declaring a value that contains one float per lane — 8 floats on AVX2, 16 on AVX-512. But conceptually, `x` is a single value, not an array of 8 or 16 values.

When you write:

```
let y : float rack = x * <two> + <one> in ...
```

You are performing one multiply and one add. These operations happen in all lanes simultaneously. There is no loop; there is one instruction that affects all lanes at once.

**Scalars and Broadcasting**

A **scalar** is a single value, uniform across all lanes. Scalars are marked with angle brackets:

```rake
let <threshold> : float = 0.5 in ...
```

When a scalar appears in an expression with a rack, it **broadcasts** — the scalar value is replicated to all lanes:

```rake
let values : float rack = ... in
let result : float rack = values * <threshold> in
(* <threshold> broadcasts: each lane multiplies by 0.5 *)
```

The angle bracket notation makes this broadcast visible at every use site. The brackets visually suggest the value "spreading outward" to fill all lanes — `<threshold>` expands to meet the width of `values`. This notation is central to Rake's design and shall not be confused with any other syntax.

**Stacks and Structure-of-Arrays**

A **stack** (Structure-of-Arrays) defines parallel data containing racks:

```
stack Particles = {
  pos   : vec3 rack;
  vel   : vec3 rack;
  life  : float rack;
  alive : bool rack
}
```

A single `Particles` value contains one particle per lane. On AVX2, that's 8 particles. The fields are stored contiguously by component: all x-positions together, all y-positions together, and so on. This layout enables efficient SIMD access.

Note that fields explicitly include the `rack` keyword. This reinforces the parallel nature: every field in a `stack` is a rack, holding one value per lane.

**Memory Layout of Compound Racks**

A `vec3 rack` deserves special attention. It represents N vec3 values (where N is the target width), but stored as **three separate float racks**:

```
vec3 rack =
  x : float rack   (* all N x-components *)
  y : float rack   (* all N y-components *)
  z : float rack   (* all N z-components *)
```

When you access `.x` on a `vec3 rack`, you get a `float rack` containing all the x-components. This Structure-of-Arrays layout is what makes SIMD operations efficient — all x-components can be loaded into one register and processed together.

**Packs**

A **pack** is a collection of stack chunks:

```
particles : Particles pack
```

This represents many chunks of particles — enough to hold thousands or millions of particles organized in SIMD-friendly groups. The size of a pack varies based on the target architecture's lane width.

**Rails and Predicates: Predication, Not Branching**

In scalar programming, an `if` statement selects one path of execution. In vector programming, different lanes may need different paths. This is impossible with branching — you cannot branch to two places simultaneously.

Instead, SIMD hardware uses **masks** and **predicated execution**. A mask is a vector of booleans, one per lane. Operations are predicated on masks: computations occur in all lanes, but results are selected based on which predicate matched.

Rake makes this explicit through **rails**, written with `|` syntax inspired by ML pattern matching:

```
rake classify x =
  | negative := x < <0>  -> <-1>
  | zero     := x is <0> -> <0>
  | positive := x > <0>  -> <1>
```

Each `|` introduces a rail. The `:=` defines a named predicate (mask). The `->` introduces the expression for lanes matching that predicate.

**Critical: All rails execute.** There is no branching. The expressions `<-1>`, `<0>`, and `<1>` are all computed, then the mask determines which result each lane receives. This is `select`, not `branch`.

For lanes where `x` contains `[-3, 0, 5, -1, 2, 0, -7, 4]`:
- All three expressions compute their results
- The `negative` mask is `[true, false, false, true, false, false, true, false]`
- The `zero` mask is `[false, true, false, false, false, true, false, false]`
- The `positive` mask is `[false, false, true, false, true, false, false, true]`
- Results merge: `[-1, 0, 1, -1, 1, 0, -1, 1]`

Rails must be **mutually exclusive and exhaustive**. The compiler shall verify that:
1. No lane can match multiple rails (mutual exclusivity)
2. Every lane matches exactly one rail (exhaustiveness)

Use `otherwise` as a catch-all for remaining lanes:

```
| otherwise -> default_value
```

**Sweeping Packs**

Processing a pack involves **sweeping** through it:

```rake
sweep particles -> p:
  p <- step p <cfg>
```

The `sweep` construct iterates through the pack, binding one stack chunk at a time to `p`. Each iteration processes one chunk — 8 or 16 particles — in parallel.

**Temporal Iteration**

The `run` construct provides iteration over time:

```rake
run simulate particles <cfg> <frames> =
  repeat <frames> times:
    sweep particles -> p:
      p <- step p <cfg>
  results <| particles
```

The outer `repeat` loops `<frames>` times. Each iteration sweeps through all particles. This separates two dimensions of execution: sweeping is parallel across lanes; running is sequential across time. The `results <|` clause explicitly declares what the `run` produces.

**Core Distribution**

The `spread` construct distributes work across CPU cores:

```
spread particles across cores -> chunk:
  sweep chunk -> p:
    p <- step p <cfg>
```

Each core receives a portion of `particles` and processes it independently. The `spread` boundary is an implicit synchronization point.

**Retiring Lanes**

The `retire` keyword permanently deactivates a lane within the current rake:

```
rake step p <cfg> =
  | dead := not p.alive -> retire
  | ... -> ...
```

When a lane retires:
1. The lane's mask bit is set to false
2. Subsequent operations in this rake skip the lane
3. The lane remains inactive for any crunch calls within this rake
4. When the rake returns, the lane's data is unchanged

**Compaction with `compact`**

After processing, some lanes may be retired (dead particles, converged iterations, etc.). The `compact` operation compacts a pack, physically removing retired entries:

```rake
sweep particles -> p:
  p <- step p <cfg>   (* some lanes retire *)

compact particles     (* remove retired lanes, shrink pack *)
```

Compaction is expensive (cross-lane shuffles, memory movement) and is marked with a distinct keyword to make this cost visible. Note that `compact` is an operation (verb), while `pack` is a type suffix (noun) — `Particles pack` declares a collection of stack chunks.

### 2.2 Core Vocabulary Summary

Rake introduces specific vocabulary to describe its execution model:

| Term | Meaning |
|------|---------|
| **rack** | A vector of values occupying one SIMD register |
| **lane** | One position within a rack |
| **scalar** | A single value, uniform across lanes (marked with `<>`) |
| **stack** | A Structure-of-Arrays data type containing racks (one element per lane) |
| **aos** | An Array-of-Structures data type (for C interop) |
| **single** | An all-scalar configuration structure |
| **pack** | A collection of stack chunks |
| **rail** | A named predicate that partitions lanes |
| **mask** | A boolean rack indicating which lanes are active |
| **broadcast** | Replicating a scalar to all lanes |
| **gather** | Loading from non-contiguous memory locations |
| **scatter** | Storing to non-contiguous memory locations |
| **retire** | Permanently deactivating a lane within a rake |
| **compact** | Compacting a pack to remove retired lanes |
| **reduce** | Collapsing all lanes to a single scalar value |

The data structures:

| Construct | Layout | Typical Use |
|-----------|--------|-------------|
| `stack Name = { ... }` | Structure-of-Arrays | Parallel entity data (contains racks) |
| `aos Name = { ... }` | Array-of-Structures | C interop, single entities |
| `single Name = { ... }` | All scalar | Configuration, parameters |
| `Name pack` | Collection of stack chunks | Large parallel datasets |
| `Name array` | Array of aos structs | Interop with C arrays |

The function types:

| Construct | Purpose |
|-----------|---------|
| `crunch name params = expr` | Single-rail pure function (inlinable) |
| `rake name params = expr` | Multi-rail parallel function |
| `run name params = body` | Sequential composition with iteration |

The control constructs:

| Construct | Purpose |
|-----------|---------|
| `\| name := pred -> expr` | Define a named rail with consequence |
| `\| pred -> expr` | Anonymous rail |
| `\| otherwise -> expr` | Catch-all rail for remaining lanes |
| `sweep pack -> item: body` | Iterate over pack chunks |
| `repeat n times: body` | Counted temporal iteration |
| `repeat until <cond>: body` | Conditional temporal iteration |
| `spread data across cores -> chunk: body` | Core-level parallelism |
| `retire` | Deactivate current lane |
| `halt` | Stop all lanes immediately |
| `compact pack` | Compact pack, removing retired lanes |
| `results <\| expr` | Declare return value of run/rake |

### 2.3 Syntax Philosophy

Rake adopts an ML-family syntax, drawing from OCaml and similar languages. This choice is deliberate:

**Rails mirror pattern matching.** The `|` syntax for rails directly parallels ML's pattern matching. Programmers familiar with functional languages will immediately recognize the structure:

```
(* OCaml pattern matching *)
match x with
| Negative -> -1
| Zero -> 0
| Positive -> 1

(* Rake rail matching *)
| negative := x < <0>  -> <-1>
| zero     := x is <0> -> <0>
| positive := x > <0>  -> <1>
```

The visual similarity is intentional. Rails are predicates that partition lanes, just as patterns are predicates that partition values.

**Predicates use `is` for clarity.** Rail predicates use `is` and `is not` rather than `==` and `!=`:

```
| zero := x is <0> -> handle_zero
| nonzero := x is not <0> -> handle_nonzero
```

This distinguishes predicates (which partition lanes) from equality expressions in general code. The word `is` reads naturally as a question: "is x equal to 0?"

**Let bindings are expressions.** Following ML tradition, `let ... in` introduces a binding within an expression:

```
let oc = ray.origin - sphere.center in
let <a> = dot ray.dir ray.dir in
let <b> = <2> * dot oc ray.dir in
...
```

This encourages a functional style where values are bound once and transformations are chained.

**Record updates are functional.** The `{ ... with }` syntax creates a new record with some fields changed:

```
{ particle with
  pos = particle.pos + particle.vel * <dt>;
  life = particle.life - <dt> }
```

This encourages thinking in terms of transformations rather than mutations, which aligns well with SIMD's execution model where "modifying one lane" is not a primitive operation.

**Mutation is restricted and marked.** Mutation with `<-` is only permitted in imperative contexts (`run` and `sweep` bodies):

```
sweep particles -> p:
  p <- step p <cfg>   (* mutation: update stack slot *)
```

Within `rake` and `crunch` definitions, all operations are pure — you produce new values via expressions and `{ ... with }` updates.

**Comments use ML notation.** Block comments are `(* ... *)`, and single-line comments use `--`:

```
(* This function computes the magnitude of a vector.
   It operates on all lanes simultaneously. *)
crunch magnitude v =
  sqrt (v.x * v.x + v.y * v.y + v.z * v.z)

-- Single line comment
```

### 2.4 A First Example

The following program simulates particles falling under gravity:

```rake
--------------------
-- Stacks
--------------------

(* Particle data - one particle per lane, SoA layout *)
stack Particles = {
  pos   : vec3 rack;
  vel   : vec3 rack;
  life  : float rack;
  alive : bool rack
}

(* Simulation parameters - uniform across all lanes *)
single Config = {
  gravity : vec3;
  dt      : float;
  bounce  : float
}

--------------------
-- Crunches
--------------------

(* Apply gravity - single rail, can inline into other functions *)
crunch apply_gravity p <cfg> =
  { p with vel = p.vel + <cfg>.gravity * <cfg>.dt }

--------------------
-- Rakes
--------------------

(* Physics step - multiple rails for different particle states *)
rake step p <cfg> =
  | dead := not p.alive ->
      retire

  | expired := p.life <= <0> ->
      { p with alive = false }

  | flying := p.alive and p.pos.y >= <0> ->
      let p = apply_gravity p <cfg> in
      { p with
        pos  = p.pos + p.vel * <cfg>.dt;
        life = p.life - <cfg>.dt }

  | hitting := p.alive and p.pos.y < <0> ->
      let p = apply_gravity p <cfg> in
      { p with
        pos.y  = <0> - p.pos.y;
        vel.y  = (<0> - p.vel.y) * <cfg>.bounce;
        life   = p.life - <cfg>.dt }

--------------------
-- Runs
--------------------

(* Main simulation loop *)
run simulate particles <cfg> <frames> =
  repeat <frames> times:
    spread particles across cores -> chunk:
      sweep chunk -> p:
        p <- step p <cfg>
  results <| particles
```

Key observations:

1. **Data layout is explicit.** `stack Particles` declares SoA layout with explicit `rack` on each field. `single Config` declares all-scalar fields. There is no ambiguity about memory organization.

2. **Stack names are plural.** A `Particles` value contains multiple particles (one per lane). This naming convention reinforces the parallel mental model.

3. **Scalars use angle brackets.** `<cfg>`, `<frames>`, `<dt>`, `<bounce>` are all clearly marked. The angle brackets visually suggest broadcasting.

4. **Rails partition lanes with `|` syntax.** The four rails (`dead`, `expired`, `flying`, `hitting`) define mutually exclusive categories. All rail bodies execute; masks select results.

5. **Predicates use `:=` for definition.** The `dead := not p.alive` syntax clearly distinguishes predicate definition from assignment.

6. **Consequences follow `->`.** Each rail's consequence appears after the arrow, just as in pattern matching.

7. **`retire` removes lanes from computation.** Dead particles exit early, saving computation on subsequent operations within this rake.

8. **`crunch` defines inlinable single-rail functions.** The `apply_gravity` function has no rails; it applies uniformly to all active lanes and can be inlined into `rake` functions.

9. **`rake` defines multi-rail parallel functions.** The `step` function uses multiple rails to handle different particle states. The body is pure — it produces a `Particles` value.

10. **`run` provides temporal iteration.** The outer loop runs for `<frames>` timesteps.

11. **`spread` distributes across cores.** Each core processes a portion of the particle pack.

12. **`sweep` iterates over pack chunks.** Each chunk (8 or 16 particles) is processed by calling `step`.

13. **Mutation with `<-` only in sweep.** The assignment `p <- step p <cfg>` updates the pack slot. This is the only place mutation occurs.

14. **Functional updates with `{ ... with }`.** Within `rake` bodies, state changes create new values rather than mutating in place.

15. **`results <|` declares output.** The ligature `<|` visually connects the computation to its result, emphasizing data flow.

### 2.5 Comparison with Existing Approaches

**Versus C with Auto-Vectorization**

```c
// C with auto-vectorization
void scale(float* data, int n, float factor) {
    for (int i = 0; i < n; i++) {
        data[i] *= factor;  // Compiler may or may not vectorize
    }
}
```

```rake
(* Rake *)
run scale data <factor> =
  sweep data -> d:
    d <- d * <factor>
  results <| data
```

The C version may vectorize if the compiler can prove it's safe. The Rake version is semantically parallel — vectorization is guaranteed.

**Versus C with Intrinsics**

```c
// C with AVX2 intrinsics
void scale_avx2(float* data, int n, float factor) {
    __m256 vfactor = _mm256_set1_ps(factor);
    for (int i = 0; i < n; i += 8) {
        __m256 v = _mm256_load_ps(&data[i]);
        v = _mm256_mul_ps(v, vfactor);
        _mm256_store_ps(&data[i], v);
    }
}
```

```rake
(* Rake - same as above *)
run scale data <factor> =
  sweep data -> d:
    d <- d * <factor>
  results <| data
```

The intrinsics version is unreadable, non-portable (AVX2 only), and requires manual handling of array bounds. The Rake version is clear, portable, and handles boundaries automatically.

**Versus ISPC**

```ispc
// ISPC
export void scale(uniform float data[], uniform int n, uniform float factor) {
    foreach (i = 0 ... n) {
        data[i] *= factor;
    }
}
```

```rake
(* Rake *)
run scale data <factor> =
  sweep data -> d:
    d <- d * <factor>
  results <| data
```

ISPC and Rake are similar in capability. The differences:

- ISPC uses `uniform` and `varying`; Rake uses `<>` notation and `rack` keyword
- ISPC is a kernel language requiring a C++ host; Rake is complete
- ISPC uses `foreach`; Rake uses `sweep` with pack iteration
- Rake's `|` rail syntax makes predication more explicit than ISPC's `if`
- Rake's ML-style syntax enables more expressive functional patterns

**Versus CUDA**

```cuda
// CUDA
__global__ void scale(float* data, int n, float factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] *= factor;
    }
}
```

```rake
(* Rake *)
run scale data <factor> =
  sweep data -> d:
    d <- d * <factor>
  results <| data
```

CUDA and Rake share the SPMD philosophy but target different hardware:

- CUDA targets GPUs with thousands of threads; Rake targets CPUs with 8-16 lanes
- CUDA has explicit thread indexing; Rake abstracts lane indices
- CUDA has complex memory hierarchy (global, shared, registers); Rake has simpler memory model
- CUDA has high kernel launch latency; Rake has no launch overhead

Rake is for CPU-bound workloads where GPU launch latency is prohibitive or where data already resides in CPU memory.

**Versus OCaml/Haskell**

```ocaml
(* OCaml *)
let scale data factor =
  Array.map (fun x -> x *. factor) data
```

```rake
(* Rake *)
run scale data <factor> =
  sweep data -> d:
    d <- d * <factor>
  results <| data
```

Rake borrows syntax from ML but with crucial semantic differences:

- ML arrays are scalar; Rake packs are parallel
- ML map is sequential abstraction; Rake sweep is parallel primitive
- ML has no concept of lanes; Rake makes lanes explicit
- ML uses garbage collection; Rake uses linear ownership

The syntactic similarity is intentional — it allows ML programmers to transfer their intuitions about pattern matching and functional updates while learning new parallel semantics.

---

*End of Part I*

---

**Summary of Key Terminology**

| Term | Definition |
|------|------------|
| **Rake** | The programming language |
| **rack** | A SIMD vector type; the primitive parallel value |
| **stack** | A Structure-of-Arrays data type; fields are racks |
| **aos** | An Array-of-Structures data type; for C interop |
| **single** | An all-scalar configuration structure |
| **pack** | A collection of stack chunks; the primary large data structure (type suffix) |
| **rail** | A named predicate partitioning lanes; written with `\|` syntax |
| **crunch** | A single-rail pure function (inlinable) |
| **rake** | A multi-rail parallel function |
| **run** | A sequential composition with temporal iteration |
| **sweep** | Iteration over pack chunks |
| **spread** | Distribution across CPU cores |
| **scalar** | A single value uniform across lanes; marked with `<>` |
| **lane** | One position within a rack |
| **mask** | A boolean rack indicating active lanes |
| **retire** | Deactivate a lane within current rake |
| **halt** | Stop all lanes immediately |
| **compact** | Compact a pack, removing retired lanes (operation) |
| **broadcast** | Replicate a scalar to all lanes |
| **results <\|** | Ligature declaring a function's return value |
