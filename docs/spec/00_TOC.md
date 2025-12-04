# The Rake Programming Language Specification

## Table of Contents

### Part I: Overview

1. **Introduction** 1.1. Purpose and Scope 1.2. Design Philosophy 1.3. Target
   Domains 1.4. Notation Conventions 1.5. Document Organization

2. **Language Overview** 2.1. The Vector-First Model 2.2. Core Vocabulary
   Summary 2.3. A First Example 2.4. Comparison with Existing Approaches

### Part II: Lexical Structure

3. **Source Text** 3.1. Character Set (UTF-8) 3.2. Line Structure 3.3.
   Whitespace and Indentation 3.4. Comments

4. **Tokens** 4.1. Keywords 4.2. Identifiers 4.3. Scalar Identifier Convention
   (`<name>` angle bracket notation) 4.4. Operators and Punctuation 4.5. Ligatures
   (`<|` for results, `|>` for pipe, `->` for arrow, `<-` for assignment)
   4.6. Literals 4.6.1. Integer Literals 4.6.2. Floating-Point Literals
   4.6.3. Boolean Literals 4.6.4. String Literals 4.7. Token Separation Rules

### Part III: Types

5. **Type System Overview** 5.1. Rack Types vs Scalar Types 5.2. Type Inference
   5.3. Type Compatibility and Conversion

6. **Primitive Types** 6.1. Numeric Rack Types 6.1.1. `float rack` 6.1.2.
   `double rack` 6.1.3. `int rack`, `int8 rack`, `int16 rack`, `int64 rack`
   6.1.4. `uint rack`, `uint8 rack`, `uint16 rack`, `uint64 rack` 6.2. Boolean
   Rack Type (`bool rack`) 6.3. Scalar Types (`<name>` angle bracket notation)
   6.4. Width and Target Dependency

7. **Compound Rack Types** 7.1. Vector Types (`vec2 rack`, `vec3 rack`,
   `vec4 rack`) 7.2. Matrix Types (`mat3 rack`, `mat4 rack`) 7.3. Component
   Access

8. **Stack Types** 8.1. Fixed-Size Stacks 8.2. Dynamic Stacks (Slices) 8.3.
   Stack Indexing and Slicing 8.4. Memory Layout

9. **Structure Types** 9.1. Structure Definition 9.2. Rack Fields vs Scalar
   Fields 9.3. Structure-of-Arrays Layout 9.4. Structure Instantiation and
   Access 9.5. Nested Structures

10. **Type Aliases** 10.1. Alias Definition 10.2. Scope and Visibility

### Part IV: Declarations and Scope

11. **Declarations** 11.1. Rack Declarations 11.2. Scalar Declarations 11.3.
    Stack Declarations 11.4. Constant Declarations 11.5. Declaration Placement

12. **Scope and Visibility** 12.1. Block Scope 12.2. Function Scope 12.3. Module
    Scope 12.4. Shadowing Rules

### Part V: Expressions

13. **Primary Expressions** 13.1. Identifiers 13.2. Literals 13.3. Parenthesized
    Expressions 13.4. Tuple Expressions

14. **Operators** 14.1. Arithmetic Operators 14.2. Comparison Operators 14.3.
    Logical Operators 14.4. Bitwise Operators 14.5. Operator Precedence and
    Associativity 14.6. Scalar-Rack Broadcasting

15. **Lane Operations** 15.1. Lane Queries (`lanes()`, `lane_index()`, `lead()`)
    15.2. Lane Predicates (`any`, `all`, `none`, `tally`) 15.3. Reductions
    (`reduce(+, expr)`, `reduce(*, expr)`, `reduce(min, expr)`, etc.) 15.4. Lane
    Manipulation (`shuffle`, `rotate`, `shift`, `broadcast`, `select`) 15.5.
    Compression and Expansion (`compress`, `expand`)

16. **Memory Operations** 16.1. Stack Indexing 16.2. Gather Operations 16.3.
    Scatter Operations 16.4. Contiguous Load and Store

17. **Function Calls** 17.1. Call Syntax 17.2. Argument Passing 17.3.
    Scalar-Rack Parameter Interaction

18. **Struct Expressions** 18.1. Struct Literals 18.2. Field Access 18.3.
    Functional Update

### Part VI: Statements

19. **Assignment Statements** 19.1. Simple Assignment (`<-`) 19.2. Compound
    Assignment 19.3. Multiple Assignment

20. **Rail Definitions** 20.1. Rail Syntax (`| pred -> expr`) 20.2. Named Rails
    (`| name := pred -> expr`) 20.3. Predicate Expressions 20.3.1. Comparison
    (`is`, `is not`, `<`, `<=`, `>`, `>=`) 20.3.2. Boolean Composition
    (`and`, `or`, `not`) 20.4. Rail Composition 20.5. Rail Completeness and
    Mutual Exclusivity

21. **Control Flow Statements** 21.1. `retire` Statement (deactivate lane)
    21.2. `halt` Statement (stop all lanes)

### Part VII: Definitions

22. **Stack Definitions** 22.1. Syntax (`stack Name = { ... }`) 22.2. Field
    Declarations 22.3. Explicit `rack` on Fields 22.4. Memory Layout

23. **Crunch Definitions** 23.1. Syntax (`crunch name params = expr`) 23.2.
    ML-Style Parameters 23.3. Single-Rail Pure Functions 23.4. Inlining Semantics

24. **Rake Definitions** 24.1. Syntax (`rake name params = expr`) 24.2. Parameters
    and Returns 24.3. Body Structure 24.4. `results <|` Clause 24.5. Implicit
    Lane Masking 24.6. Calling Rakes

25. **Run Definitions** 25.1. Syntax (`run name params = body results <| expr`)
    25.2. Sequential Composition 25.3. Repeat Statements 25.3.1. Counted
    Repetition (`repeat N times`) 25.3.2. Conditional Repetition (`repeat until`)
    25.3.3. Iteration Variable Binding (`as`) 25.4. Sweep Statements 25.5.
    Compact Statement 25.6. Nested Rake and Run

26. **Spread Definitions** 26.1. Syntax 26.2. Core Distribution 26.3. Chunk
    Binding 26.4. Synchronization (`sync`)

### Part VIII: Modules and Programs

27. **Module System** 27.1. Module Declaration 27.2. Import Statements 27.3.
    Export and Visibility 27.4. Module Paths

28. **Program Structure** 28.1. Entry Point 28.2. Initialization Order 28.3.
    Command-Line Arguments

### Part IX: Memory Model

29. **Ownership and Lifetimes** 29.1. Value Semantics for Racks 29.2. Reference
    Semantics for Stacks 29.3. Borrowing Rules 29.4. Mutable References

30. **Aliasing Rules** 30.1. Within Rake Blocks 30.2. Within Spread Blocks 30.3.
    Scatter Conflict Behavior

31. **Synchronization** 31.1. Implicit Barriers 31.2. Explicit Sync 31.3. Atomic
    Operations

### Part X: Error Handling

32. **Numeric Errors** 32.1. IEEE 754 Semantics 32.2. Division by Zero 32.3.
    Overflow and Underflow 32.4. Invalid Operations (NaN)

33. **Checked Operations** 33.1. Checked Arithmetic 33.2. Error Mask Queries
    33.3. Propagation Behavior

34. **Runtime Errors** 34.1. Out-of-Bounds Access 34.2. Assertion Failures 34.3.
    Panic and Abort

### Part XI: Standard Library

35. **Core Module** 35.1. Primitive Type Definitions 35.2. Built-in Operations

36. **Math Module** 36.1. Trigonometric Functions 36.2. Hyperbolic Functions
    36.3. Exponential and Logarithmic Functions 36.4. Power and Root Functions
    36.5. Rounding Functions 36.6. Utility Functions 36.7. Fast Approximations
    36.8. Constants

37. **Vec Module** 37.1. Dot Product 37.2. Cross Product 37.3. Length and
    Distance 37.4. Normalization 37.5. Reflection and Refraction

38. **Mat Module** 38.1. Matrix Multiplication 38.2. Vector Transformation 38.3.
    Transpose and Inverse 38.4. Construction Functions 38.5. Projection
    Functions

39. **Random Module** 39.1. State Management 39.2. Uniform Distributions 39.3.
    Normal Distribution 39.4. Geometric Distributions

40. **Sort Module** 40.1. Lane Sorting 40.2. Stack Sorting 40.3. Partial Sort
    and Selection 40.4. Merge Operations

41. **IO Module** 41.1. Stream Types 41.2. File Operations 41.3. Rack I/O
    (`rack_from`, `rack_into`) 41.4. Stack I/O (`stack_from`, `stack_into`)
    41.5. Text I/O (Scalar) 41.6. Error Handling

42. **Memory Module** 42.1. Allocation Functions 42.2. Stack Construction 42.3.
    Copying and Moving

### Part XII: Foreign Function Interface

43. **C Interoperability** 43.1. External Function Declarations 43.2. Calling
    Conventions 43.3. Type Mapping 43.4. Pointer Handling

44. **Exporting Rake Functions** 44.1. Export Declaration 44.2. Generated C
    Headers 44.3. ABI Stability

45. **Scalar Context Blocks** 45.1. `single for each lane` Construct 45.2.
    Performance Implications 45.3. Use Cases

### Part XIII: Compilation Model

46. **Compilation Units** 46.1. Source Files 46.2. Compilation Order 46.3.
    Separate Compilation

47. **Target Architectures** 47.1. x86-64 (SSE4, AVX2, AVX-512) 47.2. ARM64
    (NEON) 47.3. ARM64 (SVE) — Future 47.4. WebAssembly SIMD 47.5. Target
    Selection

48. **Optimization** 48.1. Mask Simplification 48.2. Gather/Scatter Optimization
    48.3. Lane Uniformity Detection 48.4. Fusion and Unrolling

49. **Linking** 49.1. Object Files 49.2. Static Libraries 49.3. Shared Libraries
    49.4. Multi-Target Binaries

### Part XIV: Tooling

50. **Compiler** 50.1. Command-Line Interface 50.2. Diagnostic Messages 50.3.
    Warning Levels 50.4. Optimization Levels

51. **Vectorization Report** 51.1. Report Format 51.2. Lane Utilization
    Estimation 51.3. Gather/Scatter Identification

52. **Debugger Integration** 52.1. Breakpoints 52.2. Rack Inspection 52.3. Rail
    State Visualization 52.4. Lane Focus Mode 52.5. Conditional Breakpoints

53. **IDE Support** 53.1. Language Server Protocol 53.2. Syntax Highlighting
    53.3. Code Navigation 53.4. Refactoring Support

### Part XV: Formal Grammar

54. **Notation** 54.1. EBNF Conventions 54.2. Terminal and Non-Terminal Symbols

55. **Lexical Grammar** 55.1. Tokens 55.2. Keywords 55.3. Operators 55.4.
    Literals

56. **Syntactic Grammar** 56.1. Programs and Modules 56.2. Declarations 56.3.
    Types 56.4. Expressions 56.5. Statements 56.6. Definitions

### Part XVI: Appendices

**Appendix A: Keyword Reference** Complete alphabetical listing of all keywords
with brief descriptions

**Appendix B: Operator Reference** Complete listing of operators with precedence
table

**Appendix C: Standard Library Summary** Quick reference for all standard
library functions

**Appendix D: Target Architecture Details** Vector widths, supported operations,
and performance characteristics per target

**Appendix E: Glossary** Definitions of terms: rack, stack, rail, lane, mask,
broadcast, gather, scatter, etc.

**Appendix F: Design Rationale** Extended discussion of language design
decisions and alternatives considered

**Appendix G: Example Programs** Complete, annotated example programs:

- G.1. Particle System
- G.2. Ray Tracer
- G.3. Image Convolution
- G.4. Mandelbrot Renderer
- G.5. Biquad Audio Filter
- G.6. N-Body Simulation

**Appendix H: Migration Guide**

- H.1. From C/C++ with Intrinsics
- H.2. From ISPC
- H.3. From CUDA (Conceptual)

**Appendix I: Future Directions** Features under consideration for future
versions:

- I.1. GPU Backend
- I.2. SVE Variable-Width Support
- I.3. Distributed Execution
- I.4. Automatic Differentiation

---

## Summary Statistics

- **Parts:** 16
- **Chapters:** 56
- **Appendices:** 9

---

Shall I begin filling in specific sections? I'd suggest starting with:

1. **Part I (Overview)** — Sets the tone and vision
2. **Part III (Types)** — The foundation of the type system
3. **Part VI (Statements)** — Rails and lane selection (the novel parts)
4. **Appendix E (Glossary)** — Establishes vocabulary

Which would you like to tackle first?
