# Rake Language Roadmap

## Current State (v0.2.0)

Working: lexer, parser, type checker, MLIR emission for basic crunch/rake functions.
The `over` loop emits `scf.for` with explicit vectorization (width 8).

## Phase 1: Unified Parallelization Model

**Goal**: Single emitter that targets both CPU SIMD and GPU via different MLIR lowering passes.

### The Problem

Current emission assumes CPU SIMD — one thread processing 8 elements via wide registers:

```mlir
scf.for %i = 0 to %n step 1 {
    %v = vector.load %mem[%offset] : memref<?xf32>, vector<8xf32>
    %r = arith.addf %v, %c : vector<8xf32>
    vector.store %r, %out[%offset] : memref<?xf32>, vector<8xf32>
}
```

GPU execution is fundamentally different — thousands of threads, each processing one element:

```mlir
gpu.launch blocks(...) threads(...) {
    %id = gpu.thread_id x
    %v = memref.load %mem[%id] : memref<?xf32>
    %r = arith.addf %v, %c : f32
}
```

### The Solution: `scf.parallel`

Emit `scf.parallel` instead of `scf.for`:

```mlir
scf.parallel (%i) = (%c0) to (%n) step (%c1) {
    %v = memref.load %mem[%i] : memref<?xf32>
    %r = arith.addf %v, %c : f32
    memref.store %r, %out[%i] : memref<?xf32>
    scf.reduce
}
```

This declares "iterations are independent" without specifying execution strategy.

Then choose lowering path via mlir-opt flags:

| Target | Passes |
|--------|--------|
| CPU SIMD | `--convert-scf-to-cf` + vectorization passes |
| GPU SPIR-V | `--convert-scf-to-gpu --convert-gpu-to-spirv` |
| CPU Parallel | `--convert-scf-to-openmp` |

### Implementation Steps

#### 1.1 Refactor `emit_over_loop` in `lib/mlir.ml`

Current (`lib/mlir.ml:387-498`):
```ocaml
emit ctx "scf.for %s = %s to %s step %s {" iter_var zero num_iters one;
(* ... vector operations at width 8 ... *)
```

New:
```ocaml
emit ctx "scf.parallel (%s) = (%s) to (%s) step (%s) {" iter_var zero count_idx one;
(* ... scalar operations, width 1 ... *)
emit ctx "  scf.reduce";
emit ctx "}";
```

#### 1.2 Add `--scalar-emit` mode

Introduce a flag/context field to emit scalar operations:

```ocaml
type ctx = {
  (* ... existing fields ... *)
  mutable scalar_mode: bool;  (* true = emit scalar ops, false = emit vector<8> *)
}
```

When `scalar_mode = true`:
- Literals emit as `f32` not `vector<8xf32>`
- Loads emit as `memref.load` not `vector.load`
- All operations are scalar

#### 1.3 Handle tine/through/sweep in scalar mode

In scalar mode, tines produce `i1` (single bool), not `vector<8xi1>`.
Through blocks become simple `scf.if`:

```mlir
scf.if %mask {
    // body
    scf.yield %result
} else {
    scf.yield %passthru
}
```

Sweep becomes nested `arith.select` on scalars (already works, just narrower types).

#### 1.4 Update `run` function emission

Current emits pack field expansion to memrefs. Keep this — it's the right abstraction.
Just change the loop structure and operation width.

#### 1.5 Create lowering scripts

```bash
# scripts/lower-cpu.sh
mlir-opt $1 \
    --convert-scf-to-cf \
    --convert-vector-to-llvm \
    --convert-func-to-llvm \
    --convert-arith-to-llvm \
    -o ${1%.mlir}.llvm.mlir

# scripts/lower-gpu.sh
mlir-opt $1 \
    --convert-scf-to-gpu \
    --gpu-kernel-outlining \
    --convert-gpu-to-spirv \
    --spirv-lower-abi-attrs \
    --spirv-update-vce \
    -o ${1%.mlir}.spv
```

#### 1.6 Vectorization recovery for CPU

For CPU path, rely on MLIR's vectorization passes to recover SIMD:

```bash
mlir-opt $1 \
    --affine-super-vectorize="virtual-vector-size=8" \
    # or
    --test-vector-transfer-lowering-patterns \
    --convert-scf-to-cf \
    ...
```

Alternatively, keep current vector emission as `--legacy-vector` mode for comparison.

### Validation

1. Emit same Rake program in both modes
2. Compare: scalar+vectorization passes vs direct vector emission
3. Verify GPU path produces valid SPIR-V
4. Benchmark CPU paths to ensure vectorization recovery works

---

## Phase 2: Stop Silent Failures

**Goal**: Fail loudly on unhandled cases instead of producing wrong code.

### 2.1 Remove catch-all in `emit_expr`

Location: `lib/mlir.ml:353-357`

Current:
```ocaml
| _ ->
    let result = fresh ctx "todo" in
    emit ctx "%s = arith.constant dense<0.0> : %s" result vec_f32;
    (result, Rack SFloat)
```

Replace with:
```ocaml
| other ->
    failwith (Printf.sprintf "Unhandled expression: %s"
      (Ast.show_expr_kind other))
```

### 2.2 Fix field access emission

Location: `lib/mlir.ml:266-292`

Current fallback emits a comment and returns undefined SSA name:
```ocaml
emit ctx "// Field access: %s.%s" base field;
(result, get_field_type t field)  (* result is never defined! *)
```

Fix: fail if field binding not found:
```ocaml
| None ->
    failwith (Printf.sprintf "Field %s.%s not bound in context" base field)
```

### 2.3 Verify function call arguments in type checker

Location: `lib/typecheck.ml:213-216`

Current:
```ocaml
| ECall (name, _args) -> (
    match Hashtbl.find_opt env.funcs name with
    | Some (_, ret) -> ret  (* args ignored *)
```

Fix:
```ocaml
| ECall (name, args) -> (
    match Hashtbl.find_opt env.funcs name with
    | Some (param_types, ret) ->
        let arg_types = List.map (infer_expr env) args in
        if List.length arg_types <> List.length param_types then
          type_errorf expr.loc "Function %s expects %d args, got %d"
            name (List.length param_types) (List.length arg_types);
        List.iter2 (fun expected actual ->
          if not (compatible expected actual) then
            type_errorf expr.loc "Argument type mismatch: expected %s, got %s"
              (show_concise expected) (show_concise actual)
        ) param_types arg_types;
        ret
    | None -> type_errorf expr.loc "Unknown function: %s" name)
```

---

## Phase 3: Complete Core Emission

### 3.1 Fix record emission

Location: `lib/mlir.ml:323-337`

Current just emits a comment. Options:

**Option A**: Emit as separate SSA values (current implicit approach, make explicit)
```ocaml
(* Records decompose to individual field values in SSA *)
(* Store field->value mappings for later field access *)
```

**Option B**: Use MLIR tuple type
```mlir
%rec = tuple.from_elements %field1, %field2 : tuple<f32, f32>
```

**Recommendation**: Option A is simpler and matches how field access already works.
Just ensure the field->SSA mappings are complete and fail if accessed before defined.

### 3.2 Fix hardcoded return types

Location: `lib/mlir.ml:641` (`emit_crunch`)

Current:
```ocaml
emit ctx "func.func @%s(%s) -> %s ..." name ... vec_f32;  (* always f32 *)
```

Fix: use the result type from the function signature:
```ocaml
let result_type = match result.result_type with
  | Some ty -> mlir_type (typ_to_t ctx.type_env ty)
  | None -> vec_f32  (* default *)
in
emit ctx "func.func @%s(%s) -> %s ..." name ... result_type;
```

### 3.3 Handle Unknown types in type checker

Locations in `lib/typecheck.ml`:
- `EOuter` (line 270): Outer product type is matrix — implement properly or error
- `EGather` (line 254): Should infer from base pointer type
- `ELambda` (line 278): Should produce `Fun(param_types, body_type)`

For now, convert `Unknown` returns to explicit errors:
```ocaml
| EOuter (_, _) ->
    type_errorf expr.loc "Outer product not yet implemented"
```

---

## Phase 4: Semantic Completeness

### 4.1 Sweep exhaustiveness check

Add warning/error if sweep tines don't cover all cases and no catch-all `_` present.

In `check_sweep`:
```ocaml
let has_catchall = List.exists (fun arm -> arm.arm_tine = None) sw.sweep_arms in
if not has_catchall then
  (* Could warn, or require explicit catch-all *)
  ()
```

### 4.2 Document tine priority semantics

Current: first matching tine wins (due to select chain order).
Document this in spec or add mutual-exclusivity check.

### 4.3 Update spec for `arith.select` vs `vector.mask`

The current implementation uses `arith.select` for through blocks.
This is correct for side-effect-free code. Update `docs/spec/03_tines_and_through.md`
to reflect the actual implementation, or add a TODO for `vector.mask` when needed.

---

## Phase 5: Testing Infrastructure

### 5.1 Build test harness

Create `test/harness.c`:
```c
#include <stdio.h>
#include <math.h>

// Declare Rake functions (will be linked from compiled .o)
extern void compute_distances(float* ox, float* oy, float* oz,
                              float cx, float cy, float cz,
                              long count, float* out);

int main() {
    float ox[] = {1, 2, 3, 4, 5, 6, 7, 8};
    float oy[] = {0, 0, 0, 0, 0, 0, 0, 0};
    float oz[] = {0, 0, 0, 0, 0, 0, 0, 0};
    float out[8];

    compute_distances(ox, oy, oz, 0, 0, 0, 8, out);

    for (int i = 0; i < 8; i++) {
        float expected = sqrtf(ox[i]*ox[i]);
        if (fabsf(out[i] - expected) > 0.001f) {
            printf("FAIL: out[%d] = %f, expected %f\n", i, out[i], expected);
            return 1;
        }
    }
    printf("PASS\n");
    return 0;
}
```

### 5.2 End-to-end test script

```bash
#!/bin/bash
# test/run_tests.sh

set -e

for rk in examples/*.rk; do
    echo "Testing $rk..."

    # Compile Rake -> MLIR
    ./rake --emit-mlir "$rk" > "${rk%.rk}.mlir"

    # Lower to LLVM
    mlir-opt "${rk%.rk}.mlir" \
        --convert-scf-to-cf \
        --convert-vector-to-llvm \
        --convert-func-to-llvm \
        --convert-arith-to-llvm \
        --reconcile-unrealized-casts \
        -o "${rk%.rk}.llvm.mlir"

    # Translate to LLVM IR
    mlir-translate --mlir-to-llvmir "${rk%.rk}.llvm.mlir" -o "${rk%.rk}.ll"

    # Compile to object
    llc -filetype=obj "${rk%.rk}.ll" -o "${rk%.rk}.o"

    echo "  Generated ${rk%.rk}.o"
done
```

### 5.3 Test cases to implement

1. **Simple crunch** — arithmetic, verify vectorization
2. **Rake with tines** — predicated execution
3. **Over loop** — pack iteration, tail masking
4. **Reductions** — `\+/`, `\*/`
5. **Broadcasting** — scalar to rack promotion
6. **Field access** — stack/single member access
7. **Nested tines** — composed predicates

---

## Phase 6: Quality of Life

### 6.1 Better error messages

Include source location in all errors:
```ocaml
type_errorf expr.loc "Type mismatch at %s:%d:%d: ..."
  loc.file loc.line loc.col
```

### 6.2 Add `if/else` expressions

New AST node:
```ocaml
| EIf of expr * expr * expr  (* condition, then, else *)
```

Emit as `scf.if` (for scalar mode) or `arith.select` (for vector mode).

### 6.3 Add basic `for` loop

For fixed iteration counts (useful for unrolling):
```rake
for i in 0..4:
    accumulate(i)
```

Emit as `scf.for` with known bounds.

### 6.4 Complete math builtins

Type checker lists: `sqrt`, `sin`, `cos`, `tan`, `exp`, `log`, `abs`, `floor`, `ceil`, `min`, `max`, `pow`, `atan2`

Ensure emitter handles all of these (currently partial).

---

## Deferred (Post-MVP)

- Module/import system
- Closures with variable capture
- Strings and I/O
- Memory management (GC or ownership)
- Custom MLIR dialect for Rake-specific optimizations
- GPU texture/image support
- Warp-level primitives for GPU

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2024-12 | Use `arith.select` for through blocks | Simpler emission, equivalent for pure code |
| 2024-12 | Prioritize `scf.parallel` over fixing type holes | Enables GPU path, higher impact |
| TBD | Scalar mode as default, vector as optimization | Lets MLIR handle parallelization strategy |
