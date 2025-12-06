## Phase 1: Binding Operator Taxonomy

**Goal**: Establish clear semantic distinctions between binding operators to enable
guaranteed optimizations. Rake's philosophy is that syntax should encode
optimization requirements, not hints—the compiler either emits optimized code or
errors.

### 1.0 Design Rationale

Compiler optimizations often fail silently because they can't make guarantees
with complete certainty. Rake addresses this by making optimization requirements
explicit in the syntax:

- `crunch` = pure, always inlined, no divergence (guaranteed fusion)
- `rake` = divergent but structured (guaranteed predicated execution)
- `run` = entry point with iteration (guaranteed vectorization)

The binding operators extend this philosophy to storage and data flow:

| Operator | Name | Semantics | Guarantee |
|----------|------|-----------|-----------|
| `:=` | Location binding | Introduces storage, mutable | Compiler allocates |
| `=` | Value binding | SSA, immutable | No storage, single assignment |
| `<-` `->` | Location mutation | Mutates existing location | Cannot create storage |
| `<\|` `\|>` | Fusion chain | Must fuse, no intermediates | Compiler errors if unfusible |

**Referential Transparency**: With these operators, you can see at a glance:
- WHERE storage is being created (`:=`)
- WHAT is pure computation (`=`)
- WHEN mutation occurs (`<-`)
- WHICH operations are fused (`<|`, `|>`)

### 1.1 Current State Analysis

**`:=` (COLONEQ)**: Currently ONLY used for tine declarations
```rake
| #miss := (disc < <0.0>)   ~~ creates mask storage
```
Location: `lib/parser.mly:269`, `lib/lexer.mll:117`

**`=` (EQ)**: Used for let bindings (SSA)
```rake
let x = expr   ~~ SSA value, no storage
```
Location: `lib/parser.mly:496-501`

**`<-` (ASSIGN)**: Used for assignment (currently allows creating storage!)
```rake
d <- ax * bx + ay * by   ~~ BUG: can create 'd' if not declared
```
Location: `lib/parser.mly:456`, `lib/mlir.ml:552-554`

**`->` (ARROW)**: Used for result bindings and type arrows
```rake
crunch dot ax ay -> d:   ~~ declares result binding
through #maybe: ... -> t_value   ~~ captures through result
```
Location: `lib/parser.mly:79`, multiple uses

**`|>` (PIPE)**: Defined but not implemented
```rake
x |> normalize |> clamp   ~~ should fuse
```
Location: `lib/parser.mly:510-513`, `lib/mlir.ml:505-506` (failwith)

**`<|`**: NOT IMPLEMENTED - needs to be added

### 1.2 Lexer Changes

Location: `lib/lexer.mll`

Add `<|` token after other multi-character operators (~line 121):
```ocaml
| "<|" { FUSED_LEFT }
```

Update `show_token` to include new token.

### 1.3 Parser Changes

Location: `lib/parser.mly`

**1.3.1 Add token declaration** (~line 54):
```ocaml
%token FUSED_LEFT  (* <| *)
```

**1.3.2 Add precedence** (~line 80, with PIPE):
```ocaml
%left PIPE FUSED_LEFT
```

**1.3.3 Add fused binding syntax**:

Option A: Standalone fused binding statement
```ocaml
stmt:
  | PIPE_CHAR name = IDENT FUSED_LEFT e = expr {
      mk_node (SFused { fused_name = name; fused_expr = e }) $startpos $endpos
    }
```

Option B: Fused pipeline expression (complements `|>`)
```ocaml
expr_pipe:
  | l = expr_pipe FUSED_LEFT r = expr_or {
      mk_node (EFusedPipe (l, r)) $startpos $endpos  (* right-to-left *)
    }
```

**1.3.4 Generalize `:=` for location bindings** (beyond tines):
```ocaml
stmt:
  | name = IDENT COLONEQ e = expr {
      mk_node (SLocBind { loc_name = name; loc_expr = e }) $startpos $endpos
    }
```

### 1.4 AST Changes

Location: `lib/ast.ml`

**1.4.1 Add new statement kinds** (~line 231):
```ocaml
and stmt_kind =
  | SLet of binding                    (** let x = e (SSA, immutable) *)
  | SLocBind of loc_binding            (** x := e (introduces storage) *)
  | SAssign of ident * expr            (** x <- e (mutates existing) *)
  | SFused of fused_binding            (** | x <| e (must fuse) *)
  | SExpr of expr
  | SOver of over_loop

and loc_binding = {
  loc_name: ident;
  loc_type: typ option;
  loc_expr: expr;
}

and fused_binding = {
  fused_name: ident;
  fused_expr: expr;
}
```

**1.4.2 Add fused pipeline expression** (~line 107):
```ocaml
| EFusedPipe of expr * expr          (** x <| f (right-to-left, must fuse) *)
```

### 1.5 Type Checker Enforcement

Location: `lib/typecheck.ml`

**1.5.1 Track storage locations** - add to env:
```ocaml
type env = {
  types: (ident, t) Hashtbl.t;
  vars: (ident, t) Hashtbl.t;
  tines: (ident, unit) Hashtbl.t;
  funcs: (ident, t list * t) Hashtbl.t;
  locations: (ident, unit) Hashtbl.t;  (* NEW: tracks := bindings *)
}
```

**1.5.2 Enforce `:=` creates new storage**:
```ocaml
| SLocBind lb ->
    if Hashtbl.mem env.locations lb.loc_name then
      type_errorf stmt.loc "Location %s already exists (use <- to mutate)"
        lb.loc_name;
    let t = infer_expr env lb.loc_expr in
    Hashtbl.add env.locations lb.loc_name ();
    Hashtbl.add env.vars lb.loc_name t;
    env
```

**1.5.3 Enforce `<-` requires existing storage**:
```ocaml
| SAssign (name, e) ->
    if not (Hashtbl.mem env.locations name) then
      type_errorf stmt.loc "Cannot assign to %s: not a location (use := first)"
        name;
    let _ = infer_expr env e in
    env
```

**1.5.4 Enforce `=` is SSA** (no reassignment):
```ocaml
| SLet binding ->
    if Hashtbl.mem env.vars binding.bind_name then
      type_errorf stmt.loc "Cannot rebind %s (SSA violation)"
        binding.bind_name;
    (* ... rest unchanged ... *)
```

**1.5.5 Verify fusion is possible for `<|`/`|>`**:
```ocaml
let rec is_fusible env expr =
  match expr.v with
  | EInt _ | EFloat _ | EBool _ | EVar _ | EScalarVar _ -> true
  | EBinop (l, _, r) -> is_fusible env l && is_fusible env r
  | EUnop (_, e) -> is_fusible env e
  | ECall (name, args) ->
      (* Only pure functions are fusible *)
      let is_pure = List.mem name ["sqrt"; "sin"; "cos"; "abs"; (* etc *)] in
      is_pure && List.for_all (is_fusible env) args
  | EField (e, _) -> is_fusible env e
  | EBroadcast e -> is_fusible env e
  | EReduce (_, e) -> is_fusible env e
  | _ -> false  (* Side-effecting or unknown expressions are not fusible *)

| SFused fb ->
    if not (is_fusible env fb.fused_expr) then
      type_errorf stmt.loc "Expression cannot be fused (contains side effects)"
        fb.fused_name;
    let t = infer_expr env fb.fused_expr in
    Hashtbl.add env.vars fb.fused_name t;
    env
```

### 1.6 MLIR Emitter Changes

Location: `lib/mlir.ml`

**1.6.1 `:=` emits allocation**:
```ocaml
| SLocBind lb ->
    let (v, t) = emit_expr ctx lb.loc_expr in
    (* For now, locations are SSA values that can be reassigned via Hashtbl.replace *)
    (* Future: emit memref.alloc for true mutable storage *)
    Hashtbl.add ctx.vars lb.loc_name v;
    Hashtbl.add ctx.type_env.vars lb.loc_name t;
    emit ctx "// Location %s := %s" lb.loc_name v
```

**1.6.2 `<-` emits store** (verify target exists):
```ocaml
| SAssign (name, e) ->
    let (v, _) = emit_expr ctx e in
    if not (Hashtbl.mem ctx.vars name) then
      failwith (Printf.sprintf "Cannot assign to undefined location: %s" name);
    Hashtbl.replace ctx.vars name v
    (* Future: emit memref.store for true mutable storage *)
```

**1.6.3 `<|`/`|>` must NOT materialize**:
```ocaml
| SFused fb ->
    (* Fused bindings are just SSA - the guarantee is in the type checker *)
    let (v, t) = emit_expr ctx fb.fused_expr in
    Hashtbl.add ctx.vars fb.fused_name v;
    Hashtbl.add ctx.type_env.vars fb.fused_name t
    (* No comment - these are invisible in the output *)

| EFusedPipe (l, r) ->
    (* Desugar to application, fusion guaranteed by type checker *)
    emit_expr ctx { expr with v = ECall (get_func_name r, [l]) }
```

### 1.7 Migration Path

**Phase A**: Add `<|` token and grammar (backward compatible)
**Phase B**: Add `SLocBind` and `SFused` to AST
**Phase C**: Add type checker enforcement (may break existing code)
**Phase D**: Update all examples to use correct operators
**Phase E**: Update MLIR emitter

### 1.8 Example: Before and After

**Before** (current syntax, ambiguous semantics):
```rake
crunch dot ax ay az bx by bz -> d:
  d <- ax * bx + ay * by + az * bz
```

**After** (explicit guarantees):
```rake
crunch dot (ax ay az bx by bz) -> d:
  | d <| ax * bx + ay * by + az * bz
```

Or with `results` keyword:
```rake
crunch dot (ax ay az bx by bz):
  | results <| ax * bx + ay * by + az * bz
```

**Rake example** (showing all operators):
```rake
rake intersect ray <sphere> -> t:
  ~~ Setup: value bindings (SSA, =)
  let oc = ray.origin - <sphere.center>
  let a = dot(ray.dir, ray.dir)
  let b = <2.0> * dot(oc, ray.dir)
  let c = dot(oc, oc) - <sphere.r> * <sphere.r>
  let disc = b * b - <4.0> * a * c

  ~~ Tines: location bindings (:=)
  | #miss  := (disc < <0.0>)
  | #maybe := (!#miss)

  ~~ Through: fused chains (<|)
  through #maybe:
    | sqrt_disc <| sqrt(disc)
    | t <| (-b - sqrt_disc) / (<2.0> * a)
    t
  -> t_value

  sweep:
    | #miss  -> <-1.0>
    | #maybe -> t_value
  -> t
```

---

## Phase 2: Stop Silent Failures [DONE]

**Goal**: Fail loudly on unhandled cases instead of producing wrong code.

### 2.1 Remove catch-all in `emit_expr` [DONE]

Replaced catch-all `| _ ->` with explicit `failwith` for each unhandled
expression type (ELambda, EPipe, EWith, EExtract, EInsert, EScan, EShuffle,
EShift, ERotate, EGather, EScatter, ECompress, EExpand, ETines, EFma, EOuter).

### 2.2 Fix field access emission [DONE]

Changed field access fallback from emitting comment + undefined SSA to
`failwith` with clear error message.

### 2.3 Verify function call arguments in type checker [DONE]

Added argument count and type verification in `ECall` handler. Now checks
parameter count matches and types are compatible via `compatible` function.

---

## Phase 3: Complete Core Emission

### 3.1 Fix record emission

Location: `lib/mlir.ml:323-337`

Current just emits a comment. Options:

**Option A**: Emit as separate SSA values (current implicit approach, make
explicit)

```ocaml
(* Records decompose to individual field values in SSA *)
(* Store field->value mappings for later field access *)
```

**Option B**: Use MLIR tuple type

```mlir
%rec = tuple.from_elements %field1, %field2 : tuple<f32, f32>
```

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

- `EOuter` (line 270): Outer product type is matrix — implement properly or
  error
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

Add warning/error if sweep tines don't cover all cases and no catch-all `_`
present.

In `check_sweep`:

```ocaml
let has_catchall = List.exists (fun arm -> arm.arm_tine = None) sw.sweep_arms in
if not has_catchall then
  (* Could warn, or require explicit catch-all *)
  ()
```

### 4.2 Document tine priority semantics

Current: first matching tine wins (due to select chain order). Document this in
spec or add mutual-exclusivity check.

### 4.3 Update spec for `arith.select` vs `vector.mask`

The current implementation uses `arith.select` for through blocks. This is
correct for side-effect-free code. Update `docs/spec/03_tines_and_through.md` to
reflect the actual implementation

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

IDEALLY: Use an actual code coverage tool (or create one for rake) to show
coverage of unit tests over language grammar / features.

---

## Phase 6: Quality of Life

NOTE: For all controls, we must carefully consider where they should be
permitted. For example, branching conditions inside a sweep could break
vectorization. Validate through the test suite that vectorized code is always
generated for rakes. If this can't be done with conditionals, the grammar should
have a separate mode inside of rakes that does not allow the breaking
constructs.

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

Type checker lists: `sqrt`, `sin`, `cos`, `tan`, `exp`, `log`, `abs`, `floor`,
`ceil`, `min`, `max`, `pow`, `atan2`

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

| Date    | Decision                                         | Rationale                                  |
| ------- | ------------------------------------------------ | ------------------------------------------ |
| 2024-12 | Use `arith.select` for through blocks            | Simpler emission, equivalent for pure code |
| 2024-12 | Prioritize `scf.parallel` over fixing type holes | Enables GPU path, higher impact            |
| TBD     | Scalar mode as default, vector as optimization   | Lets MLIR handle parallelization strategy  |
