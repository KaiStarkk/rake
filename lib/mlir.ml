(** Rake 0.2.0 MLIR Emitter

    Generates MLIR using the vector dialect for SIMD operations.

    Key mappings:
    - Tine predicates → arith.cmpf/cmpi → vector<8xi1>
    - Through blocks → vector.mask with passthru
    - Sweep → nested arith.select
    - Scalars → vector.broadcast
    - Reductions → vector.reduction

    Target dialects: func, arith, vector, math
*)

open Ast
open Types

(** Vector width (default AVX2) *)
let vector_width = 8

(** Emission context *)
type ctx = {
  mutable buf: Buffer.t;
  mutable indent: int;
  mutable ssa_counter: int;
  vars: (string, string) Hashtbl.t;  (* Rake var -> MLIR SSA name *)
  tines: (string, string) Hashtbl.t; (* Tine name -> mask SSA name *)
  types: (string, Types.t) Hashtbl.t; (* Type definitions *)
  type_env: Typecheck.env;
  pack_memrefs: (string, string) Hashtbl.t;  (* "pack.field" -> memref SSA name *)
  mutable current_over_offset: string option; (* Current over loop offset for loads *)
  mutable current_over_mask: string option;   (* Current tail mask for masked stores *)
  mutable output_memref: string option;        (* Output memref for run function results *)
}

let create_ctx env = {
  buf = Buffer.create 4096;
  indent = 0;
  ssa_counter = 0;
  vars = Hashtbl.create 64;
  tines = Hashtbl.create 16;
  types = env.Typecheck.types;
  type_env = env;
  pack_memrefs = Hashtbl.create 32;
  current_over_offset = None;
  current_over_mask = None;
  output_memref = None;
}

(** Generate fresh SSA name *)
let fresh ctx prefix =
  let n = ctx.ssa_counter in
  ctx.ssa_counter <- n + 1;
  Printf.sprintf "%%%s%d" prefix n

(** Emit with indentation *)
let emit ctx fmt =
  for _ = 1 to ctx.indent do
    Buffer.add_string ctx.buf "  "
  done;
  Printf.kbprintf (fun _ -> Buffer.add_char ctx.buf '\n') ctx.buf fmt

(** Emit without newline *)
let emit_inline ctx fmt =
  for _ = 1 to ctx.indent do
    Buffer.add_string ctx.buf "  "
  done;
  Printf.kbprintf (fun _ -> ()) ctx.buf fmt

(** MLIR type for Rake type *)
let rec mlir_type = function
  | Rack SFloat -> Printf.sprintf "vector<%dxf32>" vector_width
  | Rack SDouble -> Printf.sprintf "vector<%dxf64>" vector_width
  | Rack SInt -> Printf.sprintf "vector<%dxi32>" vector_width
  | Rack SInt64 -> Printf.sprintf "vector<%dxi64>" vector_width
  | Rack SBool -> Printf.sprintf "vector<%dxi1>" vector_width
  | Scalar SFloat -> "f32"
  | Scalar SDouble -> "f64"
  | Scalar SInt -> "i32"
  | Scalar SInt64 -> "i64"
  | Scalar SBool -> "i1"
  | Mask -> Printf.sprintf "vector<%dxi1>" vector_width
  | Stack (_name, fields) ->
      (* For now, emit as tuple of vectors *)
      let field_types = List.map (fun (_, t) -> mlir_type t) fields in
      "tuple<" ^ String.concat ", " field_types ^ ">"
  | Single (_name, fields) ->
      (* Single types are tuples of scalars *)
      let field_types = List.map (fun (_, t) -> mlir_type t) fields in
      "tuple<" ^ String.concat ", " field_types ^ ">"
  | Tuple ts ->
      "tuple<" ^ String.concat ", " (List.map mlir_type ts) ^ ">"
  | Unit -> "()"
  | _ -> "!rake.unknown"

(** MLIR type for vector of floats *)
let vec_f32 = Printf.sprintf "vector<%dxf32>" vector_width
let vec_i1 = Printf.sprintf "vector<%dxi1>" vector_width

(** Emit a binary operation *)
let emit_binop ctx op t1 t2 lhs rhs =
  let result = fresh ctx "v" in
  let is_float = match t1 with
    | Rack SFloat | Rack SDouble | Scalar SFloat | Scalar SDouble -> true
    | _ -> false
  in
  let op_name = match op with
    | Add -> if is_float then "arith.addf" else "arith.addi"
    | Sub -> if is_float then "arith.subf" else "arith.subi"
    | Mul -> if is_float then "arith.mulf" else "arith.muli"
    | Div -> if is_float then "arith.divf" else "arith.divsi"
    | Mod -> "arith.remsi"
    | Lt -> if is_float then "arith.cmpf olt" else "arith.cmpi slt"
    | Le -> if is_float then "arith.cmpf ole" else "arith.cmpi sle"
    | Gt -> if is_float then "arith.cmpf ogt" else "arith.cmpi sgt"
    | Ge -> if is_float then "arith.cmpf oge" else "arith.cmpi sge"
    | Eq -> if is_float then "arith.cmpf oeq" else "arith.cmpi eq"
    | Ne -> if is_float then "arith.cmpf one" else "arith.cmpi ne"
    | And -> "arith.andi"
    | Or -> "arith.ori"
    | _ -> "arith.addf"  (* TODO: handle other ops *)
  in
  let result_type = match op with
    | Lt | Le | Gt | Ge | Eq | Ne -> vec_i1
    | _ -> mlir_type (binop_result t1 t2)
  in
  emit ctx "%s = %s %s, %s : %s" result op_name lhs rhs result_type;
  result

(** Emit a unary operation *)
let emit_unop ctx op t arg =
  let result = fresh ctx "v" in
  let ty = mlir_type t in
  (match op with
   | Neg | FNeg ->
       emit ctx "%s = arith.negf %s : %s" result arg ty
   | Not ->
       (* For masks, use xor with all-ones *)
       let ones = fresh ctx "ones" in
       emit ctx "%s = arith.constant dense<true> : %s" ones vec_i1;
       emit ctx "%s = arith.xori %s, %s : %s" result arg ones vec_i1);
  result

(** Emit broadcast of scalar to vector (only if needed) *)
let emit_broadcast ctx scalar_val scalar_type =
  match scalar_type with
  | Rack _ | Mask ->
      (* Already a vector, no broadcast needed *)
      scalar_val
  | _ ->
      let result = fresh ctx "bcast" in
      let vec_type = match scalar_type with
        | Scalar SFloat -> vec_f32
        | Scalar SInt -> Printf.sprintf "vector<%dxi32>" vector_width
        | _ -> vec_f32
      in
      emit ctx "%s = vector.broadcast %s : %s to %s" result scalar_val
        (mlir_type scalar_type) vec_type;
      result

(** Emit expression, return SSA name and type *)
let rec emit_expr ctx (expr: Ast.expr) : string * Types.t =
  match expr.v with
  | EInt n ->
      let result = fresh ctx "c" in
      (* In vector context, treat integers as floats for arithmetic *)
      emit ctx "%s = arith.constant dense<%Ld.0> : %s" result n vec_f32;
      (result, Rack SFloat)

  | EFloat f ->
      let result = fresh ctx "c" in
      (* Use proper float formatting to ensure decimal point *)
      let f_str = if Float.is_integer f then Printf.sprintf "%.1f" f else Printf.sprintf "%g" f in
      emit ctx "%s = arith.constant dense<%s> : %s" result f_str vec_f32;
      (result, Rack SFloat)

  | EBool b ->
      let result = fresh ctx "c" in
      emit ctx "%s = arith.constant dense<%b> : %s" result b vec_i1;
      (result, Mask)

  | EVar name -> (
      match Hashtbl.find_opt ctx.vars name with
      | Some ssa_name ->
          let t = match Hashtbl.find_opt ctx.type_env.vars name with
            | Some t -> t
            | None -> Rack SFloat
          in
          (ssa_name, t)
      | None ->
          (* Undefined variable - shouldn't happen after type checking *)
          let result = fresh ctx "undef" in
          emit ctx "%s = arith.constant dense<0.0> : %s" result vec_f32;
          (result, Rack SFloat))

  | EScalarVar name -> (
      match Hashtbl.find_opt ctx.vars name with
      | Some ssa_name ->
          let t = match Hashtbl.find_opt ctx.type_env.vars name with
            | Some t -> t
            | None -> Scalar SFloat
          in
          (ssa_name, t)
      | None ->
          let result = fresh ctx "undef" in
          emit ctx "%s = arith.constant 0.0 : f32" result;
          (result, Scalar SFloat))

  | EBinop (l, op, r) ->
      let (lhs, lt) = emit_expr ctx l in
      let (rhs, rt) = emit_expr ctx r in
      (* Handle broadcast if mixing scalar and rack *)
      let (lhs', lt') = match (lt, rt) with
        | (Scalar _, Rack _) -> (emit_broadcast ctx lhs lt, broadcast lt)
        | _ -> (lhs, lt)
      in
      let (rhs', _rt') = match (lt, rt) with
        | (Rack _, Scalar _) -> (emit_broadcast ctx rhs rt, broadcast rt)
        | _ -> (rhs, rt)
      in
      let result = emit_binop ctx op lt' rt lhs' rhs' in
      let result_t = match op with
        | Lt | Le | Gt | Ge | Eq | Ne -> Mask
        | _ -> binop_result lt' rt
      in
      (result, result_t)

  | EUnop (op, e) ->
      let (arg, t) = emit_expr ctx e in
      let result = emit_unop ctx op t arg in
      (result, t)

  | ECall (name, args) ->
      (* Handle built-in math functions *)
      let emit_math_call fname arg =
        let result = fresh ctx "r" in
        emit ctx "%s = math.%s %s : %s" result fname arg vec_f32;
        (result, Rack SFloat)
      in
      (match (name, args) with
       | ("sqrt", [e]) ->
           let (arg, _) = emit_expr ctx e in
           emit_math_call "sqrt" arg
       | ("sin", [e]) ->
           let (arg, _) = emit_expr ctx e in
           emit_math_call "sin" arg
       | ("cos", [e]) ->
           let (arg, _) = emit_expr ctx e in
           emit_math_call "cos" arg
       | ("abs", [e]) ->
           let (arg, _) = emit_expr ctx e in
           emit_math_call "absf" arg
       | _ ->
           (* User-defined function call *)
           let arg_results = List.map (emit_expr ctx) args in
           let arg_vals = List.map fst arg_results in
           let arg_types = List.map (fun (_, t) -> mlir_type t) arg_results in
           let result = fresh ctx "call" in
           emit ctx "%s = func.call @%s(%s) : (%s) -> %s"
             result name
             (String.concat ", " arg_vals)
             (String.concat ", " arg_types)
             vec_f32;
           (result, Rack SFloat))

  | EField (e, field) ->
      (* First check if this is a direct variable field access (e.g., chunk.ox) *)
      (match e.v with
       | EVar var_name ->
           (* Try chunk.field format first (for over loop bindings) *)
           let chunk_field_key = var_name ^ "." ^ field in
           (match Hashtbl.find_opt ctx.vars chunk_field_key with
            | Some ssa -> (ssa, Rack SFloat)  (* TODO: proper field type *)
            | None ->
                (* Fall back to base_field format *)
                let (base, t) = emit_expr ctx e in
                let field_var = base ^ "_" ^ field in
                (match Hashtbl.find_opt ctx.vars field_var with
                 | Some ssa -> (ssa, get_field_type t field)
                 | None ->
                     let result = fresh ctx field in
                     emit ctx "// Field access: %s.%s" base field;
                     (result, get_field_type t field)))
       | _ ->
           let (base, t) = emit_expr ctx e in
           let field_var = base ^ "_" ^ field in
           (match Hashtbl.find_opt ctx.vars field_var with
            | Some ssa -> (ssa, get_field_type t field)
            | None ->
                let result = fresh ctx field in
                emit ctx "// Field access: %s.%s" base field;
                (result, get_field_type t field)))

  | EBroadcast e ->
      let (val_, t) = emit_expr ctx e in
      let result = emit_broadcast ctx val_ t in
      (result, broadcast t)

  | EReduce (op, e) ->
      let (arg, _) = emit_expr ctx e in
      let result = fresh ctx "red" in
      let op_name = match op with
        | RAdd -> "add"
        | RMul -> "mul"
        | RMin -> "minimumf"
        | RMax -> "maximumf"
        | _ -> "add"
      in
      emit ctx "%s = vector.reduction <%s>, %s : %s into f32"
        result op_name arg vec_f32;
      (result, Scalar SFloat)

  | ELaneIndex ->
      let result = fresh ctx "idx" in
      emit ctx "%s = vector.step : vector<%dxi32>" result vector_width;
      (result, Rack SInt)

  | ELanes ->
      let result = fresh ctx "w" in
      emit ctx "%s = arith.constant %d : i32" result vector_width;
      (result, Scalar SInt)

  | ERecord (name, inits) ->
      (* For now, emit each field as a separate value and create a tuple *)
      let field_vals = List.map (fun init ->
        let (v, _) = emit_expr ctx init.init_value in
        v
      ) inits in
      let result = fresh ctx "rec" in
      emit ctx "// Record %s: %s" name (String.concat ", " field_vals);
      (* Store fields for later access *)
      List.iter2 (fun init v ->
        Hashtbl.add ctx.vars (result ^ "_" ^ init.init_field) v
      ) inits field_vals;
      (result, match Hashtbl.find_opt ctx.types name with
       | Some t -> t
       | None -> Unknown)

  | ETuple es ->
      let vals = List.map (fun e -> emit_expr ctx e) es in
      let result = fresh ctx "tup" in
      emit ctx "// Tuple: %s" (String.concat ", " (List.map fst vals));
      (result, Tuple (List.map snd vals))

  | EUnit -> ("%unit", Unit)

  | ELet (binding, body) ->
      let (v, t) = emit_expr ctx binding.bind_expr in
      Hashtbl.add ctx.vars binding.bind_name v;
      Hashtbl.add ctx.type_env.vars binding.bind_name t;
      emit_expr ctx body

  | _ ->
      (* Fallback for unhandled expressions *)
      let result = fresh ctx "todo" in
      emit ctx "%s = arith.constant dense<0.0> : %s" result vec_f32;
      (result, Rack SFloat)

and get_field_type t field =
  match t with
  | Stack (_, fields) | Single (_, fields) -> (
      match List.assoc_opt field fields with
      | Some ft -> ft
      | None -> Rack SFloat)
  | _ -> Rack SFloat

(** Emit a statement *)
let rec emit_stmt ctx (stmt: Ast.stmt) =
  match stmt.v with
  | SLet binding ->
      let (v, t) = emit_expr ctx binding.bind_expr in
      Hashtbl.add ctx.vars binding.bind_name v;
      Hashtbl.add ctx.type_env.vars binding.bind_name t

  | SAssign (name, e) ->
      let (v, _) = emit_expr ctx e in
      Hashtbl.replace ctx.vars name v

  | SExpr e ->
      ignore (emit_expr ctx e)

  | SOver over ->
      (* Emit scf.for loop over pack in stack-sized chunks *)
      emit_over_loop ctx over

(** Emit over loop: scf.for iterating over pack in chunks of lanes *)
and emit_over_loop ctx (over: Ast.over_loop) =
  (* Get count expression and its type - handle scalar variable specially *)
  let (count_val, count_type) = match over.over_count.v with
    | EScalarVar name ->
        let v = match Hashtbl.find_opt ctx.vars name with
         | Some v -> v
         | None -> failwith ("Undefined count variable: " ^ name)
        in
        let t = match Hashtbl.find_opt ctx.type_env.vars name with
         | Some t -> t
         | None -> Scalar SFloat
        in
        (v, t)
    | _ -> emit_expr ctx over.over_count
  in

  (* Constants for loop bounds *)
  let zero = fresh ctx "zero" in
  let one = fresh ctx "one" in
  let lanes_val = fresh ctx "lanes" in
  let num_iters = fresh ctx "niters" in

  emit ctx "%s = arith.constant 0 : index" zero;
  emit ctx "%s = arith.constant 1 : index" one;
  emit ctx "%s = arith.constant %d : index" lanes_val vector_width;

  (* Cast count to index - handle both integer and float types *)
  let count_idx = fresh ctx "count_idx" in
  (match count_type with
   | Scalar SInt64 ->
       emit ctx "%s = arith.index_cast %s : i64 to index" count_idx count_val
   | Scalar SInt ->
       emit ctx "%s = arith.index_cast %s : i32 to index" count_idx count_val
   | Scalar SFloat ->
       let count_i64 = fresh ctx "count_i64" in
       emit ctx "%s = arith.fptosi %s : f32 to i64" count_i64 count_val;
       emit ctx "%s = arith.index_cast %s : i64 to index" count_idx count_i64
   | _ ->
       emit ctx "%s = arith.index_cast %s : i64 to index" count_idx count_val);

  (* Compute number of full iterations: ceil(count / lanes) *)
  let count_plus = fresh ctx "count_plus" in
  let lanes_minus_one = fresh ctx "lanes_m1" in
  emit ctx "%s = arith.constant %d : index" lanes_minus_one (vector_width - 1);
  emit ctx "%s = arith.addi %s, %s : index" count_plus count_idx lanes_minus_one;
  emit ctx "%s = arith.divui %s, %s : index" num_iters count_plus lanes_val;

  (* Emit scf.for loop *)
  let iter_var = fresh ctx "i" in
  emit ctx "scf.for %s = %s to %s step %s {" iter_var zero num_iters one;
  ctx.indent <- ctx.indent + 1;

  (* Compute offset for this iteration *)
  let offset = fresh ctx "offset" in
  emit ctx "%s = arith.muli %s, %s : index" offset iter_var lanes_val;

  (* Store offset for field access *)
  ctx.current_over_offset <- Some offset;

  (* Compute tail mask for last iteration *)
  let remaining = fresh ctx "remaining" in
  let mask = fresh ctx "mask" in

  emit ctx "%s = arith.subi %s, %s : index" remaining count_idx offset;
  emit ctx "%s = vector.create_mask %s : %s" mask remaining vec_i1;

  (* Store mask in context for masked operations *)
  ctx.current_over_mask <- Some mask;

  (* Load chunk fields from pack memrefs *)
  (* Look up pack type to get field names *)
  (match Hashtbl.find_opt ctx.type_env.vars over.over_pack with
   | Some (Pack (_, fields)) ->
       List.iter (fun (field_name, _field_type) ->
         let memref_key = over.over_pack ^ "." ^ field_name in
         match Hashtbl.find_opt ctx.pack_memrefs memref_key with
         | Some memref ->
             let loaded = fresh ctx field_name in
             emit ctx "%s = vector.load %s[%s] : memref<?xf32>, %s" loaded memref offset vec_f32;
             (* Bind chunk.field_name to loaded value *)
             Hashtbl.add ctx.vars (over.over_chunk ^ "." ^ field_name) loaded
         | None ->
             emit ctx "// Warning: no memref for %s" memref_key
       ) fields
   | _ ->
       emit ctx "// Warning: pack type not found for %s" over.over_pack);

  (* Emit body statements, capture result of last expression *)
  let result_val = ref None in
  List.iter (fun stmt ->
    match stmt.Ast.v with
    | Ast.SExpr e ->
        let (v, _) = emit_expr ctx e in
        result_val := Some v
    | _ ->
        emit_stmt ctx stmt
  ) over.over_body;

  (* Store result to output memref if available *)
  (match (!result_val, ctx.output_memref) with
   | (Some result, Some out_memref) ->
       emit ctx "vector.maskedstore %s[%s], %s, %s : memref<?xf32>, vector<8xi1>, %s"
         out_memref offset mask result vec_f32
   | (Some result, None) ->
       emit ctx "// Result %s computed but no output memref" result
   | _ ->
       emit ctx "// No result expression in over loop body");

  ctx.current_over_offset <- None;
  ctx.current_over_mask <- None;
  ctx.indent <- ctx.indent - 1;
  emit ctx "}"

(** Emit predicate, return SSA name of mask *)
let rec emit_predicate ctx (pred: Ast.predicate) : string =
  match pred.v with
  | PExpr e ->
      fst (emit_expr ctx e)

  | PCmp (l, op, r) ->
      let (lhs, lt) = emit_expr ctx l in
      let (rhs, rt) = emit_expr ctx r in
      (* Handle broadcast if needed *)
      let (lhs', rhs') = match (lt, rt) with
        | (Scalar _, Rack _) -> (emit_broadcast ctx lhs lt, rhs)
        | (Rack _, Scalar _) -> (lhs, emit_broadcast ctx rhs rt)
        | _ -> (lhs, rhs)
      in
      let result = fresh ctx "cmp" in
      let op_str = match op with
        | CLt -> "olt" | CLe -> "ole" | CGt -> "ogt"
        | CGe -> "oge" | CEq -> "oeq" | CNe -> "one"
      in
      emit ctx "%s = arith.cmpf %s, %s, %s : %s" result op_str lhs' rhs' vec_f32;
      result

  | PIs (l, r) | PIsNot (l, r) ->
      let (lhs, _) = emit_expr ctx l in
      let (rhs, _) = emit_expr ctx r in
      let result = fresh ctx "is" in
      let cmp = match pred.v with PIs _ -> "oeq" | _ -> "one" in
      emit ctx "%s = arith.cmpf %s, %s, %s : %s" result cmp lhs rhs vec_f32;
      result

  | PAnd (l, r) ->
      let lm = emit_predicate ctx l in
      let rm = emit_predicate ctx r in
      let result = fresh ctx "and" in
      emit ctx "%s = arith.andi %s, %s : %s" result lm rm vec_i1;
      result

  | POr (l, r) ->
      let lm = emit_predicate ctx l in
      let rm = emit_predicate ctx r in
      let result = fresh ctx "or" in
      emit ctx "%s = arith.ori %s, %s : %s" result lm rm vec_i1;
      result

  | PNot p ->
      let m = emit_predicate ctx p in
      let ones = fresh ctx "ones" in
      let result = fresh ctx "not" in
      emit ctx "%s = arith.constant dense<true> : %s" ones vec_i1;
      emit ctx "%s = arith.xori %s, %s : %s" result m ones vec_i1;
      result

  | PTineRef name -> (
      match Hashtbl.find_opt ctx.tines name with
      | Some ssa -> ssa
      | None -> failwith ("Reference to undefined tine: " ^ name))

(** Emit tine declaration *)
let emit_tine ctx (tine: Ast.tine) =
  let mask = emit_predicate ctx tine.tine_pred in
  Hashtbl.add ctx.tines tine.tine_name mask;
  emit ctx "// Tine #%s = %s" tine.tine_name mask

(** Emit through block *)
let emit_through ctx (th: Ast.through) =
  (* Get the mask for this through block *)
  let mask = match th.through_tine with
    | TRSingle name -> (
        match Hashtbl.find_opt ctx.tines name with
        | Some m -> m
        | None -> failwith ("Through references undefined tine: " ^ name))
    | TRComposed pred -> emit_predicate ctx pred
  in

  (* Emit passthru value if present, otherwise use zero *)
  let passthru = match th.through_passthru with
    | Some e -> fst (emit_expr ctx e)
    | None ->
        let z = fresh ctx "zero" in
        emit ctx "%s = arith.constant dense<0.0> : %s" z vec_f32;
        z
  in

  (* Emit body statements *)
  List.iter (emit_stmt ctx) th.through_body;

  (* Emit result expression *)
  let (result_val, result_t) = emit_expr ctx th.through_result in

  (* Use vector.mask to apply the mask *)
  let masked = fresh ctx "masked" in
  let result_type = mlir_type result_t in
  emit ctx "%s = arith.select %s, %s, %s : %s, %s"
    masked mask result_val passthru vec_i1 result_type;

  (* Store the result binding *)
  Hashtbl.add ctx.vars th.through_binding masked;
  Hashtbl.add ctx.type_env.vars th.through_binding result_t;
  emit ctx "// Through -> %s = %s" th.through_binding masked

(** Emit sweep block *)
let emit_sweep ctx (sw: Ast.sweep) =
  (* Build nested select chain from last to first *)
  let rec build_select arms acc =
    match arms with
    | [] -> acc
    | arm :: rest ->
        let (val_, _) = emit_expr ctx arm.arm_value in
        let result = fresh ctx "sel" in
        (match arm.arm_tine with
         | Some name -> (
             match Hashtbl.find_opt ctx.tines name with
             | Some mask ->
                 emit ctx "%s = arith.select %s, %s, %s : %s, %s"
                   result mask val_ acc vec_i1 vec_f32;
                 build_select rest result
             | None ->
                 build_select rest val_)
         | None ->
             (* Catch-all: use this value as default *)
             build_select rest val_)
  in
  (* Start with undefined/zero and build up *)
  let init = fresh ctx "undef" in
  emit ctx "%s = arith.constant dense<0.0> : %s" init vec_f32;
  let result = build_select (List.rev sw.sweep_arms) init in
  Hashtbl.add ctx.vars sw.sweep_binding result;
  emit ctx "// Sweep -> %s = %s" sw.sweep_binding result;
  result

(** Emit crunch function *)
let emit_crunch ctx name params _result body =
  (* Function signature *)
  let param_strs = List.mapi (fun i p ->
    let pname = match p with PRack (n, _) | PScalar (n, _) -> n in
    let pty = vec_f32 in  (* TODO: proper type *)
    Hashtbl.add ctx.vars pname (Printf.sprintf "%%arg%d" i);
    Printf.sprintf "%%arg%d: %s" i pty
  ) params in

  emit ctx "func.func @%s(%s) -> %s attributes {llvm.alwaysinline} {" name (String.concat ", " param_strs) vec_f32;
  ctx.indent <- ctx.indent + 1;

  (* Emit body *)
  List.iter (emit_stmt ctx) body;

  (* Find result variable and return it *)
  let ret_val = match Hashtbl.find_opt ctx.vars _result.result_name with
    | Some v -> v
    | None -> "%arg0"
  in
  emit ctx "func.return %s : %s" ret_val vec_f32;

  ctx.indent <- ctx.indent - 1;
  emit ctx "}"

(** Emit rake function *)
let emit_rake ctx name params _result setup tines throughs sweep =
  (* Compute result type *)
  let result_type = match Typecheck.find_type ctx.type_env _result.result_name with
    | Some t -> mlir_type t
    | None -> vec_f32
  in

  (* Function signature *)
  let param_strs = List.mapi (fun i p ->
    let pname = match p with PRack (n, _) | PScalar (n, _) -> n in
    let is_scalar = match p with PScalar _ -> true | _ -> false in
    let pty = if is_scalar then "f32" else vec_f32 in
    let arg = Printf.sprintf "%%arg%d" i in
    Hashtbl.add ctx.vars pname arg;
    Hashtbl.add ctx.type_env.vars pname
      (if is_scalar then Scalar SFloat else Rack SFloat);
    Printf.sprintf "%s: %s" arg pty
  ) params in

  emit ctx "func.func @%s(%s) -> %s attributes {llvm.alwaysinline} {" name (String.concat ", " param_strs) result_type;
  ctx.indent <- ctx.indent + 1;

  (* Emit setup statements *)
  List.iter (emit_stmt ctx) setup;

  (* Emit tine declarations *)
  List.iter (emit_tine ctx) tines;

  (* Emit through blocks *)
  List.iter (emit_through ctx) throughs;

  (* Emit sweep *)
  let result = emit_sweep ctx sweep in

  emit ctx "func.return %s : %s" result result_type;

  ctx.indent <- ctx.indent - 1;
  emit ctx "}"

(** Emit a definition *)
let rec emit_def ctx (def: Ast.def) =
  match def.v with
  | DStack (name, fields) ->
      emit ctx "// Stack type: %s" name;
      List.iter (fun f ->
        emit ctx "//   %s: %s" f.field_name (show_typ_kind f.field_type.v)
      ) fields

  | DSingle (name, fields) ->
      emit ctx "// Single type: %s" name;
      List.iter (fun f ->
        emit ctx "//   %s: %s" f.field_name (show_typ_kind f.field_type.v)
      ) fields

  | DType (name, ty) ->
      emit ctx "// Type alias: %s = %s" name (show_typ_kind ty.v)

  | DCrunch (name, params, result, body) ->
      emit_crunch ctx name params result body

  | DRake (name, params, result, setup, tines, throughs, sweep) ->
      emit_rake ctx name params result setup tines throughs sweep

  | DRun (name, params, result, body) ->
      emit_run ctx name params result body

(** Emit run function with pack parameter expansion *)
and emit_run ctx name params _result body =
  (* Expand pack parameters to memrefs, keep scalars as-is *)
  let param_counter = ref 0 in
  let param_strs = List.concat_map (fun p ->
    let pname = match p with PRack (n, _) | PScalar (n, _) -> n in
    let pty_opt = match p with
      | PRack (_, t) -> t
      | PScalar (_, t) -> t
    in
    (* Check if this is a pack parameter *)
    let is_pack = match pty_opt with
      | Some ty -> (match ty.v with TPack _ -> true | _ -> false)
      | None -> false
    in
    let is_scalar = match p with PScalar _ -> true | _ -> false in

    if is_pack then begin
      (* Look up pack type and expand to memrefs for each field *)
      let pack_type_name = match pty_opt with
        | Some ty -> (match ty.v with TPack n -> n | _ -> pname)
        | None -> pname
      in
      match Hashtbl.find_opt ctx.types pack_type_name with
      | Some (Stack (_, fields)) | Some (Pack (_, fields)) ->
          List.map (fun (field_name, _) ->
            let arg_idx = !param_counter in
            incr param_counter;
            let arg = Printf.sprintf "%%arg%d" arg_idx in
            let memref_key = pname ^ "." ^ field_name in
            Hashtbl.add ctx.pack_memrefs memref_key arg;
            (* Also register the pack in vars for type lookup in over loop *)
            Hashtbl.add ctx.type_env.vars pname (Pack (pack_type_name, fields));
            Printf.sprintf "%s: memref<?xf32>" arg
          ) fields
      | _ ->
          let arg_idx = !param_counter in
          incr param_counter;
          let arg = Printf.sprintf "%%arg%d" arg_idx in
          Hashtbl.add ctx.vars pname arg;
          [Printf.sprintf "%s: memref<?xf32>" arg]
    end else if is_scalar then begin
      let arg_idx = !param_counter in
      incr param_counter;
      let arg = Printf.sprintf "%%arg%d" arg_idx in
      Hashtbl.add ctx.vars pname arg;
      (* Determine scalar type from annotation *)
      let scalar_type = match pty_opt with
        | Some ty -> (match ty.v with
            | TScalar PInt64 -> Scalar SInt64
            | TScalar PInt -> Scalar SInt
            | TScalar PDouble -> Scalar SDouble
            | TScalar PBool -> Scalar SBool
            | _ -> Scalar SFloat)
        | None -> Scalar SFloat
      in
      Hashtbl.add ctx.type_env.vars pname scalar_type;
      let mlir_ty = mlir_type scalar_type in
      [Printf.sprintf "%s: %s" arg mlir_ty]
    end else begin
      let arg_idx = !param_counter in
      incr param_counter;
      let arg = Printf.sprintf "%%arg%d" arg_idx in
      Hashtbl.add ctx.vars pname arg;
      Hashtbl.add ctx.type_env.vars pname (Rack SFloat);
      [Printf.sprintf "%s: %s" arg vec_f32]
    end
  ) params in

  (* Add output memref for result *)
  let output_arg = Printf.sprintf "%%arg%d" !param_counter in
  let param_strs = param_strs @ [Printf.sprintf "%s: memref<?xf32>" output_arg] in
  ctx.output_memref <- Some output_arg;

  emit ctx "func.func @%s(%s) {" name (String.concat ", " param_strs);
  ctx.indent <- ctx.indent + 1;

  (* Emit body *)
  List.iter (emit_stmt ctx) body;

  ctx.output_memref <- None;
  emit ctx "func.return";
  ctx.indent <- ctx.indent - 1;
  emit ctx "}"

(** Emit a module *)
let emit_module ctx (m: Ast.module_) =
  emit ctx "// Module: %s" m.mod_name;
  List.iter (emit_def ctx) m.mod_defs

(** Emit a program *)
let emit_program env (prog: Ast.program) =
  let ctx = create_ctx env in
  emit ctx "module {";
  ctx.indent <- 1;
  List.iter (emit_module ctx) prog;
  ctx.indent <- 0;
  emit ctx "}";
  Buffer.contents ctx.buf

(** Main entry point *)
let emit env prog =
  emit_program env prog
