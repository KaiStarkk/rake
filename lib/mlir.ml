(* MLIR Emission for rake *)
(* Uses vector, arith, func, and scf dialects *)

open Ast
open Printf

let vector_width = 8 (* AVX2 default, configurable *)

type context = {
  buf : Buffer.t;
  mutable counter : int;
  mutable indent : int;
  mutable funcs : (string * Types.t) list; (* function name -> return type *)
  mutable values : (string * string) list; (* variable name -> SSA value *)
}

let fresh ctx prefix =
  ctx.counter <- ctx.counter + 1;
  sprintf "%%%s%d" prefix ctx.counter

let emit ctx fmt =
  for _ = 1 to ctx.indent do
    Buffer.add_string ctx.buf "  "
  done;
  Printf.kbprintf (fun _ -> Buffer.add_char ctx.buf '\n') ctx.buf fmt

(* MLIR type emission *)
let mlir_scalar_type = function
  | Types.SFloat -> "f32"
  | Types.SDouble -> "f64"
  | Types.SInt -> "i32"
  | Types.SInt8 -> "i8"
  | Types.SInt16 -> "i16"
  | Types.SInt64 -> "i64"
  | Types.SUint -> "i32"
  | Types.SUint8 -> "i8"
  | Types.SUint16 -> "i16"
  | Types.SUint64 -> "i64"
  | Types.SBool -> "i1"

let rec mlir_type = function
  | Types.Rack s -> sprintf "vector<%dx%s>" vector_width (mlir_scalar_type s)
  | Types.CompoundRack Types.CVec3 -> sprintf "!rake.vec3rack<%d>" vector_width
  | Types.CompoundRack Types.CVec4 -> sprintf "!rake.vec4rack<%d>" vector_width
  | Types.CompoundRack _ -> sprintf "vector<%dxf32>" vector_width
  | Types.Scalar s -> mlir_scalar_type s
  | Types.CompoundScalar _ -> "!rake.vec3"
  | Types.Mask -> sprintf "vector<%dxi1>" vector_width
  | Types.Pack (name, _) -> sprintf "!rake.soa_%s" name
  | Types.Aos (name, _) -> sprintf "!rake.aos_%s" name
  | Types.Single (name, _) -> sprintf "!rake.single_%s" name
  | Types.Stack (inner, _) -> sprintf "!rake.stack<%s>" (mlir_type inner)
  | Types.Array (inner, _) -> sprintf "memref<?x%s>" (mlir_type inner)
  | Types.Fun (args, ret) ->
      sprintf "(%s) -> %s"
        (String.concat ", " (List.map mlir_type args))
        (mlir_type ret)
  | Types.Tuple ts ->
      sprintf "tuple<%s>" (String.concat ", " (List.map mlir_type ts))
  | Types.Unit -> "()"
  | Types.Unknown -> "!rake.unknown"

let mlir_prim_type p =
  sprintf "vector<%dx%s>" vector_width (mlir_scalar_type (Types.of_prim p))

(* Emit a splat (broadcast scalar to vector) *)
let emit_splat ctx scalar_val scalar_type =
  let result = fresh ctx "splat" in
  let vec_type = sprintf "vector<%dx%s>" vector_width scalar_type in
  emit ctx "%s = vector.splat %s : %s" result scalar_val vec_type;
  result

(* Emit vector binary operation *)
let emit_binop ctx is_float op lhs rhs vec_type =
  let result = fresh ctx "r" in
  let mlir_op =
    match op with
    | Add -> if is_float then "arith.addf" else "arith.addi"
    | Sub -> if is_float then "arith.subf" else "arith.subi"
    | Mul -> if is_float then "arith.mulf" else "arith.muli"
    | Div -> if is_float then "arith.divf" else "arith.divsi"
    | Mod -> if is_float then "arith.remf" else "arith.remsi"
    | And -> "arith.andi"
    | Or -> "arith.ori"
    | _ -> "arith.addi" (* comparisons handled separately *)
  in
  emit ctx "%s = %s %s, %s : %s" result mlir_op lhs rhs vec_type;
  result

(* Emit comparison *)
let emit_cmp ctx is_float op lhs rhs vec_type =
  let result = fresh ctx "cmp" in
  let predicate, dialect =
    match op with
    | Lt -> if is_float then ("olt", "arith.cmpf") else ("slt", "arith.cmpi")
    | Le -> if is_float then ("ole", "arith.cmpf") else ("sle", "arith.cmpi")
    | Gt -> if is_float then ("ogt", "arith.cmpf") else ("sgt", "arith.cmpi")
    | Ge -> if is_float then ("oge", "arith.cmpf") else ("sge", "arith.cmpi")
    | Eq -> if is_float then ("oeq", "arith.cmpf") else ("eq", "arith.cmpi")
    | Ne -> if is_float then ("one", "arith.cmpf") else ("ne", "arith.cmpi")
    | _ -> ("eq", "arith.cmpi")
  in
  emit ctx "%s = %s %s, %s, %s : %s" result dialect predicate lhs rhs vec_type;
  result

(* Expression emission - returns (value_name, type) *)
let rec emit_expr ctx env (e : expr) : string * Types.t =
  match e.v with
  | EInt n ->
      let scalar = fresh ctx "c" in
      emit ctx "%s = arith.constant %Ld : i32" scalar n;
      let vec = emit_splat ctx scalar "i32" in
      (vec, Types.Rack Types.SInt)
  | EFloat f ->
      let scalar = fresh ctx "c" in
      emit ctx "%s = arith.constant %f : f32" scalar f;
      let vec = emit_splat ctx scalar "f32" in
      (vec, Types.Rack Types.SFloat)
  | EBool b ->
      let scalar = fresh ctx "c" in
      let v = if b then "1" else "0" in
      emit ctx "%s = arith.constant %s : i1" scalar v;
      let vec = emit_splat ctx scalar "i1" in
      (vec, Types.Mask)
  | EVar name ->
      let t =
        match Hashtbl.find_opt env name with
        | Some t -> t
        | None -> (
            (* Could be a function - look in ctx.funcs *)
            try List.assoc name ctx.funcs with Not_found -> Types.Unknown)
      in
      (* Check if there's an SSA value mapping for this name *)
      let ssa_val =
        try List.assoc name ctx.values with Not_found -> sprintf "%%%s" name
      in
      (ssa_val, t)
  | EScalarVar name ->
      (* Scalar variable - broadcast to vector *)
      let t =
        match Hashtbl.find_opt env name with
        | Some t -> t
        | None -> Types.Scalar Types.SFloat
      in
      let vec =
        emit_splat ctx (sprintf "%%%s" name)
          (mlir_scalar_type
             (match t with Types.Scalar s -> s | _ -> Types.SFloat))
      in
      (vec, Types.broadcast t)
  | EBinop (l, op, r) -> (
      (* Handle Pipe specially before evaluating - needs AST nodes for fusion *)
      match op with
      | Pipe ->
          emit ctx "// Pipeline via binop";
          emit_pipe_fused ctx env l r
      | _ -> (
          let lv, lt = emit_expr ctx env l in
          let rv, rt = emit_expr ctx env r in
          (* Determine if either operand is float for type coercion *)
          let is_float_l =
            match lt with
            | Types.Rack Types.SFloat | Types.Rack Types.SDouble -> true
            | _ -> false
          in
          let is_float_r =
            match rt with
            | Types.Rack Types.SFloat | Types.Rack Types.SDouble -> true
            | _ -> false
          in
          let is_float = is_float_l || is_float_r in
          (* Coerce int to float if needed for mixed comparisons *)
          let lv, lt =
            if is_float && not is_float_l then (
              let conv = fresh ctx "sitofp" in
              emit ctx "%s = arith.sitofp %s : vector<%dxi32> to vector<%dxf32>"
                conv lv vector_width vector_width;
              (conv, Types.Rack Types.SFloat))
            else (lv, lt)
          in
          let rv =
            if is_float && not is_float_r then (
              let conv = fresh ctx "sitofp" in
              emit ctx "%s = arith.sitofp %s : vector<%dxi32> to vector<%dxf32>"
                conv rv vector_width vector_width;
              conv)
            else rv
          in
          let vec_type = mlir_type lt in
          match op with
          | Lt | Le | Gt | Ge | Eq | Ne ->
              let result = emit_cmp ctx is_float op lv rv vec_type in
              (result, Types.Mask)
          | _ ->
              let result = emit_binop ctx is_float op lv rv vec_type in
              (result, lt)))
  | EUnop (Neg, e) ->
      let v, t = emit_expr ctx env e in
      let result = fresh ctx "neg" in
      let zero = fresh ctx "zero" in
      emit ctx "%s = arith.constant 0 : i32" zero;
      let zero_vec = emit_splat ctx zero "i32" in
      emit ctx "%s = arith.subi %s, %s : %s" result zero_vec v (mlir_type t);
      (result, t)
  | EUnop (FNeg, e) ->
      let v, t = emit_expr ctx env e in
      let result = fresh ctx "fneg" in
      emit ctx "%s = arith.negf %s : %s" result v (mlir_type t);
      (result, t)
  | EUnop (Not, e) ->
      let v, t = emit_expr ctx env e in
      let result = fresh ctx "not" in
      let ones = fresh ctx "ones" in
      emit ctx "%s = arith.constant 1 : i1" ones;
      let ones_vec = emit_splat ctx ones "i1" in
      emit ctx "%s = arith.xori %s, %s : %s" result v ones_vec (mlir_type t);
      (result, t)
  | ECall (name, args) -> (
      let arg_vals = List.map (emit_expr ctx env) args in
      let result = fresh ctx "call" in
      (* Arguments without type annotations *)
      let args_str = String.concat ", " (List.map fst arg_vals) in
      let arg_types_str =
        String.concat ", " (List.map (fun (_, t) -> mlir_type t) arg_vals)
      in
      (* Handle built-in math functions *)
      match name with
      | "sqrt" ->
          let v, t = List.hd arg_vals in
          emit ctx "%s = math.sqrt %s : %s" result v (mlir_type t);
          (result, t)
      | "exp" ->
          let v, t = List.hd arg_vals in
          emit ctx "%s = math.exp %s : %s" result v (mlir_type t);
          (result, t)
      | "log" ->
          let v, t = List.hd arg_vals in
          emit ctx "%s = math.log %s : %s" result v (mlir_type t);
          (result, t)
      | "sin" ->
          let v, t = List.hd arg_vals in
          emit ctx "%s = math.sin %s : %s" result v (mlir_type t);
          (result, t)
      | "cos" ->
          let v, t = List.hd arg_vals in
          emit ctx "%s = math.cos %s : %s" result v (mlir_type t);
          (result, t)
      | "abs" ->
          let v, t = List.hd arg_vals in
          emit ctx "%s = math.absf %s : %s" result v (mlir_type t);
          (result, t)
      | "floor" ->
          let v, t = List.hd arg_vals in
          emit ctx "%s = math.floor %s : %s" result v (mlir_type t);
          (result, t)
      | "ceil" ->
          let v, t = List.hd arg_vals in
          emit ctx "%s = math.ceil %s : %s" result v (mlir_type t);
          (result, t)
      | "min" when List.length arg_vals = 2 ->
          let v1, t = List.nth arg_vals 0 in
          let v2, _ = List.nth arg_vals 1 in
          emit ctx "%s = arith.minimumf %s, %s : %s" result v1 v2 (mlir_type t);
          (result, t)
      | "max" when List.length arg_vals = 2 ->
          let v1, t = List.nth arg_vals 0 in
          let v2, _ = List.nth arg_vals 1 in
          emit ctx "%s = arith.maximumf %s, %s : %s" result v1 v2 (mlir_type t);
          (result, t)
      | _ ->
          (* User-defined function call *)
          let ret_type =
            try
              match List.assoc name ctx.funcs with
              | Types.Fun (_, ret) -> ret
              | t -> t
            with Not_found -> Types.Rack Types.SFloat
          in
          emit ctx "%s = func.call @%s(%s) : (%s) -> %s" result name args_str
            arg_types_str (mlir_type ret_type);
          (result, ret_type))
  | ELet (b, body) ->
      let v, t = emit_expr ctx env b.bind_expr in
      (* In SSA form, map the binding name to the computed value *)
      Hashtbl.add env b.bind_name t;
      ctx.values <- (b.bind_name, v) :: ctx.values;
      emit ctx "// let %s = %s" b.bind_name v;
      emit_expr ctx env body
  | EFun (_params, body) ->
      (* Lambda - just emit the body for now *)
      emit_expr ctx env body
  | ERails rails -> emit_rails ctx env rails
  | ELanes ->
      let result = fresh ctx "lanes" in
      emit ctx "%s = arith.constant %d : i32" result vector_width;
      (result, Types.Scalar Types.SInt)
  | ELaneIndex ->
      let result = fresh ctx "idx" in
      emit ctx "%s = vector.step : vector<%dxi32>" result vector_width;
      (result, Types.Rack Types.SInt)
  | EReduce (op, e) ->
      let v, _t = emit_expr ctx env e in
      let result = fresh ctx "reduce" in
      let reduction =
        match op with
        | RAdd -> "vector.reduction <add>"
        | RMul -> "vector.reduction <mul>"
        | RMin -> "vector.reduction <minimumf>"
        | RMax -> "vector.reduction <maximumf>"
        | RAnd -> "vector.reduction <and>"
        | ROr -> "vector.reduction <or>"
      in
      emit ctx "%s = %s, %s : vector<%dxf32> into f32" result reduction v
        vector_width;
      (result, Types.Scalar Types.SFloat)
  | EBroadcast e -> (
      let v, t = emit_expr ctx env e in
      (* If already a vector, return as-is *)
      match t with
      | Types.Scalar s ->
          let vec = emit_splat ctx v (mlir_scalar_type s) in
          (vec, Types.Rack s)
      | _ -> (v, t))
  | ERetire ->
      emit ctx "// retire - lane mask update";
      ("", Types.Unit)
  | EPipe (e1, e2) ->
      (* Pipeline fusion: transform x |> f(args) into f(x, args) *)
      emit ctx "// Pipeline fusion";
      emit_pipe_fused ctx env e1 e2
  | EField (e, field) -> (
      (* Field access - emit struct access *)
      let v, t = emit_expr ctx env e in
      emit ctx "// field access .%s on %s" field v;
      let result = fresh ctx "field" in
      match t with
      | Types.Pack (_, fields) | Types.Aos (_, fields) | Types.Single (_, fields)
        ->
          (* Struct field access - look up field type *)
          let rec find_field_idx idx = function
            | [] -> (0, Types.Rack Types.SFloat)
            | (name, ft) :: rest ->
                if name = field then (idx, ft)
                else find_field_idx (idx + 1) rest
          in
          let idx, field_type = find_field_idx 0 fields in
          let op_name =
            match t with
            | Types.Pack _ -> "racks.soa_field"
            | Types.Aos _ -> "racks.aos_field"
            | Types.Single _ -> "racks.single_field"
            | _ -> "racks.field"
          in
          emit ctx "%s = %s %s[%d] : %s -> %s" result op_name v idx
            (mlir_type t) (mlir_type field_type);
          (result, field_type)
      | Types.CompoundRack c ->
          (* Built-in vector component access (.x, .y, .z, .w) *)
          let idx =
            match field with
            | "x" -> 0
            | "y" -> 1
            | "z" -> 2
            | "w" -> 3
            | _ -> 0
          in
          let max_idx =
            match c with
            | Types.CVec2 -> 1
            | Types.CVec3 -> 2
            | Types.CVec4 -> 3
            | _ -> 3
          in
          let idx = min idx max_idx in
          emit ctx "%s = vector.extract %s[%d] : %s" result v idx (mlir_type t);
          (result, Types.Rack Types.SFloat)
      | Types.CompoundScalar c ->
          let idx =
            match field with
            | "x" -> 0
            | "y" -> 1
            | "z" -> 2
            | "w" -> 3
            | _ -> 0
          in
          let max_idx =
            match c with
            | Types.CVec2 -> 1
            | Types.CVec3 -> 2
            | Types.CVec4 -> 3
            | _ -> 3
          in
          let idx = min idx max_idx in
          emit ctx "%s = llvm.extractvalue %s[%d] : %s" result v idx
            (mlir_type t);
          (result, Types.Scalar Types.SFloat)
      | _ ->
          emit ctx "// field access on unsupported type %s" (Types.show t);
          emit ctx "%s = arith.constant 0.0 : f32" result;
          let vec = emit_splat ctx result "f32" in
          (vec, Types.Rack Types.SFloat))
  | _ ->
      emit ctx "// unimplemented expression";
      let r = fresh ctx "undef" in
      emit ctx "%s = arith.constant 0.0 : f32" r;
      let vec = emit_splat ctx r "f32" in
      (vec, Types.Unknown)

(* Pipeline fusion - the key optimization for rake performance *)
(* Transforms: x |> f |> g |> h into nested inlined calls *)
and emit_pipe_fused ctx env e1 e2 =
  (* First, collect the entire pipeline chain *)
  let rec collect_chain expr acc =
    match expr.v with
    | EPipe (left, right) -> collect_chain left (right :: acc)
    | _ -> (expr, acc)
  in
  let base_expr, chain = collect_chain e1 [ e2 ] in

  (* Emit the base expression *)
  let base_val, base_type = emit_expr ctx env base_expr in
  emit ctx "// Pipeline chain: %d stages" (List.length chain);

  (* Process each stage in the chain, threading the value through *)
  let rec process_chain current_val current_type = function
    | [] -> (current_val, current_type)
    | stage :: rest ->
        let stage_val, stage_type =
          emit_pipe_stage ctx env current_val current_type stage
        in
        process_chain stage_val stage_type rest
  in
  process_chain base_val base_type chain

(* Emit a single pipeline stage - fuses the piped value into the call *)
and emit_pipe_stage ctx env piped_val piped_type stage =
  match stage.v with
  | ECall (name, args) -> (
      (* Key fusion: x |> f(a, b) becomes f(x, a, b) *)
      emit ctx "// Fused: %s piped into %s()" piped_val name;

      (* Emit argument values *)
      let arg_vals = List.map (emit_expr ctx env) args in

      (* Build fused argument list: piped value first, then explicit args *)
      let all_args = (piped_val, piped_type) :: arg_vals in
      let args_str = String.concat ", " (List.map fst all_args) in
      let arg_types_str =
        String.concat ", " (List.map (fun (_, t) -> mlir_type t) all_args)
      in

      let result = fresh ctx "fused" in

      (* Handle built-in functions with piped argument *)
      match name with
      | "sqrt" ->
          emit ctx "%s = math.sqrt %s : %s" result piped_val
            (mlir_type piped_type);
          (result, piped_type)
      | "exp" ->
          emit ctx "%s = math.exp %s : %s" result piped_val
            (mlir_type piped_type);
          (result, piped_type)
      | "log" ->
          emit ctx "%s = math.log %s : %s" result piped_val
            (mlir_type piped_type);
          (result, piped_type)
      | "sin" ->
          emit ctx "%s = math.sin %s : %s" result piped_val
            (mlir_type piped_type);
          (result, piped_type)
      | "cos" ->
          emit ctx "%s = math.cos %s : %s" result piped_val
            (mlir_type piped_type);
          (result, piped_type)
      | "abs" ->
          emit ctx "%s = math.absf %s : %s" result piped_val
            (mlir_type piped_type);
          (result, piped_type)
      | "floor" ->
          emit ctx "%s = math.floor %s : %s" result piped_val
            (mlir_type piped_type);
          (result, piped_type)
      | "ceil" ->
          emit ctx "%s = math.ceil %s : %s" result piped_val
            (mlir_type piped_type);
          (result, piped_type)
      | "min" when List.length args = 1 ->
          let v2, _ = List.hd arg_vals in
          emit ctx "%s = arith.minimumf %s, %s : %s" result piped_val v2
            (mlir_type piped_type);
          (result, piped_type)
      | "max" when List.length args = 1 ->
          let v2, _ = List.hd arg_vals in
          emit ctx "%s = arith.maximumf %s, %s : %s" result piped_val v2
            (mlir_type piped_type);
          (result, piped_type)
      | _ ->
          (* User-defined function - emit fused call *)
          let ret_type =
            try
              match List.assoc name ctx.funcs with
              | Types.Fun (_, ret) -> ret
              | t -> t
            with Not_found -> piped_type (* Assume same type if unknown *)
          in
          emit ctx "%s = func.call @%s(%s) : (%s) -> %s" result name args_str
            arg_types_str (mlir_type ret_type);
          (result, ret_type))
  | EVar name -> (
      (* x |> f where f is a function name - becomes f(x) *)
      emit ctx "// Fused: %s piped into function %s" piped_val name;
      let result = fresh ctx "fused" in

      (* Handle built-in unary functions *)
      match name with
      | "sqrt" ->
          emit ctx "%s = math.sqrt %s : %s" result piped_val
            (mlir_type piped_type);
          (result, piped_type)
      | "exp" ->
          emit ctx "%s = math.exp %s : %s" result piped_val
            (mlir_type piped_type);
          (result, piped_type)
      | "log" ->
          emit ctx "%s = math.log %s : %s" result piped_val
            (mlir_type piped_type);
          (result, piped_type)
      | "sin" ->
          emit ctx "%s = math.sin %s : %s" result piped_val
            (mlir_type piped_type);
          (result, piped_type)
      | "cos" ->
          emit ctx "%s = math.cos %s : %s" result piped_val
            (mlir_type piped_type);
          (result, piped_type)
      | "abs" ->
          emit ctx "%s = math.absf %s : %s" result piped_val
            (mlir_type piped_type);
          (result, piped_type)
      | "floor" ->
          emit ctx "%s = math.floor %s : %s" result piped_val
            (mlir_type piped_type);
          (result, piped_type)
      | "ceil" ->
          emit ctx "%s = math.ceil %s : %s" result piped_val
            (mlir_type piped_type);
          (result, piped_type)
      | _ ->
          (* User-defined function *)
          let ret_type =
            try
              match List.assoc name ctx.funcs with
              | Types.Fun (_, ret) -> ret
              | t -> t
            with Not_found -> piped_type
          in
          emit ctx "%s = func.call @%s(%s) : (%s) -> %s" result name piped_val
            (mlir_type piped_type) (mlir_type ret_type);
          (result, ret_type))
  | EFun (params, body) -> (
      (* x |> fun y -> expr  - inline the lambda *)
      emit ctx "// Fused: inline lambda";
      (* Bind the piped value to the first parameter *)
      match params with
      | PVar (param_name, _) :: _ | PScalar (param_name, _) :: _ ->
          Hashtbl.add env param_name piped_type;
          ctx.values <- (param_name, piped_val) :: ctx.values;
          emit_expr ctx env body
      | [] -> emit_expr ctx env body)
  | EBinop (_, op, _) ->
      (* x |> (+ y) - partial application, emit the binop *)
      let rv, _rt = emit_expr ctx env stage in
      let is_float =
        match piped_type with
        | Types.Rack Types.SFloat | Types.Rack Types.SDouble -> true
        | _ -> false
      in
      let result =
        emit_binop ctx is_float op piped_val rv (mlir_type piped_type)
      in
      ( result,
        if op = Lt || op = Le || op = Gt || op = Ge || op = Eq || op = Ne then
          Types.Mask
        else piped_type )
  | _ ->
      (* Fallback: evaluate stage and combine *)
      emit ctx "// Pipeline stage fallback";
      emit_expr ctx env stage

(* Emit rails as nested selects *)
and emit_rails ctx env rails =
  match rails with
  | [] -> ("", Types.Unit)
  | [ r ] ->
      (* Single rail - just emit body *)
      emit_expr ctx env r.v.rail_body
  | _ ->
      (* Multiple rails - chain of selects *)
      (* Strategy: evaluate all, then select from last to first *)
      (* Last rail (usually "otherwise") is the default, then we wrap conditionals *)
      let rail_data =
        List.map
          (fun r ->
            let mask, _ = emit_rail_cond ctx env r.v.rail_cond in
            let body_val, body_type = emit_expr ctx env r.v.rail_body in
            (r.v.rail_cond, mask, body_val, body_type))
          rails
      in

      (* Build selection chain: start from last (default), wrap with earlier conditions *)
      let rec build_select = function
        | [] -> ("", Types.Unit)
        | [ (_, _, val_, typ) ] ->
            (val_, typ) (* Base case: otherwise/last rail *)
        | (cond, mask, val_, typ) :: rest -> (
            let rest_val, _ = build_select rest in
            (* For "otherwise", just use the value directly *)
            match cond with
            | RCOtherwise -> (val_, typ)
            | _ ->
                let result = fresh ctx "sel" in
                emit ctx "%s = arith.select %s, %s, %s : %s, %s" result mask
                  val_ rest_val (mlir_type Types.Mask) (mlir_type typ);
                (result, typ))
      in
      build_select rail_data

and emit_rail_cond ctx env = function
  | RCNamed (_name, pred) -> emit_pred ctx env pred
  | RCAnon pred -> emit_pred ctx env pred
  | RCOtherwise ->
      (* All lanes active *)
      let c = fresh ctx "true" in
      emit ctx "%s = arith.constant 1 : i1" c;
      let vec = emit_splat ctx c "i1" in
      (vec, Types.Mask)
  | RCRef name -> (sprintf "%%%s" name, Types.Mask)

and emit_pred ctx env (p : predicate) : string * Types.t =
  match p.v with
  | PExpr e -> emit_expr ctx env e
  | PIs (l, r) ->
      let lv, lt = emit_expr ctx env l in
      let rv, rt = emit_expr ctx env r in
      let is_float =
        match (lt, rt) with
        | Types.Rack Types.SFloat, _
        | _, Types.Rack Types.SFloat
        | Types.Rack Types.SDouble, _
        | _, Types.Rack Types.SDouble ->
            true
        | _ -> false
      in
      let rv =
        if
          is_float && match rt with Types.Rack Types.SInt -> true | _ -> false
        then (
          let conv = fresh ctx "sitofp" in
          emit ctx "%s = arith.sitofp %s : vector<%dxi32> to vector<%dxf32>"
            conv rv vector_width vector_width;
          conv)
        else rv
      in
      let result = emit_cmp ctx is_float Eq lv rv (mlir_type lt) in
      (result, Types.Mask)
  | PIsNot (l, r) ->
      let lv, lt = emit_expr ctx env l in
      let rv, rt = emit_expr ctx env r in
      let is_float =
        match (lt, rt) with
        | Types.Rack Types.SFloat, _
        | _, Types.Rack Types.SFloat
        | Types.Rack Types.SDouble, _
        | _, Types.Rack Types.SDouble ->
            true
        | _ -> false
      in
      let rv =
        if
          is_float && match rt with Types.Rack Types.SInt -> true | _ -> false
        then (
          let conv = fresh ctx "sitofp" in
          emit ctx "%s = arith.sitofp %s : vector<%dxi32> to vector<%dxf32>"
            conv rv vector_width vector_width;
          conv)
        else rv
      in
      let result = emit_cmp ctx is_float Ne lv rv (mlir_type lt) in
      (result, Types.Mask)
  | PCmp (l, op, r) ->
      let lv, lt = emit_expr ctx env l in
      let rv, rt = emit_expr ctx env r in
      (* Type coercion for mixed int/float comparisons *)
      let is_float_l =
        match lt with
        | Types.Rack Types.SFloat | Types.Rack Types.SDouble -> true
        | _ -> false
      in
      let is_float_r =
        match rt with
        | Types.Rack Types.SFloat | Types.Rack Types.SDouble -> true
        | _ -> false
      in
      let is_float = is_float_l || is_float_r in
      let lv, lt =
        if is_float && not is_float_l then (
          let conv = fresh ctx "sitofp" in
          emit ctx "%s = arith.sitofp %s : vector<%dxi32> to vector<%dxf32>"
            conv lv vector_width vector_width;
          (conv, Types.Rack Types.SFloat))
        else (lv, lt)
      in
      let rv =
        if is_float && not is_float_r then (
          let conv = fresh ctx "sitofp" in
          emit ctx "%s = arith.sitofp %s : vector<%dxi32> to vector<%dxf32>"
            conv rv vector_width vector_width;
          conv)
        else rv
      in
      let ast_op =
        match op with CLt -> Lt | CLe -> Le | CGt -> Gt | CGe -> Ge
      in
      let result = emit_cmp ctx is_float ast_op lv rv (mlir_type lt) in
      (result, Types.Mask)
  | PAnd (l, r) ->
      let lv, _ = emit_pred ctx env l in
      let rv, _ = emit_pred ctx env r in
      let result = fresh ctx "and" in
      emit ctx "%s = arith.andi %s, %s : vector<%dxi1>" result lv rv
        vector_width;
      (result, Types.Mask)
  | POr (l, r) ->
      let lv, _ = emit_pred ctx env l in
      let rv, _ = emit_pred ctx env r in
      let result = fresh ctx "or" in
      emit ctx "%s = arith.ori %s, %s : vector<%dxi1>" result lv rv vector_width;
      (result, Types.Mask)
  | PNot p ->
      let v, _ = emit_pred ctx env p in
      let result = fresh ctx "not" in
      let c = fresh ctx "ones" in
      emit ctx "%s = arith.constant 1 : i1" c;
      let ones = emit_splat ctx c "i1" in
      emit ctx "%s = arith.xori %s, %s : vector<%dxi1>" result v ones
        vector_width;
      (result, Types.Mask)

(* Emit a function definition *)
let emit_func ctx check_env name sig_ body is_rake =
  let ret_type = Check.check_type check_env sig_.sig_return in

  (* Extract parameter names from body *)
  let body_params = match body.v with EFun (params, _) -> params | _ -> [] in

  (* Build parameter list with actual names *)
  let params =
    List.mapi
      (fun i (_, typ) ->
        let param_type = Check.check_type check_env typ in
        let param_name =
          match List.nth_opt body_params i with
          | Some (PVar (n, _)) -> n
          | Some (PScalar (n, _)) -> n
          | None -> sprintf "arg%d" i
        in
        (param_name, param_type))
      sig_.sig_params
  in

  let params_str =
    String.concat ", "
      (List.map (fun (n, t) -> sprintf "%%%s: %s" n (mlir_type t)) params)
  in

  emit ctx "";
  emit ctx "// %s function: %s" (if is_rake then "rake" else "crunch") name;
  emit ctx "func.func @%s(%s) -> %s {" name params_str (mlir_type ret_type);
  ctx.indent <- ctx.indent + 1;

  (* Reset value mappings for new function scope *)
  ctx.values <- [];

  (* Create local environment with parameters *)
  let local_env = Hashtbl.create 16 in
  List.iter (fun (n, t) -> Hashtbl.add local_env n t) params;

  (* Get actual body (unwrap EFun) *)
  let actual_body = match body.v with EFun (_, b) -> b | _ -> body in

  let result, _ = emit_expr ctx local_env actual_body in
  emit ctx "func.return %s : %s" result (mlir_type ret_type);
  ctx.indent <- ctx.indent - 1;
  emit ctx "}";

  (* Record function for call resolution *)
  ctx.funcs <- (name, Types.Fun (List.map snd params, ret_type)) :: ctx.funcs

(* Emit SoA struct type definition *)
let emit_soa_type ctx name fields check_env =
  emit ctx "";
  emit ctx "// SoA type: %s" name;
  emit ctx "// Each field is a vector of %d elements" vector_width;
  let field_types =
    List.map
      (fun f ->
        let t = Check.check_type check_env f.field_type in
        sprintf "%s: %s" f.field_name (mlir_type t))
      fields
  in
  emit ctx "// !rake.soa_%s = { %s }" name (String.concat ", " field_types)

(* Emit a definition *)
let emit_def ctx check_env (d : def) =
  match d.v with
  | DSoa (name, fields) -> emit_soa_type ctx name fields check_env
  | DAos (name, _fields) ->
      emit ctx "";
      emit ctx "// AoS type: %s" name
  | DSingle (name, _fields) ->
      emit ctx "";
      emit ctx "// Single type: %s" name
  | DType (name, _t) ->
      emit ctx "";
      emit ctx "// Type alias: %s" name
  | DCrunch (name, sig_, body) -> emit_func ctx check_env name sig_ body false
  | DRake (name, sig_, body) -> emit_func ctx check_env name sig_ body true
  | DRun (name, _sig_, _body) ->
      emit ctx "";
      emit ctx "// Run block: %s (sequential, not yet implemented)" name

(* Main entry point *)
let emit_program (check_env : Check.env) (p : program) : string =
  let ctx =
    {
      buf = Buffer.create 8192;
      counter = 0;
      indent = 0;
      funcs = [];
      values = [];
    }
  in

  (* Header *)
  emit ctx "// rake Language - Generated MLIR";
  emit ctx "// Target: %d-wide vectors (AVX2)" vector_width;
  emit ctx "// Dialects: func, arith, vector, scf";
  emit ctx "";
  emit ctx "module {";
  ctx.indent <- 1;

  (* Definitions *)
  List.iter
    (fun m ->
      emit ctx "";
      emit ctx "// Module: %s" m.mod_name;
      List.iter (emit_def ctx check_env) m.mod_defs)
    p;

  ctx.indent <- 0;
  emit ctx "}";

  Buffer.contents ctx.buf
