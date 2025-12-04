(* Emit LLVM IR *)

open Ast
open Printf

let vector_width = 8 (* AVX2 default *)

type context = { buf : Buffer.t; mutable counter : int; mutable indent : int }

let fresh ctx =
  ctx.counter <- ctx.counter + 1;
  ctx.counter

let emit ctx fmt =
  for _ = 1 to ctx.indent do
    Buffer.add_string ctx.buf "  "
  done;
  Printf.kbprintf (fun _ -> Buffer.add_char ctx.buf '\n') ctx.buf fmt

let emit_raw ctx fmt = Printf.kbprintf (fun _ -> ()) ctx.buf fmt

(* LLVM type emission *)
let llvm_type = function
  | Types.Rack Types.SFloat -> sprintf "<%d x float>" vector_width
  | Types.Rack Types.SDouble -> sprintf "<%d x double>" vector_width
  | Types.Rack Types.SInt -> sprintf "<%d x i32>" vector_width
  | Types.Rack Types.SInt64 -> sprintf "<%d x i64>" vector_width
  | Types.Rack Types.SUint -> sprintf "<%d x i32>" vector_width
  | Types.Rack Types.SBool -> sprintf "<%d x i1>" vector_width
  | Types.Scalar Types.SFloat -> "float"
  | Types.Scalar Types.SDouble -> "double"
  | Types.Scalar Types.SInt -> "i32"
  | Types.Scalar Types.SInt64 -> "i64"
  | Types.Scalar Types.SUint -> "i32"
  | Types.Scalar Types.SBool -> "i1"
  | Types.Mask -> sprintf "<%d x i1>" vector_width
  | Types.CompoundRack Types.CVec3 ->
      sprintf "{ <%d x float>, <%d x float>, <%d x float> }" vector_width
        vector_width vector_width
  | Types.Unit -> "void"
  | Types.Pack (name, _) -> sprintf "%%struct.%s" name
  | _ -> "i32" (* Fallback *)

let llvm_prim_type = function
  | Float -> sprintf "<%d x float>" vector_width
  | Double -> sprintf "<%d x double>" vector_width
  | Int | Uint -> sprintf "<%d x i32>" vector_width
  | Int64 | Uint64 -> sprintf "<%d x i64>" vector_width
  | Bool -> sprintf "<%d x i1>" vector_width
  | Int8 | Uint8 -> sprintf "<%d x i8>" vector_width
  | Int16 | Uint16 -> sprintf "<%d x i16>" vector_width

let llvm_scalar_type = function
  | Float -> "float"
  | Double -> "double"
  | Int | Uint -> "i32"
  | Int64 | Uint64 -> "i64"
  | Bool -> "i1"
  | Int8 | Uint8 -> "i8"
  | Int16 | Uint16 -> "i16"

(* Binop emission *)
let llvm_binop is_float = function
  | Add -> if is_float then "fadd" else "add"
  | Sub -> if is_float then "fsub" else "sub"
  | Mul -> if is_float then "fmul" else "mul"
  | Div -> if is_float then "fdiv" else "sdiv"
  | Mod -> if is_float then "frem" else "srem"
  | And -> "and"
  | Or -> "or"
  | _ -> "add" (* Comparisons handled separately *)

let llvm_cmp is_float = function
  | Lt -> if is_float then "fcmp olt" else "icmp slt"
  | Le -> if is_float then "fcmp ole" else "icmp sle"
  | Gt -> if is_float then "fcmp ogt" else "icmp sgt"
  | Ge -> if is_float then "fcmp oge" else "icmp sge"
  | Eq -> if is_float then "fcmp oeq" else "icmp eq"
  | Ne -> if is_float then "fcmp one" else "icmp ne"
  | _ -> "icmp eq"

(* Expression emission - returns (value, type) *)
let rec emit_expr ctx env (e : expr) : string * Types.t =
  match e.v with
  | EInt n ->
      let r = fresh ctx in
      let t = sprintf "<%d x i32>" vector_width in
      let splat =
        List.init vector_width (fun _ -> sprintf "i32 %Ld" n)
        |> String.concat ", "
      in
      emit ctx "%%r%d = add %s zeroinitializer, <%s>" r t splat;
      (sprintf "%%r%d" r, Types.Rack Types.SInt)
  | EFloat f ->
      let r = fresh ctx in
      let t = sprintf "<%d x float>" vector_width in
      let splat =
        List.init vector_width (fun _ -> sprintf "float %f" f)
        |> String.concat ", "
      in
      emit ctx "%%r%d = fadd %s zeroinitializer, <%s>" r t splat;
      (sprintf "%%r%d" r, Types.Rack Types.SFloat)
  | EBool b ->
      let r = fresh ctx in
      let t = sprintf "<%d x i1>" vector_width in
      let v = if b then "true" else "false" in
      let splat =
        List.init vector_width (fun _ -> sprintf "i1 %s" v)
        |> String.concat ", "
      in
      emit ctx "%%r%d = add %s zeroinitializer, <%s>" r t splat;
      (sprintf "%%r%d" r, Types.Mask)
  | EVar name ->
      let t = Hashtbl.find env name in
      (sprintf "%%%s" name, t)
  | EScalarVar name ->
      (* Broadcast scalar to vector - scalar idents now use <name> syntax *)
      let t = Hashtbl.find env name in
      let r = fresh ctx in
      let vec_t = llvm_type (Types.broadcast t) in
      let scalar_t = llvm_type t in
      let r1 = fresh ctx in
      emit ctx "%%r%d = insertelement %s undef, %s %%%s, i32 0" r1 vec_t
        scalar_t name;
      let zeros =
        List.init vector_width (fun _ -> "i32 0") |> String.concat ", "
      in
      emit ctx "%%r%d = shufflevector %s %%r%d, %s undef, <%d x i32> <%s>" r
        vec_t r1 vec_t vector_width zeros;
      (sprintf "%%r%d" r, Types.broadcast t)
  | EBinop (l, op, r) -> (
      let lv, lt = emit_expr ctx env l in
      let rv, _rt = emit_expr ctx env r in
      let res = fresh ctx in
      let t = llvm_type lt in
      let is_float =
        match lt with Types.Rack Types.SFloat -> true | _ -> false
      in
      match op with
      | Lt | Le | Gt | Ge | Eq | Ne ->
          emit ctx "%%r%d = %s %s %s, %s" res (llvm_cmp is_float op) t lv rv;
          (sprintf "%%r%d" res, Types.Mask)
      | _ ->
          emit ctx "%%r%d = %s %s %s, %s" res (llvm_binop is_float op) t lv rv;
          (sprintf "%%r%d" res, lt))
  | EUnop (Neg, e) ->
      let v, t = emit_expr ctx env e in
      let res = fresh ctx in
      let lt = llvm_type t in
      emit ctx "%%r%d = sub %s zeroinitializer, %s" res lt v;
      (sprintf "%%r%d" res, t)
  | EUnop (FNeg, e) ->
      let v, t = emit_expr ctx env e in
      let res = fresh ctx in
      let lt = llvm_type t in
      emit ctx "%%r%d = fneg %s %s" res lt v;
      (sprintf "%%r%d" res, t)
  | EUnop (Not, e) ->
      let v, t = emit_expr ctx env e in
      let res = fresh ctx in
      let lt = llvm_type t in
      emit ctx "%%r%d = xor %s %s, <i1 true, ...>" res lt v;
      (* TODO: Proper splat *)
      (sprintf "%%r%d" res, t)
  | ECall (name, args) ->
      let arg_vals = List.map (emit_expr ctx env) args in
      let res = fresh ctx in
      let args_str =
        arg_vals
        |> List.map (fun (v, t) -> sprintf "%s %s" (llvm_type t) v)
        |> String.concat ", "
      in
      (* TODO: Look up return type properly *)
      let ret_t = sprintf "<%d x float>" vector_width in
      emit ctx "%%r%d = call %s @%s(%s)" res ret_t name args_str;
      (sprintf "%%r%d" res, Types.Rack Types.SFloat)
  | EField (e, f) -> (
      let v, t = emit_expr ctx env e in
      let res = fresh ctx in
      match t with
      | Types.Pack (_, fields) | Types.Aos (_, fields) | Types.Single (_, fields)
        ->
          (* Look up field index and type in struct *)
          let rec find_field_idx idx = function
            | [] ->
                (0, Types.Rack Types.SFloat)
                (* Fallback - type checker should catch this *)
            | (name, ft) :: rest ->
                if name = f then (idx, ft) else find_field_idx (idx + 1) rest
          in
          let idx, field_type = find_field_idx 0 fields in
          emit ctx "%%r%d = extractvalue %s %s, %d" res (llvm_type t) v idx;
          (sprintf "%%r%d" res, field_type)
      | Types.CompoundRack _ | Types.CompoundScalar _ ->
          (* Built-in vector components x, y, z, w *)
          let idx =
            match f with "x" -> 0 | "y" -> 1 | "z" -> 2 | "w" -> 3 | _ -> 0
          in
          let field_type =
            match t with
            | Types.CompoundRack _ -> Types.Rack Types.SFloat
            | Types.CompoundScalar _ -> Types.Scalar Types.SFloat
            | _ -> Types.Rack Types.SFloat
          in
          emit ctx "%%r%d = extractvalue %s %s, %d" res (llvm_type t) v idx;
          (sprintf "%%r%d" res, field_type)
      | _ ->
          emit ctx "; field access on unsupported type";
          emit ctx "%%r%d = add i32 0, 0" res;
          (sprintf "%%r%d" res, Types.Unknown))
  | ELet (b, body) ->
      let v, t = emit_expr ctx env b.bind_expr in
      emit ctx "%%%s = add %s %s, zeroinitializer" b.bind_name (llvm_type t) v;
      Hashtbl.add env b.bind_name t;
      emit_expr ctx env body
  | ELetScalar (b, body) ->
      let v, t = emit_expr ctx env b.sbind_expr in
      (* Scalar idents now use <name> syntax, stored directly *)
      emit ctx "%%%s = add %s %s, zeroinitializer" b.sbind_name (llvm_type t) v;
      Hashtbl.add env b.sbind_name t;
      emit_expr ctx env body
  | ERails rails -> emit_rails ctx env rails
  | EFun (params, body) ->
      (* Inline function - just bind params and emit body *)
      let _ = params in
      (* TODO *)
      emit_expr ctx env body
  | ELanes ->
      let r = fresh ctx in
      emit ctx "%%r%d = add i32 0, %d" r vector_width;
      (sprintf "%%r%d" r, Types.Scalar Types.SInt)
  | ELaneIndex ->
      let r = fresh ctx in
      let _indices =
        List.init vector_width string_of_int |> String.concat ", "
      in
      let t = sprintf "<%d x i32>" vector_width in
      emit ctx "%%r%d = add %s zeroinitializer, <%s>" r t
        (List.init vector_width (fun i -> sprintf "i32 %d" i)
        |> String.concat ", ");
      (sprintf "%%r%d" r, Types.Rack Types.SInt)
  | EReduce (op, e) ->
      let v, _t = emit_expr ctx env e in
      let res = fresh ctx in
      let intrinsic =
        match op with
        | RAdd -> "llvm.vector.reduce.fadd"
        | RMul -> "llvm.vector.reduce.fmul"
        | RMin -> "llvm.vector.reduce.fmin"
        | RMax -> "llvm.vector.reduce.fmax"
        | RAnd -> "llvm.vector.reduce.and"
        | ROr -> "llvm.vector.reduce.or"
      in
      emit ctx "%%r%d = call float @%s.v%df32(float 0.0, <%d x float> %s)" res
        intrinsic vector_width vector_width v;
      (sprintf "%%r%d" res, Types.Scalar Types.SFloat)
  | ERetire -> ("void", Types.Unit)
  | _ ->
      let r = fresh ctx in
      emit ctx "; unimplemented expression";
      emit ctx "%%r%d = add i32 0, 0" r;
      (sprintf "%%r%d" r, Types.Unknown)

and emit_rails ctx env rails =
  (* For now, simplified rail emission using select *)
  match rails with
  | [] -> ("void", Types.Unit)
  | [ r ] ->
      (* Single rail - just emit the body *)
      emit_expr ctx env r.v.rail_body
  | _ ->
      (* Multiple rails - chain of selects *)
      let rec process_rails _prev_mask prev_val = function
        | [] -> prev_val
        | r :: rest ->
            let mask, _ = emit_rail_cond ctx env r.v.rail_cond in
            let val_, t = emit_expr ctx env r.v.rail_body in
            let res = fresh ctx in
            let lt = llvm_type t in
            emit ctx "%%r%d = select %s %s, %s %s, %s %s" res
              (sprintf "<%d x i1>" vector_width)
              mask lt val_ lt prev_val;
            process_rails mask (sprintf "%%r%d" res) rest
      in
      let first = List.hd rails in
      let first_val, first_t = emit_expr ctx env first.v.rail_body in
      let result = process_rails "" first_val (List.tl rails) in
      (result, first_t)

and emit_rail_cond ctx env = function
  | RCNamed (name, pred) ->
      let v, t = emit_pred ctx env pred in
      Hashtbl.add env name Types.Mask;
      (* TODO: Store mask for later reference *)
      (v, t)
  | RCAnon pred -> emit_pred ctx env pred
  | RCOtherwise ->
      (* All ones mask *)
      let r = fresh ctx in
      let ones =
        List.init vector_width (fun _ -> "i1 true") |> String.concat ", "
      in
      emit ctx "%%r%d = or <%d x i1> zeroinitializer, <%s>" r vector_width ones;
      (sprintf "%%r%d" r, Types.Mask)
  | RCRef name -> (sprintf "%%%s" name, Types.Mask)

and emit_pred ctx env (p : predicate) : string * Types.t =
  match p.v with
  | PExpr e -> emit_expr ctx env e
  | PIs (l, r) ->
      let lv, lt = emit_expr ctx env l in
      let rv, _ = emit_expr ctx env r in
      let res = fresh ctx in
      emit ctx "%%r%d = fcmp oeq %s %s, %s" res (llvm_type lt) lv rv;
      (sprintf "%%r%d" res, Types.Mask)
  | PIsNot (l, r) ->
      let lv, lt = emit_expr ctx env l in
      let rv, _ = emit_expr ctx env r in
      let res = fresh ctx in
      emit ctx "%%r%d = fcmp one %s %s, %s" res (llvm_type lt) lv rv;
      (sprintf "%%r%d" res, Types.Mask)
  | PCmp (l, op, r) ->
      let lv, lt = emit_expr ctx env l in
      let rv, _ = emit_expr ctx env r in
      let res = fresh ctx in
      let cmp =
        match op with
        | CLt -> "fcmp olt"
        | CLe -> "fcmp ole"
        | CGt -> "fcmp ogt"
        | CGe -> "fcmp oge"
      in
      emit ctx "%%r%d = %s %s %s, %s" res cmp (llvm_type lt) lv rv;
      (sprintf "%%r%d" res, Types.Mask)
  | PAnd (l, r) ->
      let lv, _ = emit_pred ctx env l in
      let rv, _ = emit_pred ctx env r in
      let res = fresh ctx in
      emit ctx "%%r%d = and <%d x i1> %s, %s" res vector_width lv rv;
      (sprintf "%%r%d" res, Types.Mask)
  | POr (l, r) ->
      let lv, _ = emit_pred ctx env l in
      let rv, _ = emit_pred ctx env r in
      let res = fresh ctx in
      emit ctx "%%r%d = or <%d x i1> %s, %s" res vector_width lv rv;
      (sprintf "%%r%d" res, Types.Mask)
  | PNot p ->
      let v, _ = emit_pred ctx env p in
      let res = fresh ctx in
      let ones =
        List.init vector_width (fun _ -> "i1 true") |> String.concat ", "
      in
      emit ctx "%%r%d = xor <%d x i1> %s, <%s>" res vector_width v ones;
      (sprintf "%%r%d" res, Types.Mask)

let emit_def ctx (check_env : Check.env)
    (vars_env : (string, Types.t) Hashtbl.t) (d : def) =
  match d.v with
  | DSoa (name, fields) ->
      emit ctx ";; SoA: %s" name;
      emit ctx "%%struct.%s = type {" name;
      ctx.indent <- ctx.indent + 1;
      List.iteri
        (fun i f ->
          let comma = if i < List.length fields - 1 then "," else "" in
          emit ctx "%s%s  ; %s"
            (llvm_prim_type
               (match f.field_type.v with
               | TRack p -> p
               | TCompoundRack _ -> Float
               | _ -> Float))
            comma f.field_name)
        fields;
      ctx.indent <- ctx.indent - 1;
      emit ctx "}"
  | DCrunch (name, sig_, body) | DRake (name, sig_, body) ->
      let ret_t = llvm_type (Check.check_type check_env sig_.sig_return) in
      let params_str =
        sig_.sig_params
        |> List.map (fun (pname, ptyp) ->
            let t = Check.check_type check_env ptyp in
            let lt = llvm_type t in
            sprintf "%s %%%s" lt pname)
        |> String.concat ", "
      in

      emit ctx "";
      emit ctx "define %s @%s(%s) {" ret_t name params_str;
      emit ctx "entry:";
      ctx.indent <- 1;

      (* Add params to local env *)
      let local_env = Hashtbl.copy vars_env in
      List.iter
        (fun (pname, ptyp) ->
          let t = Check.check_type check_env ptyp in
          Hashtbl.add local_env pname t)
        sig_.sig_params;

      let result, _ = emit_expr ctx local_env body in
      emit ctx "ret %s %s" ret_t result;
      ctx.indent <- 0;
      emit ctx "}"
  | DRun (name, sig_, _body) ->
      emit ctx ";; Run: %s (not yet implemented)" name;
      ignore sig_
  | _ -> emit ctx ";; unimplemented definition"

let emit_program (check_env : Check.env) (p : program) : string =
  let ctx = { buf = Buffer.create 8192; counter = 0; indent = 0 } in

  (* Header *)
  emit ctx "; rake Language - Generated LLVM IR";
  emit ctx "; Target: AVX2 (%d-wide vectors)" vector_width;
  emit ctx "";
  emit ctx "target triple = \"x86_64-unknown-linux-gnu\"";
  emit ctx "";

  (* Intrinsics *)
  emit ctx "; Intrinsics";
  emit ctx "declare <%d x float> @llvm.sqrt.v%df32(<%d x float>)" vector_width
    vector_width vector_width;
  emit ctx "declare <%d x float> @llvm.fabs.v%df32(<%d x float>)" vector_width
    vector_width vector_width;
  emit ctx "declare <%d x float> @llvm.sin.v%df32(<%d x float>)" vector_width
    vector_width vector_width;
  emit ctx "declare <%d x float> @llvm.cos.v%df32(<%d x float>)" vector_width
    vector_width vector_width;
  emit ctx "declare float @llvm.vector.reduce.fadd.v%df32(float, <%d x float>)"
    vector_width vector_width;
  emit ctx "";

  (* Definitions *)
  List.iter
    (fun m ->
      emit ctx "; Module: %s" m.mod_name;
      emit ctx "";
      List.iter (emit_def ctx check_env check_env.vars) m.mod_defs)
    p;

  Buffer.contents ctx.buf
