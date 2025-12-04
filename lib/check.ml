(* Type checking *)

open Ast

type error =
  | TypeError of string * loc
  | UnboundVar of string * loc
  | RailError of string * loc
[@@deriving show]

type env = {
  vars : (ident, Types.t) Hashtbl.t;
  scalars : (scalar_ident, Types.t) Hashtbl.t;
  types : (ident, Types.t) Hashtbl.t;
  rails : (ident, Types.t) Hashtbl.t;
}

let make_env () =
  {
    vars = Hashtbl.create 32;
    scalars = Hashtbl.create 32;
    types = Hashtbl.create 32;
    rails = Hashtbl.create 32;
  }

let clone_env env =
  {
    vars = Hashtbl.copy env.vars;
    scalars = Hashtbl.copy env.scalars;
    types = Hashtbl.copy env.types;
    rails = Hashtbl.copy env.rails;
  }

exception Check_error of error

let error msg loc = raise (Check_error (TypeError (msg, loc)))
let unbound name loc = raise (Check_error (UnboundVar (name, loc)))
let rail_error msg loc = raise (Check_error (RailError (msg, loc)))

let rec check_type env (t : typ) : Types.t =
  match t.v with
  | TRack p -> Types.Rack (Types.of_prim p)
  | TCompoundRack c -> Types.CompoundRack (Types.of_compound c)
  | TScalar p -> Types.Scalar (Types.of_prim p)
  | TCompoundScalar c -> Types.CompoundScalar (Types.of_compound c)
  | TStack (inner, n) -> Types.Stack (check_type env inner, n)
  | TArray (inner, n) -> Types.Array (check_type env inner, n)
  | TSoa name -> (
      match Hashtbl.find_opt env.types name with
      | Some t -> t
      | None -> error (Printf.sprintf "Unknown soa type: %s" name) t.loc)
  | TAos name -> (
      match Hashtbl.find_opt env.types name with
      | Some t -> t
      | None -> error (Printf.sprintf "Unknown aos struct: %s" name) t.loc)
  | TSingle name -> (
      match Hashtbl.find_opt env.types name with
      | Some t -> t
      | None -> error (Printf.sprintf "Unknown single: %s" name) t.loc)
  | TMask -> Types.Mask
  | TFun (args, ret) ->
      Types.Fun (List.map (check_type env) args, check_type env ret)
  | TUnit -> Types.Unit

let rec check_expr env (e : expr) : Types.t =
  match e.v with
  | EInt _ -> Types.Rack Types.SInt (* Literals default to rack *)
  | EFloat _ -> Types.Rack Types.SFloat
  | EBool _ -> Types.Mask
  | EString _ -> error "Strings not yet supported in rack context" e.loc
  | EVar name -> (
      match Hashtbl.find_opt env.vars name with
      | Some t -> t
      | None -> unbound name e.loc)
  | EScalarVar name -> (
      match Hashtbl.find_opt env.scalars name with
      | Some t -> t
      | None -> unbound name e.loc)
  | EBinop (l, op, r) -> (
      let lt = check_expr env l in
      let rt = check_expr env r in
      match op with
      | Lt | Le | Gt | Ge | Eq | Ne -> Types.cmp_result lt rt
      | And | Or -> Types.Mask
      | _ -> Types.binop_result lt rt)
  | EUnop (_, e) -> check_expr env e
  | EField (e, f) -> (
      let t = check_expr env e in
      match t with
      | Types.Pack (_, fields) | Types.Aos (_, fields) | Types.Single (_, fields)
        -> (
          match List.assoc_opt f fields with
          | Some ft -> ft
          | None -> error (Printf.sprintf "Unknown field: %s" f) e.loc)
      | Types.CompoundRack c -> (
          match (c, f) with
          | (Types.CVec2 | Types.CVec3 | Types.CVec4), ("x" | "y" | "z" | "w")
            ->
              Types.Rack Types.SFloat
          | _ -> error (Printf.sprintf "Unknown field: %s" f) e.loc)
      | Types.CompoundScalar c -> (
          match (c, f) with
          | (Types.CVec2 | Types.CVec3 | Types.CVec4), ("x" | "y" | "z" | "w")
            ->
              Types.Scalar Types.SFloat
          | _ -> error (Printf.sprintf "Unknown field: %s" f) e.loc)
      | _ -> error "Field access on non-struct type" e.loc)
  | EIndex (base, _idx) -> (
      let bt = check_expr env base in
      match bt with
      | Types.Stack (inner, _) -> inner
      | Types.Array (inner, _) -> inner
      | _ -> error "Index on non-array type" e.loc)
  | ECall (_name, args) ->
      let _arg_types = List.map (check_expr env) args in
      (* TODO: Look up function and check args *)
      Types.Unknown
  | EPipe (e1, e2) ->
      let _t1 = check_expr env e1 in
      check_expr env e2 (* Result is type of second expr *)
  | ETuple es -> Types.Tuple (List.map (check_expr env) es)
  | ERecord fields ->
      (* TODO: Infer record type *)
      let field_types = List.map (fun (n, e) -> (n, check_expr env e)) fields in
      Types.Pack ("anonymous", field_types)
  | EWith (base, updates) ->
      let bt = check_expr env base in
      List.iter (fun (_, e) -> ignore (check_expr env e)) updates;
      bt
  | EFun (params, body) ->
      let env' = clone_env env in
      let param_types =
        List.map
          (fun p ->
            match p with
            | PVar (name, Some t) ->
                let ty = check_type env t in
                Hashtbl.add env'.vars name ty;
                ty
            | PVar (name, None) ->
                Hashtbl.add env'.vars name Types.Unknown;
                Types.Unknown
            | PScalar (name, Some t) ->
                let ty = check_type env t in
                Hashtbl.add env'.scalars name ty;
                ty
            | PScalar (name, None) ->
                Hashtbl.add env'.scalars name Types.Unknown;
                Types.Unknown)
          params
      in
      let ret_type = check_expr env' body in
      Types.Fun (param_types, ret_type)
  | ELet (b, body) ->
      let t = check_expr env b.bind_expr in
      let env' = clone_env env in
      Hashtbl.add env'.vars b.bind_name t;
      check_expr env' body
  | ELetScalar (b, body) ->
      let t = check_expr env b.sbind_expr in
      let env' = clone_env env in
      Hashtbl.add env'.scalars b.sbind_name t;
      check_expr env' body
  | EReduce (_, e) -> (
      let t = check_expr env e in
      match t with
      | Types.Rack s -> Types.Scalar s
      | _ -> Types.Scalar Types.SFloat)
  | ELanes -> Types.Scalar Types.SInt
  | ELaneIndex -> Types.Rack Types.SInt
  | ETally _ -> Types.Scalar Types.SInt
  | EAny _ | EAll _ | ENone _ -> Types.Scalar Types.SBool
  | EGather (_, _) -> Types.Rack Types.SFloat (* TODO: Proper type *)
  | EScatter (_, _, _) -> Types.Unit
  | EBroadcast e ->
      let t = check_expr env e in
      Types.broadcast t
  | ERails rails -> (
      (* Check each rail, all must produce compatible types *)
      let types = List.map (check_rail env) rails in
      match types with [] -> Types.Unit | t :: _ -> t (* TODO: Unify types *))
  | ERetire -> Types.Unit

and check_rail env (r : rail) : Types.t =
  let env' = clone_env env in
  (* Add named rail to environment *)
  (match r.v.rail_cond with
  | RCNamed (name, pred) ->
      let _pt = check_pred env pred in
      Hashtbl.add env'.rails name Types.Mask
  | RCAnon pred ->
      let _pt = check_pred env pred in
      ()
  | RCOtherwise -> ()
  | RCRef name ->
      if not (Hashtbl.mem env.rails name) then
        rail_error (Printf.sprintf "Unknown rail: %s" name) r.loc);
  check_expr env' r.v.rail_body

and check_pred env (p : predicate) : Types.t =
  match p.v with
  | PExpr e -> check_expr env e
  | PIs (l, r) | PIsNot (l, r) ->
      ignore (check_expr env l);
      ignore (check_expr env r);
      Types.Mask
  | PCmp (l, _, r) ->
      ignore (check_expr env l);
      ignore (check_expr env r);
      Types.Mask
  | PAnd (l, r) | POr (l, r) ->
      ignore (check_pred env l);
      ignore (check_pred env r);
      Types.Mask
  | PNot p ->
      ignore (check_pred env p);
      Types.Mask

(* Extract parameter names from a function expression *)
let extract_params (e : expr) : param list =
  match e.v with EFun (params, _) -> params | _ -> []

(* Bind function parameters from body to types from signature *)
let bind_params env' body_params sig_types =
  let types = List.map snd sig_types in
  (* Use combine_safe to handle length mismatches *)
  let rec bind_pairs params typs =
    match (params, typs) with
    | [], _ | _, [] -> ()
    | param :: rest_params, typ :: rest_typs ->
        let t =
          match typ.v with
          | TRack p -> Types.Rack (Types.of_prim p)
          | TCompoundRack c -> Types.CompoundRack (Types.of_compound c)
          | TScalar p -> Types.Scalar (Types.of_prim p)
          | TCompoundScalar c -> Types.CompoundScalar (Types.of_compound c)
          | TSoa name -> (
              match Hashtbl.find_opt env'.types name with
              | Some t -> t
              | None -> Types.Unknown)
          | TAos name -> (
              match Hashtbl.find_opt env'.types name with
              | Some t -> t
              | None -> Types.Unknown)
          | TSingle name -> (
              match Hashtbl.find_opt env'.types name with
              | Some t -> t
              | None -> Types.Unknown)
          | TMask -> Types.Mask
          | TUnit -> Types.Unit
          | _ -> Types.Unknown
        in
        (match param with
        | PVar (pname, _) -> Hashtbl.add env'.vars pname t
        | PScalar (pname, _) -> Hashtbl.add env'.scalars pname t);
        bind_pairs rest_params rest_typs
  in
  bind_pairs body_params types

let rec check_def env (d : def) : unit =
  match d.v with
  | DSoa (name, fields) ->
      let field_types =
        List.map
          (fun f ->
            let t = check_type env f.field_type in
            (* Pack fields should be racks *)
            let t =
              match t with
              | Types.Scalar s -> Types.Rack s
              | Types.CompoundScalar c -> Types.CompoundRack c
              | t -> t
            in
            (f.field_name, t))
          fields
      in
      Hashtbl.add env.types name (Types.Pack (name, field_types))
  | DAos (name, fields) ->
      let field_types =
        List.map
          (fun f ->
            let t = check_type env f.field_type in
            (* AoS fields should be scalars *)
            let t =
              match t with
              | Types.Rack s -> Types.Scalar s
              | Types.CompoundRack c -> Types.CompoundScalar c
              | t -> t
            in
            (f.field_name, t))
          fields
      in
      Hashtbl.add env.types name (Types.Aos (name, field_types))
  | DSingle (name, fields) ->
      let field_types =
        List.map
          (fun f ->
            let t = check_type env f.field_type in
            (f.field_name, t))
          fields
      in
      Hashtbl.add env.types name (Types.Single (name, field_types))
  | DType (name, t) ->
      let ty = check_type env t in
      Hashtbl.add env.types name ty
  | DCrunch (name, sig_, body) ->
      let env' = clone_env env in
      (* Extract actual parameter names from function body *)
      let body_params = extract_params body in
      if List.length body_params > 0 then
        bind_params env' body_params sig_.sig_params
      else
        (* Fallback: use signature names *)
        List.iter
          (fun (pname, ptyp) ->
            let t = check_type env ptyp in
            Hashtbl.add env'.vars pname t)
          sig_.sig_params;
      let _body_type = check_expr env' body in
      let ret_type = check_type env sig_.sig_return in
      let param_types =
        List.map (fun (_, t) -> check_type env t) sig_.sig_params
      in
      Hashtbl.add env.vars name (Types.Fun (param_types, ret_type))
  | DRake (name, sig_, body) ->
      let env' = clone_env env in
      let body_params = extract_params body in
      if List.length body_params > 0 then
        bind_params env' body_params sig_.sig_params
      else
        List.iter
          (fun (pname, ptyp) ->
            let t = check_type env ptyp in
            Hashtbl.add env'.vars pname t)
          sig_.sig_params;
      let _body_type = check_expr env' body in
      let ret_type = check_type env sig_.sig_return in
      let param_types =
        List.map (fun (_, t) -> check_type env t) sig_.sig_params
      in
      Hashtbl.add env.vars name (Types.Fun (param_types, ret_type))
  | DRun (name, sig_, body) ->
      let env' = clone_env env in
      (* For run blocks, params come from signature directly *)
      List.iter
        (fun (pname, ptyp) ->
          let t = check_type env ptyp in
          Hashtbl.add env'.vars pname t)
        sig_.sig_params;
      List.iter (check_stmt env') body;
      let ret_type = check_type env sig_.sig_return in
      let param_types =
        List.map (fun (_, t) -> check_type env t) sig_.sig_params
      in
      Hashtbl.add env.vars name (Types.Fun (param_types, ret_type))

and check_stmt env (s : stmt) : unit =
  match s.v with
  | SLet b ->
      let t = check_expr env b.bind_expr in
      Hashtbl.add env.vars b.bind_name t
  | SLetScalar b ->
      let t = check_expr env b.sbind_expr in
      Hashtbl.add env.scalars b.sbind_name t
  | SAssign (lhs, rhs) ->
      ignore (check_expr env lhs);
      ignore (check_expr env rhs)
  | SExpr e -> ignore (check_expr env e)
  | SSweep (src, dst, body) ->
      let st =
        match Hashtbl.find_opt env.vars src with
        | Some t -> t
        | None -> unbound src s.loc
      in
      let elem_type =
        match st with Types.Stack (inner, _) -> inner | t -> t
      in
      let env' = clone_env env in
      Hashtbl.add env'.vars dst elem_type;
      List.iter (check_stmt env') body
  | SPack (src, dst) ->
      (* pack compacts active lanes from src stack into dst stack *)
      let st =
        match Hashtbl.find_opt env.vars src with
        | Some t -> t
        | None -> unbound src s.loc
      in
      (* dst should have same stack type as src *)
      (match Hashtbl.find_opt env.vars dst with
      | Some _ -> ()
      | None -> unbound dst s.loc);
      ignore st
  | SSpread (src, chunk, body) ->
      let st =
        match Hashtbl.find_opt env.vars src with
        | Some t -> t
        | None -> unbound src s.loc
      in
      let env' = clone_env env in
      Hashtbl.add env'.vars chunk st;
      List.iter (check_stmt env') body
  | SRepeat (_, var, body) ->
      let env' = clone_env env in
      (match var with
      | Some name -> Hashtbl.add env'.scalars name (Types.Scalar Types.SInt)
      | None -> ());
      List.iter (check_stmt env') body
  | SRepeatUntil (_, body) -> List.iter (check_stmt env) body
  | SSync -> ()

let check_program (p : program) : (env, error) result =
  let env = make_env () in
  try
    List.iter (fun m -> List.iter (check_def env) m.mod_defs) p;
    Ok env
  with Check_error e -> Error e
