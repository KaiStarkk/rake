%{
  open Ast

  let make_loc startpos _endpos = {
    file = startpos.Lexing.pos_fname;
    line = startpos.Lexing.pos_lnum;
    col = startpos.Lexing.pos_cnum - startpos.Lexing.pos_bol;
    offset = startpos.Lexing.pos_cnum;
  }

  let mknode v sp ep = { v; loc = make_loc sp ep }
%}

(* Tokens *)
%token STACK AOS SINGLE TYPE PACK ARRAY RACK
%token CRUNCH RAKE RUN LET IN WITH RESULTS
%token SWEEP COMPACT
%token OTHERWISE RETIRE HALT
%token REPEAT TIMES UNTIL AS SPREAD ACROSS CORES SYNC
%token IS NOT AND OR
%token FLOAT DOUBLE INT INT8 INT16 INT64 UINT UINT8 UINT16 UINT64 BOOL
%token VEC2 VEC3 VEC4 MAT3 MAT4
%token TRUE FALSE
%token LANES LANE_INDEX LEAD TALLY ANY ALL NONE REDUCE GATHER SCATTER BROADCAST
%token SHUFFLE ROTATE SHIFT COMPRESS EXPAND SELECT

%token <string> IDENT
%token <string> SCALAR_IDENT
%token <int64> INT_LIT
%token <float> FLOAT_LIT
%token <string> STRING_LIT

%token ARROW LARROW RESULT_TO COLONEQ PIPE COMPOSE
%token PLUS MINUS STAR SLASH PERCENT
%token LT LE GT GE EQ EQEQ NE
%token BAR LPAREN RPAREN LBRACKET RBRACKET LBRACE RBRACE
%token COLON SEMI COMMA DOT
%token EOF

(* Precedence - lowest to highest *)
%right ARROW
%left PIPE
%right COMPOSE
%left OR
%left AND
%nonassoc NOT
%nonassoc IS LT LE GT GE EQEQ NE
%left PLUS MINUS
%left STAR SLASH PERCENT
%nonassoc UMINUS

%start <Ast.program> program

%%

(* Program structure *)
program:
  | m = module_def EOF { [m] }

module_def:
  | defs = list(definition)
    { { mod_name = "Main"; mod_imports = []; mod_defs = defs } }

(* Definitions *)
definition:
  (* stack Name = { fields } - Structure-of-Arrays, the default parallel data type *)
  (* Fields MUST have rack types: float rack, vec3 rack, etc. *)
  | STACK name = IDENT EQ LBRACE fields = stack_fields RBRACE
    { mknode (DStack (name, fields)) $startpos $endpos }

  (* aos Name = { fields } - Array-of-Structures for interop *)
  (* Fields can be any type for C interoperability *)
  | AOS name = IDENT EQ LBRACE fields = aos_fields RBRACE
    { mknode (DAos (name, fields)) $startpos $endpos }

  (* single Name = { fields } - all-scalar config struct *)
  (* Fields MUST be scalar types: float, vec3, etc. (no rack keyword) *)
  | SINGLE name = IDENT EQ LBRACE fields = single_fields RBRACE
    { mknode (DSingle (name, fields)) $startpos $endpos }

  (* type alias *)
  | TYPE name = IDENT EQ t = typ
    { mknode (DType (name, t)) $startpos $endpos }

  (* crunch name params = expr - single-rail pure function, ML-style *)
  | CRUNCH name = IDENT ps = list(param) EQ body = expr
    { mknode (DCrunch (name, ps, body)) $startpos $endpos }

  (* rake name params = expr results <| result_expr - multi-rail parallel function *)
  | RAKE name = IDENT ps = list(param) EQ body = expr
    { mknode (DRake (name, ps, body, None)) $startpos $endpos }

  | RAKE name = IDENT ps = list(param) EQ body = expr RESULTS RESULT_TO result = expr
    { mknode (DRake (name, ps, body, Some result)) $startpos $endpos }

  (* run name params = stmts results <| result_expr - sequential with iteration *)
  | RUN name = IDENT ps = list(param) EQ body = stmt_block RESULTS RESULT_TO result = expr
    { mknode (DRun (name, ps, body, result)) $startpos $endpos }

(* Stack fields - require rack types *)
stack_fields:
  | fs = separated_list(SEMI, stack_field) option(SEMI) { fs }

stack_field:
  | name = IDENT COLON t = rack_typ
    { { field_name = name; field_type = t } }

(* AoS fields - any type for interop *)
aos_fields:
  | fs = separated_list(SEMI, aos_field) option(SEMI) { fs }

aos_field:
  | name = IDENT COLON t = typ
    { { field_name = name; field_type = t } }

(* Single fields - scalar types only *)
single_fields:
  | fs = separated_list(SEMI, single_field) option(SEMI) { fs }

single_field:
  | name = IDENT COLON t = scalar_typ
    { { field_name = name; field_type = t } }

(* Types *)
typ:
  | t = typ_atom { t }
  | t = typ_atom ARROW u = typ
    { mknode (TFun ([t], u)) $startpos $endpos }

(* Rack types - for stack fields (require 'rack' keyword) *)
rack_typ:
  | p = prim RACK
    { mknode (TRack p) $startpos $endpos }
  | c = compound RACK
    { mknode (TCompoundRack c) $startpos $endpos }

(* Scalar types - for single fields (no 'rack' keyword) *)
scalar_typ:
  | p = prim
    { mknode (TScalar p) $startpos $endpos }
  | c = compound
    { mknode (TCompoundScalar c) $startpos $endpos }

typ_atom:
  (* Primitive racks: float rack, int rack, etc. *)
  | p = prim RACK
    { mknode (TRack p) $startpos $endpos }
  (* Compound racks: vec3 rack, mat4 rack, etc. *)
  | c = compound RACK
    { mknode (TCompoundRack c) $startpos $endpos }
  (* Scalar primitives (used in single structs) *)
  | p = prim
    { mknode (TScalar p) $startpos $endpos }
  (* Scalar compounds (used in single structs) *)
  | c = compound
    { mknode (TCompoundScalar c) $startpos $endpos }
  (* Pack: collection of stack chunks - Particles pack *)
  | name = IDENT PACK
    { mknode (TPack (mknode (TStack name) $startpos $endpos, None)) $startpos $endpos }
  | name = IDENT PACK LBRACKET n = INT_LIT RBRACKET
    { mknode (TPack (mknode (TStack name) $startpos $endpos, Some (Int64.to_int n))) $startpos $endpos }
  (* Array: array of aos structs - Particle array *)
  | name = IDENT ARRAY
    { mknode (TArray (mknode (TAos name) $startpos $endpos, None)) $startpos $endpos }
  | name = IDENT ARRAY LBRACKET n = INT_LIT RBRACKET
    { mknode (TArray (mknode (TAos name) $startpos $endpos, Some (Int64.to_int n))) $startpos $endpos }
  (* Named type reference *)
  | name = IDENT
    { mknode (TStack name) $startpos $endpos }
  | LPAREN t = typ RPAREN
    { t }
  | LPAREN RPAREN
    { mknode TUnit $startpos $endpos }

prim:
  | FLOAT { Float } | DOUBLE { Double }
  | INT { Int } | INT8 { Int8 } | INT16 { Int16 } | INT64 { Int64 }
  | UINT { Uint } | UINT8 { Uint8 } | UINT16 { Uint16 } | UINT64 { Uint64 }
  | BOOL { Bool }

compound:
  | VEC2 { Vec2 } | VEC3 { Vec3 } | VEC4 { Vec4 }
  | MAT3 { Mat3 } | MAT4 { Mat4 }

(* Expressions *)
expr:
  | e = expr_pipe { e }
  | LET b = binding IN body = expr
    { mknode (ELet (b, body)) $startpos $endpos }
  | LET b = scalar_binding IN body = expr
    { mknode (ELetScalar (b, body)) $startpos $endpos }
  | rails = nonempty_list(rail)
    { mknode (ERails rails) $startpos $endpos }

expr_pipe:
  | e = expr_or { e }
  | e1 = expr_pipe PIPE e2 = expr_or
    { mknode (EPipe (e1, e2)) $startpos $endpos }

expr_or:
  | e = expr_and { e }
  | e1 = expr_or OR e2 = expr_and
    { mknode (EBinop (e1, Or, e2)) $startpos $endpos }

expr_and:
  | e = expr_not { e }
  | e1 = expr_and AND e2 = expr_not
    { mknode (EBinop (e1, And, e2)) $startpos $endpos }

expr_not:
  | e = expr_cmp { e }
  | NOT e = expr_not
    { mknode (EUnop (Not, e)) $startpos $endpos }

expr_cmp:
  | e = expr_add { e }
  | e1 = expr_add LT e2 = expr_add
    { mknode (EBinop (e1, Lt, e2)) $startpos $endpos }
  | e1 = expr_add LE e2 = expr_add
    { mknode (EBinop (e1, Le, e2)) $startpos $endpos }
  | e1 = expr_add GT e2 = expr_add
    { mknode (EBinop (e1, Gt, e2)) $startpos $endpos }
  | e1 = expr_add GE e2 = expr_add
    { mknode (EBinop (e1, Ge, e2)) $startpos $endpos }
  | e1 = expr_add EQEQ e2 = expr_add
    { mknode (EBinop (e1, Eq, e2)) $startpos $endpos }
  | e1 = expr_add NE e2 = expr_add
    { mknode (EBinop (e1, Ne, e2)) $startpos $endpos }

expr_add:
  | e = expr_mul { e }
  | e1 = expr_add PLUS e2 = expr_mul
    { mknode (EBinop (e1, Add, e2)) $startpos $endpos }
  | e1 = expr_add MINUS e2 = expr_mul
    { mknode (EBinop (e1, Sub, e2)) $startpos $endpos }

expr_mul:
  | e = expr_unary { e }
  | e1 = expr_mul STAR e2 = expr_unary
    { mknode (EBinop (e1, Mul, e2)) $startpos $endpos }
  | e1 = expr_mul SLASH e2 = expr_unary
    { mknode (EBinop (e1, Div, e2)) $startpos $endpos }
  | e1 = expr_mul PERCENT e2 = expr_unary
    { mknode (EBinop (e1, Mod, e2)) $startpos $endpos }

expr_unary:
  | e = expr_postfix { e }
  | MINUS e = expr_unary %prec UMINUS
    { mknode (EUnop (Neg, e)) $startpos $endpos }

expr_postfix:
  | e = expr_atom { e }
  | e = expr_postfix DOT f = IDENT
    { mknode (EField (e, f)) $startpos $endpos }
  | e = expr_postfix LBRACKET i = expr RBRACKET
    { mknode (EIndex (e, i)) $startpos $endpos }
  | f = IDENT LPAREN args = separated_list(COMMA, expr) RPAREN
    { mknode (ECall (f, args)) $startpos $endpos }

expr_atom:
  | n = INT_LIT
    { mknode (EInt n) $startpos $endpos }
  | f = FLOAT_LIT
    { mknode (EFloat f) $startpos $endpos }
  | TRUE
    { mknode (EBool true) $startpos $endpos }
  | FALSE
    { mknode (EBool false) $startpos $endpos }
  | s = STRING_LIT
    { mknode (EString s) $startpos $endpos }
  | name = IDENT
    { mknode (EVar name) $startpos $endpos }
  | name = SCALAR_IDENT
    { mknode (EScalarVar name) $startpos $endpos }
  | LPAREN RPAREN
    { mknode (ETuple []) $startpos $endpos }
  | LPAREN e = expr RPAREN
    { e }
  | LPAREN e = expr COMMA es = separated_nonempty_list(COMMA, expr) RPAREN
    { mknode (ETuple (e :: es)) $startpos $endpos }
  | LBRACE fs = record_fields RBRACE
    { mknode (ERecord fs) $startpos $endpos }
  | LBRACE e = expr WITH fs = record_fields RBRACE
    { mknode (EWith (e, fs)) $startpos $endpos }
  (* Control flow expressions *)
  | RETIRE
    { mknode ERetire $startpos $endpos }
  | HALT
    { mknode EHalt $startpos $endpos }

  (* Built-in lane queries *)
  | LANES LPAREN RPAREN
    { mknode ELanes $startpos $endpos }
  | LANE_INDEX LPAREN RPAREN
    { mknode ELaneIndex $startpos $endpos }
  | LEAD LPAREN e = expr RPAREN
    { mknode (ELead e) $startpos $endpos }
  | TALLY LPAREN e = expr RPAREN
    { mknode (ETally e) $startpos $endpos }
  | ANY LPAREN e = expr RPAREN
    { mknode (EAny e) $startpos $endpos }
  | ALL LPAREN e = expr RPAREN
    { mknode (EAll e) $startpos $endpos }
  | NONE LPAREN e = expr RPAREN
    { mknode (ENone e) $startpos $endpos }

  (* Reductions *)
  | REDUCE LPAREN op = reduce_op COMMA e = expr RPAREN
    { mknode (EReduce (op, e)) $startpos $endpos }

  (* Memory operations *)
  | GATHER LPAREN base = expr COMMA idx = expr RPAREN
    { mknode (EGather (base, idx)) $startpos $endpos }
  | SCATTER LPAREN base = expr COMMA idx = expr COMMA val_ = expr RPAREN
    { mknode (EScatter (base, idx, val_)) $startpos $endpos }

  (* Lane manipulation *)
  | BROADCAST LPAREN e = expr RPAREN
    { mknode (EBroadcast e) $startpos $endpos }
  | SHUFFLE LPAREN src = expr COMMA idx = expr RPAREN
    { mknode (EShuffle (src, idx)) $startpos $endpos }
  | ROTATE LPAREN e = expr COMMA n = expr RPAREN
    { mknode (ERotate (e, n)) $startpos $endpos }
  | SHIFT LPAREN e = expr COMMA n = expr RPAREN
    { mknode (EShift (e, n)) $startpos $endpos }
  | SELECT LPAREN mask = expr COMMA a = expr COMMA b = expr RPAREN
    { mknode (ESelect (mask, a, b)) $startpos $endpos }

  (* Compression and expansion *)
  | COMPRESS LPAREN e = expr COMMA mask = expr RPAREN
    { mknode (ECompress (e, mask)) $startpos $endpos }
  | EXPAND LPAREN e = expr COMMA mask = expr RPAREN
    { mknode (EExpand (e, mask)) $startpos $endpos }

reduce_op:
  | PLUS { RAdd }
  | STAR { RMul }
  | IDENT { match $1 with "min" -> RMin | "max" -> RMax | _ -> failwith "Unknown reduce op" }
  | AND { RAnd }
  | OR { ROr }

record_fields:
  | fs = separated_list(SEMI, record_field) option(SEMI) { fs }

record_field:
  | name = IDENT EQ e = expr { (name, e) }

params:
  | ps = nonempty_list(param) { ps }

param:
  | name = IDENT
    { PVar (name, None) }
  | LPAREN name = IDENT COLON t = typ RPAREN
    { PVar (name, Some t) }
  | name = SCALAR_IDENT
    { PScalar (name, None) }
  | LPAREN name = SCALAR_IDENT COLON t = typ RPAREN
    { PScalar (name, Some t) }

binding:
  | name = IDENT EQ e = expr
    { { bind_name = name; bind_type = None; bind_expr = e } }
  | name = IDENT COLON t = typ EQ e = expr
    { { bind_name = name; bind_type = Some t; bind_expr = e } }

scalar_binding:
  | name = SCALAR_IDENT EQ e = expr
    { { sbind_name = name; sbind_type = None; sbind_expr = e } }
  | name = SCALAR_IDENT COLON t = typ EQ e = expr
    { { sbind_name = name; sbind_type = Some t; sbind_expr = e } }

(* Rails *)
rail:
  | BAR cond = rail_cond ARROW body = rail_body
    { mknode { rail_cond = cond; rail_body = body } $startpos $endpos }

rail_cond:
  | name = IDENT COLONEQ p = predicate
    { RCNamed (name, p) }
  | p = predicate
    { RCAnon p }
  | OTHERWISE
    { RCOtherwise }

rail_body:
  | e = expr { e }
  | rails = nonempty_list(rail)
    { mknode (ERails rails) $startpos $endpos }

predicate:
  | p = pred_or { p }

pred_or:
  | p = pred_and { p }
  | p1 = pred_or OR p2 = pred_and
    { mknode (POr (p1, p2)) $startpos $endpos }

pred_and:
  | p = pred_not { p }
  | p1 = pred_and AND p2 = pred_not
    { mknode (PAnd (p1, p2)) $startpos $endpos }

pred_not:
  | p = pred_cmp { p }
  | NOT p = pred_not
    { mknode (PNot p) $startpos $endpos }

pred_cmp:
  | e = expr_add
    { mknode (PExpr e) $startpos $endpos }
  | e1 = expr_add IS e2 = expr_add
    { mknode (PIs (e1, e2)) $startpos $endpos }
  | e1 = expr_add IS NOT e2 = expr_add
    { mknode (PIsNot (e1, e2)) $startpos $endpos }
  | e1 = expr_add LT e2 = expr_add
    { mknode (PCmp (e1, CLt, e2)) $startpos $endpos }
  | e1 = expr_add LE e2 = expr_add
    { mknode (PCmp (e1, CLe, e2)) $startpos $endpos }
  | e1 = expr_add GT e2 = expr_add
    { mknode (PCmp (e1, CGt, e2)) $startpos $endpos }
  | e1 = expr_add GE e2 = expr_add
    { mknode (PCmp (e1, CGe, e2)) $startpos $endpos }
  | LPAREN p = predicate RPAREN
    { p }

(* Statements *)
stmt_block:
  | stmts = nonempty_list(stmt) { stmts }

stmt:
  | LET b = binding
    { mknode (SLet b) $startpos $endpos }
  | LET b = scalar_binding
    { mknode (SLetScalar b) $startpos $endpos }
  | lhs = expr_postfix LARROW rhs = expr
    { mknode (SAssign (lhs, rhs)) $startpos $endpos }
  | SWEEP src = IDENT ARROW dst = IDENT COLON body = stmt_block
    { mknode (SSweep (src, dst, body)) $startpos $endpos }
  (* compact pack - remove retired lanes, expensive cross-lane operation *)
  | COMPACT src = IDENT
    { mknode (SCompact src) $startpos $endpos }
  | SPREAD src = IDENT ACROSS CORES ARROW chunk = IDENT COLON body = stmt_block
    { mknode (SSpread (src, chunk, body)) $startpos $endpos }
  | REPEAT n = expr TIMES COLON body = stmt_block
    { mknode (SRepeat (n, None, body)) $startpos $endpos }
  | REPEAT n = expr TIMES AS i = SCALAR_IDENT COLON body = stmt_block
    { mknode (SRepeat (n, Some i, body)) $startpos $endpos }
  | REPEAT UNTIL cond = SCALAR_IDENT COLON body = stmt_block
    { mknode (SRepeatUntil (cond, body)) $startpos $endpos }
  | SYNC
    { mknode SSync $startpos $endpos }
  | HALT
    { mknode SHalt $startpos $endpos }
  | e = expr
    { mknode (SExpr e) $startpos $endpos }

%%
