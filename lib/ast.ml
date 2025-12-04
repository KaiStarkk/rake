(* Rake Abstract Syntax Tree *)

(* Source locations *)
type loc = { file : string; line : int; col : int; offset : int }
[@@deriving show, eq]

let dummy_loc = { file = ""; line = 0; col = 0; offset = 0 }

type 'a node = { v : 'a; loc : loc } [@@deriving show, eq]

let node v loc = { v; loc }
let node_v n = n.v
let node_loc n = n.loc
let dummy v = { v; loc = dummy_loc }

(* Identifiers *)
type ident = string [@@deriving show, eq]

type scalar_ident = string
(* written as <name> in source *) [@@deriving show, eq]

(* Primitive types *)
type prim =
  | Float
  | Double
  | Int
  | Int8
  | Int16
  | Int64
  | Uint
  | Uint8
  | Uint16
  | Uint64
  | Bool
[@@deriving show, eq]

(* Compound vector types *)
type compound = Vec2 | Vec3 | Vec4 | Mat3 | Mat4 [@@deriving show, eq]

(* Types *)
type typ = typ_kind node

and typ_kind =
  | TRack of prim (* float rack *)
  | TCompoundRack of compound (* vec3 rack *)
  | TScalar of prim (* used with <> idents *)
  | TCompoundScalar of compound (* vec3 for single structs *)
  | TStack of ident (* Stack type name - SoA parallel data *)
  | TPack of typ * int option
    (* Particles pack or pack[1000] - collection of stack chunks *)
  | TArray of typ * int option (* Particle array or array[1000] *)
  | TAos of ident (* aos struct name *)
  | TSingle of ident (* single struct name *)
  | TMask (* bool rack / rail result *)
  | TFun of typ list * typ (* function type *)
  | TUnit
[@@deriving show, eq]

(* Binary operators *)
type binop =
  | Add
  | Sub
  | Mul
  | Div
  | Mod
  | Lt
  | Le
  | Gt
  | Ge
  | Eq
  | Ne
  | And
  | Or
  | Pipe (* |> *)
[@@deriving show, eq]

(* Unary operators *)
type unop = Neg | Not [@@deriving show, eq]

(* Reduce operations *)
type reduce_op = RAdd | RMul | RMin | RMax | RAnd | ROr [@@deriving show, eq]

(* Expressions *)
type expr = expr_kind node

and expr_kind =
  (* Literals *)
  | EInt of int64
  | EFloat of float
  | EBool of bool
  | EString of string
  (* Variables *)
  | EVar of ident
  | EScalarVar of scalar_ident
  (* Operators *)
  | EBinop of expr * binop * expr
  | EUnop of unop * expr
  (* Access *)
  | EField of expr * ident
  | EIndex of expr * expr
  (* Calls *)
  | ECall of ident * expr list
  | EPipe of expr * expr (* x |> f *)
  (* Constructors *)
  | ETuple of expr list
  | ERecord of (ident * expr) list
  | EWith of expr * (ident * expr) list (* { e with field = val } *)
  (* Bindings *)
  | ELet of binding * expr
  | ELetScalar of scalar_binding * expr
  (* Lane queries *)
  | ELanes (* lanes() *)
  | ELaneIndex (* lane_index() *)
  | ELead of expr (* lead(e) - first active lane's value *)
  | ETally of expr (* tally(mask) - count active lanes *)
  | EAny of expr (* any(mask) - true if any lane active *)
  | EAll of expr (* all(mask) - true if all lanes active *)
  | ENone of expr (* none(mask) - true if no lanes active *)
  (* Reductions *)
  | EReduce of reduce_op * expr
  (* Memory operations *)
  | EGather of expr * expr
  | EScatter of expr * expr * expr
  (* Lane manipulation *)
  | EBroadcast of expr (* broadcast(e) - scalar to all lanes *)
  | EShuffle of expr * expr (* shuffle(src, idx) - permute lanes *)
  | ERotate of expr * expr (* rotate(e, n) - rotate lanes by n *)
  | EShift of expr * expr (* shift(e, n) - shift lanes by n *)
  | ESelect of expr * expr * expr (* select(mask, a, b) - masked selection *)
  (* Compression and expansion *)
  | ECompress of expr * expr (* compress(e, mask) - compact active lanes *)
  | EExpand of expr * expr (* expand(e, mask) - inverse of compress *)
  (* Control flow (within expressions) *)
  | ERails of rail list
  | ERetire (* retire - deactivate current lane *)
  | EHalt (* halt - stop all lanes *)

and binding = { bind_name : ident; bind_type : typ option; bind_expr : expr }

and scalar_binding = {
  sbind_name : scalar_ident;
  sbind_type : typ option;
  sbind_expr : expr;
}

and param = PVar of ident * typ option | PScalar of scalar_ident * typ option

(* Rails - the core parallel branching construct *)
and rail = rail_kind node
and rail_kind = { rail_cond : rail_cond; rail_body : expr }

and rail_cond =
  | RCNamed of ident * predicate (* name := predicate *)
  | RCAnon of predicate (* just predicate *)
  | RCOtherwise (* otherwise *)
  | RCRef of ident (* reference to named rail *)

and predicate = pred_kind node

and pred_kind =
  | PExpr of expr
  | PIs of expr * expr
  | PIsNot of expr * expr
  | PCmp of expr * cmp_op * expr
  | PAnd of predicate * predicate
  | POr of predicate * predicate
  | PNot of predicate

and cmp_op = CLt | CLe | CGt | CGe [@@deriving show, eq]

(* Statements for imperative sections *)
type stmt = stmt_kind node

and stmt_kind =
  | SLet of binding
  | SLetScalar of scalar_binding
  | SAssign of expr * expr (* lhs <- rhs *)
  | SExpr of expr
  | SSweep of ident * ident * stmt list (* sweep source -> target: body *)
  | SCompact of ident (* compact pack - remove retired lanes *)
  | SSpread of
      ident * ident * stmt list (* spread source across cores -> chunk: body *)
  | SRepeat of expr * scalar_ident option * stmt list
  | SRepeatUntil of scalar_ident * stmt list
  | SSync (* sync - synchronization barrier *)
  | SHalt (* halt - stop all execution *)
[@@deriving show, eq]

(* Top-level definitions *)
type field = { field_name : ident; field_type : typ } [@@deriving show, eq]

type def = def_kind node

and def_kind =
  (* Stack: Structure-of-Arrays, the default parallel data type *)
  | DStack of ident * field list
  (* AoS: Array-of-Structures for interop *)
  | DAos of ident * field list
  (* Single: all-scalar config struct *)
  | DSingle of ident * field list
  (* Type alias *)
  | DType of ident * typ
  (* Crunch: single-rail pure function, ML-style params *)
  | DCrunch of ident * param list * expr
  (* Rake: multi-rail parallel function with optional result clause *)
  | DRake of ident * param list * expr * expr option
  (* Run: sequential composition with temporal iteration, requires result clause *)
  | DRun of ident * param list * stmt list * expr
[@@deriving show, eq]

(* Module *)
type modul = { mod_name : ident; mod_imports : ident list; mod_defs : def list }
[@@deriving show, eq]

(* Program *)
type program = modul list [@@deriving show, eq]
