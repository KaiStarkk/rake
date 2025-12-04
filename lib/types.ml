(* Type representation for checking *)

type lane_count = int (* Target-dependent, e.g., 8 for AVX2 *)

type scalar =
  | SFloat
  | SDouble
  | SInt
  | SInt8
  | SInt16
  | SInt64
  | SUint
  | SUint8
  | SUint16
  | SUint64
  | SBool
[@@deriving show, eq]

type compound = CVec2 | CVec3 | CVec4 | CMat3 | CMat4 [@@deriving show, eq]

type t =
  | Rack of scalar
  | CompoundRack of compound
  | Scalar of scalar
  | CompoundScalar of compound
  | Stack of t * int option
  | Array of t * int option
  | Pack of string * (string * t) list
  | Aos of string * (string * t) list
  | Single of string * (string * t) list
  | Mask
  | Fun of t list * t
  | Tuple of t list
  | Unit
  | Unknown
[@@deriving show, eq]

let of_prim (p : Ast.prim) : scalar =
  match p with
  | Ast.Float -> SFloat
  | Ast.Double -> SDouble
  | Ast.Int -> SInt
  | Ast.Int8 -> SInt8
  | Ast.Int16 -> SInt16
  | Ast.Int64 -> SInt64
  | Ast.Uint -> SUint
  | Ast.Uint8 -> SUint8
  | Ast.Uint16 -> SUint16
  | Ast.Uint64 -> SUint64
  | Ast.Bool -> SBool

let of_compound (c : Ast.compound) : compound =
  match c with
  | Ast.Vec2 -> CVec2
  | Ast.Vec3 -> CVec3
  | Ast.Vec4 -> CVec4
  | Ast.Mat3 -> CMat3
  | Ast.Mat4 -> CMat4

let is_rack = function
  | Rack _ | CompoundRack _ | Mask -> true
  | Pack _ -> true (* Packs contain racks *)
  | _ -> false

let is_scalar = function
  | Scalar _ | CompoundScalar _ -> true
  | Single _ -> true
  | _ -> false

let is_numeric = function
  | Rack s | Scalar s -> ( match s with SBool -> false | _ -> true)
  | CompoundRack _ | CompoundScalar _ -> true
  | _ -> false

(* Type of a broadcast: scalar -> rack *)
let broadcast = function
  | Scalar s -> Rack s
  | CompoundScalar c -> CompoundRack c
  | t -> t

(* Result of binary operation *)
let binop_result t1 t2 =
  match (t1, t2) with
  | Rack _, _ -> t1
  | _, Rack _ -> t2
  | CompoundRack _, _ -> t1
  | _, CompoundRack _ -> t2
  | _ -> t1

(* Result of comparison *)
let cmp_result t1 t2 =
  match (t1, t2) with
  | Rack _, _ | _, Rack _ -> Mask
  | CompoundRack _, _ | _, CompoundRack _ -> Mask
  | _ -> Scalar SBool

let rec pp fmt = function
  | Rack s -> Format.fprintf fmt "%s rack" (pp_scalar s)
  | CompoundRack c -> Format.fprintf fmt "%s rack" (pp_compound c)
  | Scalar s -> Format.fprintf fmt "%s" (pp_scalar s)
  | CompoundScalar c -> Format.fprintf fmt "%s" (pp_compound c)
  | Stack (inner, None) -> Format.fprintf fmt "%a stack" pp inner
  | Stack (inner, Some n) -> Format.fprintf fmt "%a stack[%d]" pp inner n
  | Array (inner, None) -> Format.fprintf fmt "%a array" pp inner
  | Array (inner, Some n) -> Format.fprintf fmt "%a array[%d]" pp inner n
  | Pack (name, _) -> Format.fprintf fmt "%s" name
  | Aos (name, _) -> Format.fprintf fmt "%s" name
  | Single (name, _) -> Format.fprintf fmt "%s" name
  | Mask -> Format.fprintf fmt "mask"
  | Fun (args, ret) ->
      Format.fprintf fmt "(%a) -> %a"
        (Format.pp_print_list
           ~pp_sep:(fun fmt () -> Format.fprintf fmt ", ")
           pp)
        args pp ret
  | Tuple ts ->
      Format.fprintf fmt "(%a)"
        (Format.pp_print_list
           ~pp_sep:(fun fmt () -> Format.fprintf fmt " * ")
           pp)
        ts
  | Unit -> Format.fprintf fmt "()"
  | Unknown -> Format.fprintf fmt "?"

and pp_scalar = function
  | SFloat -> "float"
  | SDouble -> "double"
  | SInt -> "int"
  | SInt8 -> "int8"
  | SInt16 -> "int16"
  | SInt64 -> "int64"
  | SUint -> "uint"
  | SUint8 -> "uint8"
  | SUint16 -> "uint16"
  | SUint64 -> "uint64"
  | SBool -> "bool"

and pp_compound = function
  | CVec2 -> "vec2"
  | CVec3 -> "vec3"
  | CVec4 -> "vec4"
  | CMat3 -> "mat3"
  | CMat4 -> "mat4"
