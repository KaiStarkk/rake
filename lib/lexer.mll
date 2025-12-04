{
  open Parser
  open Lexing

  exception Error of string * position

  let error lexbuf msg = raise (Error (msg, lexbuf.lex_curr_p))

  let keywords = Hashtbl.create 64
  let () = List.iter (fun (k, v) -> Hashtbl.add keywords k v) [
    (* Data types - stack is now the definition keyword for SoA *)
    "stack", STACK;
    "aos", AOS;
    "single", SINGLE;
    "type", TYPE;
    "pack", PACK;       (* Type suffix: Particles pack *)
    "array", ARRAY;
    "rack", RACK;

    (* Functions *)
    "crunch", CRUNCH;
    "rake", RAKE;
    "run", RUN;
    "let", LET;
    "in", IN;
    "with", WITH;
    "results", RESULTS;

    (* Control - iteration and compaction *)
    "sweep", SWEEP;
    "compact", COMPACT;  (* Compaction operation: compact particles *)

    (* Rails *)
    "otherwise", OTHERWISE;
    "retire", RETIRE;
    "halt", HALT;

    (* Control *)
    "repeat", REPEAT;
    "times", TIMES;
    "until", UNTIL;
    "as", AS;
    "spread", SPREAD;
    "across", ACROSS;
    "cores", CORES;
    "sync", SYNC;

    (* Predicates *)
    "is", IS;
    "not", NOT;
    "and", AND;
    "or", OR;

    (* Primitives *)
    "float", FLOAT;
    "double", DOUBLE;
    "int", INT;
    "int8", INT8;
    "int16", INT16;
    "int64", INT64;
    "uint", UINT;
    "uint8", UINT8;
    "uint16", UINT16;
    "uint64", UINT64;
    "bool", BOOL;

    (* Compounds *)
    "vec2", VEC2;
    "vec3", VEC3;
    "vec4", VEC4;
    "mat3", MAT3;
    "mat4", MAT4;

    (* Booleans *)
    "true", TRUE;
    "false", FALSE;

    (* Built-in lane operations *)
    "lanes", LANES;
    "lane_index", LANE_INDEX;
    "lead", LEAD;
    "tally", TALLY;
    "any", ANY;
    "all", ALL;
    "none", NONE;
    "reduce", REDUCE;
    "gather", GATHER;
    "scatter", SCATTER;
    "broadcast", BROADCAST;
    "shuffle", SHUFFLE;
    "rotate", ROTATE;
    "shift", SHIFT;
    "compress", COMPRESS;
    "expand", EXPAND;
    "select", SELECT;
  ]

  let ident_or_keyword s =
    match Hashtbl.find_opt keywords s with
    | Some tok -> tok
    | None -> IDENT s

  let make_loc lexbuf = Ast.{
    file = lexbuf.lex_curr_p.pos_fname;
    line = lexbuf.lex_curr_p.pos_lnum;
    col = lexbuf.lex_curr_p.pos_cnum - lexbuf.lex_curr_p.pos_bol;
    offset = lexbuf.lex_curr_p.pos_cnum;
  }
}

let digit = ['0'-'9']
let hex = ['0'-'9' 'a'-'f' 'A'-'F']
let alpha = ['a'-'z' 'A'-'Z' '_']
let alnum = alpha | digit
let ident = alpha alnum*

let int_lit = digit+ | "0x" hex+
let float_lit = digit+ '.' digit* (['e' 'E'] ['+' '-']? digit+)?
              | digit* '.' digit+ (['e' 'E'] ['+' '-']? digit+)?
              | digit+ ['e' 'E'] ['+' '-']? digit+

let ws = [' ' '\t']+
let newline = '\r'? '\n'

rule token = parse
  | ws        { token lexbuf }
  | newline   { Lexing.new_line lexbuf; token lexbuf }
  
  (* Comments *)
  | "(*"      { comment 1 lexbuf }
  | "--"      { line_comment lexbuf }
  
  (* Multi-char operators and ligatures *)
  | "->"      { ARROW }
  | "<-"      { LARROW }
  | "<|"      { RESULT_TO }  (* results <| for result clause *)
  | ":="      { COLONEQ }
  | "|>"      { PIPE }
  | ">>"      { COMPOSE }
  | "<="      { LE }
  | ">="      { GE }
  | "<>"      { NE }
  | "!="      { NE }
  | "=="      { EQEQ }

  (* Scalar identifiers: <name> - must come before single '<' *)
  | '<' (ident as s) '>'  { SCALAR_IDENT s }
  
  (* Single-char operators *)
  | '+'       { PLUS }
  | '-'       { MINUS }
  | '*'       { STAR }
  | '/'       { SLASH }
  | '%'       { PERCENT }
  | '<'       { LT }
  | '>'       { GT }
  | '='       { EQ }
  | '|'       { BAR }
  | '('       { LPAREN }
  | ')'       { RPAREN }
  | '['       { LBRACKET }
  | ']'       { RBRACKET }
  | '{'       { LBRACE }
  | '}'       { RBRACE }
  | ':'       { COLON }
  | ';'       { SEMI }
  | ','       { COMMA }
  | '.'       { DOT }
  
  (* Literals *)
  | int_lit as n    { INT_LIT (Int64.of_string n) }
  | float_lit as f  { FLOAT_LIT (float_of_string f) }
  | '"'             { string (Buffer.create 32) lexbuf }
  
  (* Identifiers *)
  | ident as s      { ident_or_keyword s }
  
  | eof             { EOF }
  | _ as c          { error lexbuf (Printf.sprintf "Unexpected character: %c" c) }

and comment depth = parse
  | "*)"      { if depth = 1 then token lexbuf else comment (depth - 1) lexbuf }
  | "(*"      { comment (depth + 1) lexbuf }
  | newline   { Lexing.new_line lexbuf; comment depth lexbuf }
  | _         { comment depth lexbuf }
  | eof       { error lexbuf "Unterminated comment" }

and line_comment = parse
  | newline   { Lexing.new_line lexbuf; token lexbuf }
  | _         { line_comment lexbuf }
  | eof       { EOF }

and string buf = parse
  | '"'           { STRING_LIT (Buffer.contents buf) }
  | '\\' 'n'      { Buffer.add_char buf '\n'; string buf lexbuf }
  | '\\' 't'      { Buffer.add_char buf '\t'; string buf lexbuf }
  | '\\' '\\'     { Buffer.add_char buf '\\'; string buf lexbuf }
  | '\\' '"'      { Buffer.add_char buf '"'; string buf lexbuf }
  | [^ '\\' '"' '\n']+ as s { Buffer.add_string buf s; string buf lexbuf }
  | newline       { error lexbuf "Unterminated string" }
  | eof           { error lexbuf "Unterminated string" }

