(** Rake 0.2.0 Compiler

    Command-line interface for the rake compiler.
*)

let usage = {|
rake - vector-first language for CPU SIMD

Usage:
  rake <file.rk>                Parse and type-check
  rake --emit-tokens <file.rk>  Emit tokens (for debugging)
  rake --emit-ast <file.rk>     Emit AST (for debugging)
  rake --emit-mlir <file.rk>    Emit MLIR
  rake --version                Show version
  rake --help                   Show this help

Options:
  --width <n>    Vector width (default: 8 for AVX2)
  --output <f>   Output file (default: stdout)
|}

let version = "rake 0.2.0"

let parse_file filename =
  let ic = open_in filename in
  let lexbuf = Lexing.from_channel ic in
  lexbuf.Lexing.lex_curr_p <- {
    lexbuf.Lexing.lex_curr_p with
    Lexing.pos_fname = filename;
  };
  try
    let program = Rake.Parser.program Rake.Lexer.token lexbuf in
    close_in ic;
    Ok program
  with
  | Rake.Lexer.LexError (msg, pos) ->
      close_in ic;
      Error (Printf.sprintf "%s:%d:%d: Lexical error: %s"
        pos.Lexing.pos_fname
        pos.Lexing.pos_lnum
        (pos.Lexing.pos_cnum - pos.Lexing.pos_bol)
        msg)
  | Rake.Parser.Error ->
      close_in ic;
      let pos = lexbuf.Lexing.lex_curr_p in
      Error (Printf.sprintf "%s:%d:%d: Syntax error"
        pos.Lexing.pos_fname
        pos.Lexing.pos_lnum
        (pos.Lexing.pos_cnum - pos.Lexing.pos_bol))

let emit_tokens filename =
  let ic = open_in filename in
  let lexbuf = Lexing.from_channel ic in
  lexbuf.Lexing.lex_curr_p <- { lexbuf.Lexing.lex_curr_p with Lexing.pos_fname = filename };
  let result = Rake.Lexer.emit lexbuf in
  close_in ic;
  result

let emit_ast program =
  Rake.Ast.show_program program

let emit_mlir env program =
  Rake.Mlir.emit env program

let () =
  let args = Array.to_list Sys.argv |> List.tl in

  let process_args = function
    | [] ->
        print_endline usage

    | ["--version"] ->
        print_endline version

    | ["--help"] | ["-h"] ->
        print_endline usage

    | ["--emit-tokens"; filename] ->
        print_string (emit_tokens filename)

    | ["--emit-ast"; filename] -> (
        match parse_file filename with
        | Ok program ->
            print_endline (emit_ast program)
        | Error msg ->
            prerr_endline msg;
            exit 1)

    | ["--emit-mlir"; filename] -> (
        match parse_file filename with
        | Ok program -> (
            match Rake.Typecheck.check program with
            | Ok env ->
                print_endline (emit_mlir env program)
            | Error msg ->
                prerr_endline msg;
                exit 1)
        | Error msg ->
            prerr_endline msg;
            exit 1)

    | [filename] when String.length filename > 3 &&
                      String.sub filename (String.length filename - 3) 3 = ".rk" -> (
        match parse_file filename with
        | Ok program -> (
            match Rake.Typecheck.check program with
            | Ok _env ->
                Printf.printf "Parsed and type-checked %s successfully.\n" filename
            | Error msg ->
                prerr_endline msg;
                exit 1)
        | Error msg ->
            prerr_endline msg;
            exit 1)

    | unknown :: _ ->
        Printf.eprintf "Unknown option or file: %s\n" unknown;
        prerr_endline usage;
        exit 1
  in
  process_args args
