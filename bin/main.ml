(* Rake compiler main entry point *)

let usage = "Usage: rakec [options] <source.rk>"
let input_file = ref None
let output_file = ref None
let _dump_tokens = ref false
let dump_ast = ref false
let dump_types = ref false
let target = ref "avx2"
let emit_llvm = ref false

let specs =
  Arg.
    [
      ("-o", String (fun s -> output_file := Some s), "Output file");
      ("--dump-ast", Set dump_ast, "Dump AST and exit");
      ("--dump-types", Set dump_types, "Dump types after checking");
      ("--target", Set_string target, "Target (avx2, avx512, neon)");
      ("--llvm", Set emit_llvm, "Emit LLVM IR instead of MLIR");
    ]

let anon_fun filename =
  match !input_file with
  | None -> input_file := Some filename
  | Some _ -> raise (Arg.Bad "Multiple input files not supported")

let read_file filename =
  let ic = open_in filename in
  let n = in_channel_length ic in
  let s = really_input_string ic n in
  close_in ic;
  s

let write_file filename content =
  let oc = open_out filename in
  output_string oc content;
  close_out oc

let () =
  Arg.parse specs anon_fun usage;

  match !input_file with
  | None ->
      prerr_endline "Error: No input file";
      prerr_endline usage;
      exit 1
  | Some filename ->
      let source = read_file filename in
      let lexbuf = Lexing.from_string source in
      lexbuf.Lexing.lex_curr_p <-
        { lexbuf.Lexing.lex_curr_p with Lexing.pos_fname = filename };

      (* Parse *)
      let ast =
        try Rake.Parser.program Rake.Lexer.token lexbuf with
        | Rake.Parser.Error ->
            let pos = lexbuf.Lexing.lex_curr_p in
            Printf.eprintf "Parse error at %s:%d:%d\n" pos.Lexing.pos_fname
              pos.Lexing.pos_lnum
              (pos.Lexing.pos_cnum - pos.Lexing.pos_bol);
            exit 1
        | Rake.Lexer.Error (msg, pos) ->
            Printf.eprintf "Lexical error at %s:%d:%d: %s\n"
              pos.Lexing.pos_fname pos.Lexing.pos_lnum
              (pos.Lexing.pos_cnum - pos.Lexing.pos_bol)
              msg;
            exit 1
      in

      if !dump_ast then begin
        Printf.printf "%s\n" (Rake.Ast.show_program ast);
        exit 0
      end;

      (* Type check *)
      let env =
        match Rake.Check.check_program ast with
        | Ok env -> env
        | Error e ->
            Printf.eprintf "Type error: %s\n" (Rake.Check.show_error e);
            exit 1
      in

      if !dump_types then begin
        Printf.printf "Type checking passed\n";
        Hashtbl.iter
          (fun name typ ->
            Printf.printf "  %s : %s\n" name (Rake.Types.show typ))
          env.Rake.Check.vars;
        exit 0
      end;

      (* Emit code *)
      if !emit_llvm then begin
        let llvm_ir = Rake.Emit.emit_program env ast in
        let output =
          match !output_file with
          | Some f -> f
          | None -> Filename.chop_extension filename ^ ".ll"
        in
        write_file output llvm_ir;
        Printf.printf "Wrote LLVM IR: %s\n" output
      end
      else begin
        let mlir = Rake.Mlir.emit_program env ast in
        let output =
          match !output_file with
          | Some f -> f
          | None -> Filename.chop_extension filename ^ ".mlir"
        in
        write_file output mlir;
        Printf.printf "Wrote MLIR: %s\n" output
      end
