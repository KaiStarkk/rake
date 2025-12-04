(** Rake Evaluation Arena CLI

    Main entry point for the benchmark arena system.
    Discovers, compiles, runs, and reports on benchmarks.
*)

open Rake_eval_lib
open Cmdliner

(** Get the eval directory (parent of arena/) *)
let get_eval_dir () =
  let exe = Sys.executable_name in
  let dir = Filename.dirname exe in
  if Filename.basename dir = "arena" then
    Filename.dirname dir
  else
    (* Fallback: assume we're in the project root *)
    Filename.concat (Sys.getcwd ()) "eval"

(** Status command - show what's available *)
let status_cmd eval_dir =
  Printf.printf "Rake Evaluation Arena\n";
  Printf.printf "=====================\n\n";
  Printf.printf "Eval directory: %s\n\n" eval_dir;

  (* Discover benchmarks *)
  let benchmarks = Config.discover_benchmarks ~eval_dir in

  Printf.printf "Benchmark Status:\n";
  Config.print_status_table benchmarks;

  let ready_count = List.length (Config.ready_benchmarks benchmarks) in
  let total_count = List.length benchmarks in
  Printf.printf "\nReady to run: %d/%d benchmarks\n" ready_count total_count;

  (* Check available compilers *)
  Printf.printf "\nCompiler Detection:\n";
  let compiler_config = Compiler.detect_compilers ~eval_dir in
  Printf.printf "  C compiler:    %s\n" compiler_config.cc;
  Printf.printf "  Rust compiler: %s\n"
    (match compiler_config.rustc with Some _ -> "available" | None -> "not found");
  Printf.printf "  Zig compiler:  %s\n"
    (match compiler_config.zig with Some _ -> "available" | None -> "not found");
  Printf.printf "  Mojo compiler: %s\n"
    (match compiler_config.mojo with Some _ -> "available" | None -> "not found");
  ()

(** Run command - compile and benchmark *)
let run_cmd eval_dir apps langs warmup runs output_format output_file =
  Printf.printf "Rake Evaluation Arena - Running Benchmarks\n";
  Printf.printf "===========================================\n\n";

  (* Filter benchmarks based on args *)
  let all_benchmarks = Config.discover_benchmarks ~eval_dir in
  let benchmarks =
    let filtered = match apps with
      | [] -> all_benchmarks
      | apps ->
        List.filter (fun b ->
          List.mem (Config.app_to_dir b.Config.app) apps
        ) all_benchmarks
    in
    match langs with
    | [] -> filtered
    | langs ->
      List.filter (fun b ->
        List.mem (Config.lang_to_dir b.Config.lang) langs
      ) filtered
  in

  let ready = Config.ready_benchmarks benchmarks in
  if ready = [] then begin
    Printf.printf "No ready benchmarks found. Run 'rake-arena status' to see available benchmarks.\n";
    exit 1
  end;

  Printf.printf "Benchmarks to run: %d\n\n" (List.length ready);

  (* Setup compiler *)
  let compiler_config = Compiler.detect_compilers ~eval_dir in

  (* Compile all *)
  Printf.printf "=== Compilation Phase ===\n";
  let compiled = Compiler.compile_all compiler_config ready in

  let successful = List.filter (fun (_, r) -> r.Compiler.success) compiled in
  Printf.printf "\nCompiled: %d/%d successful\n\n"
    (List.length successful) (List.length compiled);

  if successful = [] then begin
    Printf.printf "No benchmarks compiled successfully.\n";
    exit 1
  end;

  (* Run benchmarks *)
  Printf.printf "=== Benchmark Phase ===\n";
  let run_config = { Config.default_run_config with
    warmup_runs = warmup;
    timed_runs = runs;
  } in

  let results = Runner.run_all run_config compiled in

  (* Collect metrics *)
  let metrics = List.map Metrics.collect_metrics results in

  (* Generate report *)
  let report_format = match output_format with
    | "json" -> Report.Json
    | "markdown" | "md" -> Report.Markdown
    | "html" -> Report.Html
    | _ -> Report.Console
  in

  let report_config = { Report.default_report_config with
    format = report_format;
    output_file;
  } in

  Report.generate_report report_config results metrics;

  (* Save results *)
  let results_dir = Filename.concat eval_dir "results" in
  let _ = Unix.system (Printf.sprintf "mkdir -p %s" results_dir) in
  let _ = Report.save_results ~results_dir results metrics in
  ()

(** Compare command - compare two result files *)
let compare_cmd old_file new_file =
  let old_json = Report.load_results old_file in
  let new_json = Report.load_results new_file in
  Report.compare_results old_json new_json

(** CLI definitions *)
let eval_dir_arg =
  let doc = "Path to the eval directory" in
  Arg.(value & opt string (get_eval_dir ()) & info ["eval-dir"] ~doc)

let apps_arg =
  let doc = "Applications to benchmark (comma-separated). Default: all" in
  Arg.(value & opt (list string) [] & info ["apps"; "a"] ~doc)

let langs_arg =
  let doc = "Languages to benchmark (comma-separated). Default: all" in
  Arg.(value & opt (list string) [] & info ["langs"; "l"] ~doc)

let warmup_arg =
  let doc = "Number of warmup runs" in
  Arg.(value & opt int 3 & info ["warmup"; "w"] ~doc)

let runs_arg =
  let doc = "Number of timed runs" in
  Arg.(value & opt int 10 & info ["runs"; "r"] ~doc)

let format_arg =
  let doc = "Output format: console, markdown, json" in
  Arg.(value & opt string "console" & info ["format"; "f"] ~doc)

let output_arg =
  let doc = "Output file (for markdown/json formats)" in
  Arg.(value & opt (some string) None & info ["output"; "o"] ~doc)

let old_file_arg =
  let doc = "Old results JSON file" in
  Arg.(required & pos 0 (some string) None & info [] ~docv:"OLD_FILE" ~doc)

let new_file_arg =
  let doc = "New results JSON file" in
  Arg.(required & pos 1 (some string) None & info [] ~docv:"NEW_FILE" ~doc)

(** Command implementations *)
let status_term =
  Term.(const status_cmd $ eval_dir_arg)

let run_term =
  Term.(const run_cmd $ eval_dir_arg $ apps_arg $ langs_arg
        $ warmup_arg $ runs_arg $ format_arg $ output_arg)

let compare_term =
  Term.(const compare_cmd $ old_file_arg $ new_file_arg)

(** Command info *)
let status_info =
  Cmd.info "status" ~doc:"Show benchmark status and available compilers"

let run_info =
  Cmd.info "run" ~doc:"Compile and run benchmarks"

let compare_info =
  Cmd.info "compare" ~doc:"Compare two result files"

let main_info =
  Cmd.info "rake-arena" ~version:"0.1.0"
    ~doc:"Rake language evaluation arena"
    ~man:[
      `S Manpage.s_description;
      `P "The Rake evaluation arena benchmarks the Rake SIMD language \
          against competitor languages (C, Rust, Zig, Mojo) across \
          various applications.";
      `S Manpage.s_commands;
      `P "Use 'rake-arena <command> --help' for command-specific help.";
    ]

let () =
  let status_cmd = Cmd.v status_info status_term in
  let run_cmd = Cmd.v run_info run_term in
  let compare_cmd = Cmd.v compare_info compare_term in

  let main_cmd = Cmd.group main_info [status_cmd; run_cmd; compare_cmd] in
  exit (Cmd.eval main_cmd)
