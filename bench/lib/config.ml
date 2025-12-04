(** Evaluation Arena Configuration

    This module defines the configuration types and discovery logic
    for the Rake evaluation arena. It automatically discovers which
    applications, languages, and metrics are available for benchmarking.
*)

(** Applications that can be benchmarked *)
type app =
  | Particles     (** Particle simulation - tests branching vectorization *)
  | Mandelbrot    (** Mandelbrot set - tests complex number computation *)
  | DSP           (** Digital signal processing - tests audio/filter pipelines *)
  | Filters       (** Image filters - tests convolution and data layout *)
  | RayTracing    (** Ray tracing - tests recursive structures and BVH *)
  | NBody         (** N-body simulation - tests O(n^2) vs O(n log n) algos *)
  | Inference     (** ML inference - tests matrix ops and activation functions *)
  | Physics       (** Physics simulation - tests collision detection *)
[@@deriving show { with_path = false }]

(** Languages competing in the arena *)
type lang =
  | Rake          (** Our SIMD-first language *)
  | C             (** C baseline (auto-vectorization) *)
  | Rust          (** Rust baseline (auto-vectorization) *)
  | Zig           (** Zig baseline (explicit SIMD or auto) *)
  | Mojo          (** Mojo (Python-like with SIMD) *)
  | Bend          (** Bend (massively parallel, HVM-based) *)
  | Odin          (** Odin (C alternative with SIMD intrinsics) *)
[@@deriving show { with_path = false }]

(** Performance metrics we can measure *)
type metric =
  | Throughput        (** Items processed per second *)
  | Vectorization     (** % of instructions using SIMD *)
  | MemoryBandwidth   (** Memory throughput achieved *)
  | LayoutEfficiency  (** SoA vs AoS efficiency *)
  | CodeSize          (** Binary size *)
  | CompileTime       (** Time to compile *)
[@@deriving show { with_path = false }]

(** Readiness status for a benchmark implementation *)
type status =
  | Ready             (** Fully implemented and working *)
  | WIP               (** Work in progress *)
  | Stub              (** Placeholder only *)
  | Unavailable       (** Not applicable or blocked *)
[@@deriving show { with_path = false }]

(** A benchmark entry: app + language combination *)
type benchmark = {
  app : app;
  lang : lang;
  status : status;
  source_file : string option;  (** Path to source if exists *)
  notes : string option;        (** Any relevant notes *)
}

(** Configuration for running benchmarks *)
type run_config = {
  warmup_runs : int;        (** Number of warmup iterations *)
  timed_runs : int;         (** Number of measured runs *)
  timeout_sec : float;      (** Max time per benchmark *)
  params : (string * string) list;  (** App-specific parameters *)
}

(** Default run configuration *)
let default_run_config = {
  warmup_runs = 3;
  timed_runs = 10;
  timeout_sec = 60.0;
  params = [];
}

(** All applications in the arena *)
let all_apps = [Particles; Mandelbrot; DSP; Filters; RayTracing; NBody; Inference; Physics]

(** All languages in the arena *)
let all_langs = [Rake; C; Rust; Zig; Mojo; Bend; Odin]

(** All metrics we track *)
let all_metrics = [Throughput; Vectorization; MemoryBandwidth; LayoutEfficiency; CodeSize; CompileTime]

(** Convert app to directory name *)
let app_to_dir = function
  | Particles -> "particles"
  | Mandelbrot -> "mandelbrot"
  | DSP -> "dsp"
  | Filters -> "filters"
  | RayTracing -> "raytracing"
  | NBody -> "nbody"
  | Inference -> "inference"
  | Physics -> "physics"

(** Convert language to directory name *)
let lang_to_dir = function
  | Rake -> "rake"
  | C -> "c"
  | Rust -> "rust"
  | Zig -> "zig"
  | Mojo -> "mojo"
  | Bend -> "bend"
  | Odin -> "odin"

(** Convert language to expected source file extension *)
let lang_extension = function
  | Rake -> ".rk"
  | C -> ".c"
  | Rust -> ".rs"
  | Zig -> ".zig"
  | Mojo -> ".mojo"
  | Bend -> ".bend"
  | Odin -> ".odin"

(** Get the expected source filename for an app/lang combination *)
let source_filename app lang =
  let basename = app_to_dir app in
  basename ^ lang_extension lang

(** Check if a file exists *)
let file_exists path =
  try
    Unix.access path [Unix.F_OK];
    true
  with Unix.Unix_error _ -> false

(** Check if string contains a pattern *)
let string_contains s pattern =
  try
    let _ = Str.search_forward (Str.regexp_string pattern) s 0 in
    true
  with Not_found -> false

(** Check for stub markers in file content *)
let check_stub_markers path =
  let ic = open_in path in
  let first_line = try input_line ic with End_of_file -> "" in
  close_in ic;
  (* Check for STUB marker in various comment styles *)
  if string_contains first_line "STUB" then
    `Stub
  else if string_contains first_line "WIP" then
    `WIP
  else
    `Ready

(** Determine the status of a benchmark by examining its source file *)
let detect_status ~bench_dir app lang =
  let lang_dir = Filename.concat bench_dir (lang_to_dir lang) in
  let source = Filename.concat lang_dir (source_filename app lang) in

  (* For Rust, also check for Cargo.toml with src/main.rs *)
  let rust_cargo = Filename.concat lang_dir "Cargo.toml" in
  let rust_main = Filename.concat lang_dir "src/main.rs" in

  if file_exists source then begin
    match check_stub_markers source with
    | `Stub -> (Stub, Some source)
    | `WIP -> (WIP, Some source)
    | `Ready -> (Ready, Some source)
  end
  else if lang = Rust && file_exists rust_cargo && file_exists rust_main then begin
    (* Cargo project - check main.rs for stub markers *)
    match check_stub_markers rust_main with
    | `Stub -> (Stub, Some rust_main)
    | `WIP -> (WIP, Some rust_main)
    | `Ready -> (Ready, Some rust_main)
  end
  else
    (Unavailable, None)

(** Discover all benchmarks in the arena *)
let discover_benchmarks ~eval_dir =
  let bench_dir = Filename.concat eval_dir "bench" in
  List.concat_map (fun app ->
    let app_dir = Filename.concat bench_dir (app_to_dir app) in
    List.map (fun lang ->
      let status, source_file = detect_status ~bench_dir:app_dir app lang in
      { app; lang; status; source_file; notes = None }
    ) all_langs
  ) all_apps

(** Filter benchmarks to only those ready to run *)
let ready_benchmarks benchmarks =
  List.filter (fun b -> b.status = Ready) benchmarks

(** Get benchmarks for a specific app *)
let benchmarks_for_app benchmarks app =
  List.filter (fun b -> b.app = app) benchmarks

(** Get benchmarks for a specific language *)
let benchmarks_for_lang benchmarks lang =
  List.filter (fun b -> b.lang = lang) benchmarks

(** Print a status summary table *)
let print_status_table benchmarks =
  Printf.printf "\n%-12s" "";
  List.iter (fun lang -> Printf.printf " %-8s" (show_lang lang)) all_langs;
  Printf.printf "\n";
  Printf.printf "%s\n" (String.make 84 '-');
  List.iter (fun app ->
    Printf.printf "%-12s" (app_to_dir app);
    List.iter (fun lang ->
      let b = List.find (fun b -> b.app = app && b.lang = lang) benchmarks in
      let symbol = match b.status with
        | Ready -> "[OK]"
        | WIP -> "[WIP]"
        | Stub -> "[---]"
        | Unavailable -> "[ ]"
      in
      Printf.printf " %-8s" symbol
    ) all_langs;
    Printf.printf "\n"
  ) all_apps
