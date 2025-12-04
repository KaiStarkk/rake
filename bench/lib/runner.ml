(** Benchmark Runner Module

    This module handles execution of compiled benchmarks
    and collection of timing/performance data.
*)

open Config
open Compiler

(** A single benchmark run result *)
type run_result = {
  elapsed_sec : float;
  throughput : float option;     (** Items/sec if applicable *)
  memory_kb : int option;        (** Peak memory usage *)
  output : string;               (** Program stdout *)
  error : string;                (** Program stderr *)
  exit_code : int;
}

(** Aggregated results from multiple runs *)
type benchmark_result = {
  benchmark : benchmark;
  compile_result : compile_result;
  runs : run_result list;
  mean_time_sec : float;
  stddev_time_sec : float;
  min_time_sec : float;
  max_time_sec : float;
  mean_throughput : float option;
}

(** Default parameters for each application *)
let default_params = function
  | Particles -> [("particles", "1000000"); ("iterations", "100")]
  | Mandelbrot -> [("width", "1920"); ("height", "1080"); ("iterations", "1000")]
  | DSP -> [("samples", "1000000"); ("filter_size", "64")]
  | Filters -> [("width", "1920"); ("height", "1080")]
  | RayTracing -> [("width", "800"); ("height", "600"); ("samples", "16")]
  | NBody -> [("bodies", "10000"); ("steps", "100")]
  | Inference -> [("batch_size", "32"); ("input_size", "784")]
  | Physics -> [("objects", "10000"); ("steps", "100")]

(** Build command line arguments from params *)
let build_args params =
  List.map (fun (_, v) -> v) params |> String.concat " "

(** Run a single benchmark iteration *)
let run_once executable params =
  let args = build_args params in
  let cmd = Printf.sprintf "%s %s" executable args in

  (* Use /usr/bin/time for memory measurement if available *)
  let time_cmd = Printf.sprintf "/usr/bin/time -v %s" cmd in
  let start = Unix.gettimeofday () in

  let stdout_file = Filename.temp_file "bench_out" ".txt" in
  let stderr_file = Filename.temp_file "bench_err" ".txt" in
  let full_cmd = Printf.sprintf "%s >%s 2>%s" time_cmd stdout_file stderr_file in

  let exit_status = Unix.system full_cmd in
  let elapsed = Unix.gettimeofday () -. start in

  let read_file f =
    let ic = open_in f in
    let n = in_channel_length ic in
    let s = really_input_string ic n in
    close_in ic;
    Sys.remove f;
    s
  in

  let output = read_file stdout_file in
  let error = read_file stderr_file in
  let exit_code = match exit_status with
    | Unix.WEXITED c -> c
    | Unix.WSIGNALED s -> -s
    | Unix.WSTOPPED s -> -s
  in

  (* Try to extract memory from time output *)
  let memory_kb =
    let pattern = Str.regexp "Maximum resident set size.*: \\([0-9]+\\)" in
    try
      let _ = Str.search_forward pattern error 0 in
      Some (int_of_string (Str.matched_group 1 error))
    with Not_found -> None
  in

  (* Try to extract throughput from output (format: "X.XX M items/sec") *)
  let throughput =
    let pattern = Str.regexp "\\([0-9.]+\\) M [a-z]+/sec" in
    try
      let _ = Str.search_forward pattern output 0 in
      Some (float_of_string (Str.matched_group 1 output) *. 1e6)
    with Not_found ->
      (* Try alternative format: "X.XX particles/sec" *)
      let pattern2 = Str.regexp "\\([0-9.]+\\) [a-z]+/sec" in
      try
        let _ = Str.search_forward pattern2 output 0 in
        Some (float_of_string (Str.matched_group 1 output))
      with Not_found -> None
  in

  { elapsed_sec = elapsed;
    throughput;
    memory_kb;
    output;
    error;
    exit_code }

(** Calculate statistics from a list of floats *)
let statistics values =
  let n = float_of_int (List.length values) in
  let sum = List.fold_left (+.) 0.0 values in
  let mean = sum /. n in
  let sq_diff = List.map (fun x -> (x -. mean) ** 2.0) values in
  let variance = List.fold_left (+.) 0.0 sq_diff /. n in
  let stddev = sqrt variance in
  let sorted = List.sort compare values in
  let min_val = List.hd sorted in
  let max_val = List.hd (List.rev sorted) in
  (mean, stddev, min_val, max_val)

(** Run a complete benchmark with warmup and measurement *)
let run_benchmark run_config compile_result benchmark =
  match compile_result.executable with
  | None ->
    { benchmark;
      compile_result;
      runs = [];
      mean_time_sec = 0.0;
      stddev_time_sec = 0.0;
      min_time_sec = 0.0;
      max_time_sec = 0.0;
      mean_throughput = None }
  | Some exe ->
    let params =
      if run_config.params = [] then
        default_params benchmark.app
      else
        run_config.params
    in

    (* Warmup runs *)
    Printf.printf "  Warming up (%d runs)..." run_config.warmup_runs;
    flush stdout;
    for _ = 1 to run_config.warmup_runs do
      let _ = run_once exe params in ()
    done;
    Printf.printf " done\n";

    (* Timed runs *)
    Printf.printf "  Running (%d runs)..." run_config.timed_runs;
    flush stdout;
    let runs = List.init run_config.timed_runs (fun i ->
      Printf.printf " %d" (i + 1);
      flush stdout;
      run_once exe params
    ) in
    Printf.printf " done\n";

    (* Calculate statistics *)
    let times = List.map (fun r -> r.elapsed_sec) runs in
    let (mean, stddev, min_val, max_val) = statistics times in

    let throughputs = List.filter_map (fun r -> r.throughput) runs in
    let mean_throughput =
      if throughputs = [] then None
      else Some (List.fold_left (+.) 0.0 throughputs /. float_of_int (List.length throughputs))
    in

    { benchmark;
      compile_result;
      runs;
      mean_time_sec = mean;
      stddev_time_sec = stddev;
      min_time_sec = min_val;
      max_time_sec = max_val;
      mean_throughput }

(** Run all benchmarks for a given application *)
let run_app_benchmarks run_config compiled_benchmarks app =
  let app_benchmarks = List.filter (fun (b, _) -> b.app = app) compiled_benchmarks in
  Printf.printf "\n=== Running %s benchmarks ===\n" (app_to_dir app);
  List.map (fun (b, compile_result) ->
    Printf.printf "\n%s:\n" (lang_to_dir b.lang);
    if compile_result.success then
      run_benchmark run_config compile_result b
    else begin
      Printf.printf "  Skipped (compilation failed)\n";
      { benchmark = b;
        compile_result;
        runs = [];
        mean_time_sec = 0.0;
        stddev_time_sec = 0.0;
        min_time_sec = 0.0;
        max_time_sec = 0.0;
        mean_throughput = None }
    end
  ) app_benchmarks

(** Run all ready benchmarks *)
let run_all run_config compiled_benchmarks =
  let apps = List.sort_uniq compare (List.map (fun (b, _) -> b.app) compiled_benchmarks) in
  List.concat_map (run_app_benchmarks run_config compiled_benchmarks) apps

(** Print results summary *)
let print_results_summary results =
  Printf.printf "\n";
  Printf.printf "=============================================================\n";
  Printf.printf "                    BENCHMARK RESULTS                         \n";
  Printf.printf "=============================================================\n\n";

  (* Group by application *)
  let by_app = List.fold_left (fun acc r ->
    let app = r.benchmark.app in
    let existing = try List.assoc app acc with Not_found -> [] in
    (app, r :: existing) :: List.remove_assoc app acc
  ) [] results in

  List.iter (fun (app, app_results) ->
    Printf.printf "--- %s ---\n" (String.uppercase_ascii (app_to_dir app));
    Printf.printf "%-10s %12s %12s %12s %15s\n"
      "Language" "Mean (s)" "Stddev" "Min (s)" "Throughput";
    Printf.printf "%s\n" (String.make 65 '-');

    let sorted = List.sort (fun r1 r2 ->
      compare r1.mean_time_sec r2.mean_time_sec
    ) app_results in

    List.iter (fun r ->
      if r.compile_result.success && r.runs <> [] then begin
        let throughput_str = match r.mean_throughput with
          | Some t when t > 1e9 -> Printf.sprintf "%.2f G/s" (t /. 1e9)
          | Some t when t > 1e6 -> Printf.sprintf "%.2f M/s" (t /. 1e6)
          | Some t when t > 1e3 -> Printf.sprintf "%.2f K/s" (t /. 1e3)
          | Some t -> Printf.sprintf "%.2f /s" t
          | None -> "N/A"
        in
        Printf.printf "%-10s %12.4f %12.4f %12.4f %15s\n"
          (lang_to_dir r.benchmark.lang)
          r.mean_time_sec
          r.stddev_time_sec
          r.min_time_sec
          throughput_str
      end else
        Printf.printf "%-10s %12s %12s %12s %15s\n"
          (lang_to_dir r.benchmark.lang)
          "FAILED" "-" "-" "-"
    ) sorted;
    Printf.printf "\n"
  ) (List.rev by_app)
