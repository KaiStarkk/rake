(** Metrics Analysis Module

    This module provides advanced metrics analysis beyond simple timing,
    including vectorization analysis, memory layout efficiency, etc.
*)

open Config
open Compiler

(** Vectorization analysis result *)
type vectorization_analysis = {
  total_instructions : int;
  vector_instructions : int;      (** SSE/AVX/NEON instructions *)
  scalar_instructions : int;
  vectorization_ratio : float;    (** vector / total *)
  vector_width : int option;      (** Detected width (128, 256, 512) *)
  key_patterns : string list;     (** Notable instruction patterns *)
}

(** Memory analysis result *)
type memory_analysis = {
  peak_memory_kb : int option;
  estimated_working_set_kb : int option;
  cache_friendly : bool option;   (** Estimated from access patterns *)
}

(** Code quality metrics *)
type code_metrics = {
  binary_size_bytes : int option;
  source_lines : int option;
  compile_time_ms : float;
}

(** Complete metrics for a benchmark *)
type full_metrics = {
  benchmark : benchmark;
  throughput : float option;
  vectorization : vectorization_analysis option;
  memory : memory_analysis option;
  code : code_metrics;
}

(** Known vector instruction patterns for x86-64 *)
let x86_vector_patterns = [
  (* AVX-512 *)
  (Str.regexp "\\bv[a-z]+.*zmm", 512);
  (* AVX2/AVX *)
  (Str.regexp "\\bv[a-z]+.*ymm", 256);
  (* SSE *)
  (Str.regexp "\\b[a-z]+.*xmm", 128);
  (* Specific vector ops *)
  (Str.regexp "\\bvmovups\\b", 256);
  (Str.regexp "\\bvmovaps\\b", 256);
  (Str.regexp "\\bvaddps\\b", 256);
  (Str.regexp "\\bvmulps\\b", 256);
  (Str.regexp "\\bvsubps\\b", 256);
  (Str.regexp "\\bvdivps\\b", 256);
  (Str.regexp "\\bvsqrtps\\b", 256);
  (Str.regexp "\\bvblendvps\\b", 256);
  (Str.regexp "\\bvcmpps\\b", 256);
  (Str.regexp "\\bvfmadd", 256);
]

(** Known scalar instruction patterns *)
let x86_scalar_patterns = [
  (Str.regexp "\\bmovss\\b");
  (Str.regexp "\\baddss\\b");
  (Str.regexp "\\bmulss\\b");
  (Str.regexp "\\bsubss\\b");
  (Str.regexp "\\bdivss\\b");
  (Str.regexp "\\bsqrtss\\b");
]

(** Count matches of a pattern in text *)
let count_pattern pattern text =
  let count = ref 0 in
  let pos = ref 0 in
  try
    while true do
      let _ = Str.search_forward pattern text !pos in
      incr count;
      pos := Str.match_end ()
    done;
    !count
  with Not_found -> !count

(** Analyze vectorization from assembly file *)
let analyze_vectorization assembly_file =
  if not (Sys.file_exists assembly_file) then None
  else begin
    let ic = open_in assembly_file in
    let content = really_input_string ic (in_channel_length ic) in
    close_in ic;

    (* Count vector instructions and detect width *)
    let vector_counts = List.map (fun (pat, width) ->
      (count_pattern pat content, width)
    ) x86_vector_patterns in

    let total_vector = List.fold_left (fun acc (c, _) -> acc + c) 0 vector_counts in

    (* Determine predominant vector width *)
    let width_512 = List.fold_left (fun acc (c, w) -> if w = 512 then acc + c else acc) 0 vector_counts in
    let width_256 = List.fold_left (fun acc (c, w) -> if w = 256 then acc + c else acc) 0 vector_counts in
    let width_128 = List.fold_left (fun acc (c, w) -> if w = 128 then acc + c else acc) 0 vector_counts in

    let detected_width =
      if width_512 > 0 then Some 512
      else if width_256 > 0 then Some 256
      else if width_128 > 0 then Some 128
      else None
    in

    (* Count scalar instructions *)
    let scalar_count = List.fold_left (fun acc pat ->
      acc + count_pattern pat content
    ) 0 x86_scalar_patterns in

    (* Count total arithmetic-ish instructions (rough estimate) *)
    let total_arith = total_vector + scalar_count in

    let ratio =
      if total_arith = 0 then 0.0
      else float_of_int total_vector /. float_of_int total_arith
    in

    (* Identify key patterns present *)
    let key_patterns = List.filter_map (fun (pat, _) ->
      if count_pattern pat content > 0 then
        Some (Str.replace_first (Str.regexp "\\\\b\\|\\*\\|\\+\\|\\.") ""
                (Str.replace_first (Str.regexp "\\\\b") ""
                   (Str.global_replace (Str.regexp "[\\\\]") "" "")))
      else None
    ) x86_vector_patterns in

    Some {
      total_instructions = total_arith;  (* Approximation *)
      vector_instructions = total_vector;
      scalar_instructions = scalar_count;
      vectorization_ratio = ratio;
      vector_width = detected_width;
      key_patterns = List.sort_uniq compare key_patterns;
    }
  end

(** Get binary size *)
let get_binary_size executable =
  match executable with
  | None -> None
  | Some path ->
    if Sys.file_exists path then
      Some (Unix.stat path).Unix.st_size
    else None

(** Count source lines *)
let count_source_lines source_file =
  match source_file with
  | None -> None
  | Some path ->
    if Sys.file_exists path then begin
      let ic = open_in path in
      let count = ref 0 in
      (try
        while true do
          let _ = input_line ic in
          incr count
        done
      with End_of_file -> ());
      close_in ic;
      Some !count
    end else None

(** Collect full metrics for a benchmark result *)
let collect_metrics (result : Runner.benchmark_result) : full_metrics =
  let vectorization = match result.compile_result.assembly with
    | Some asm -> analyze_vectorization asm
    | None -> None
  in

  let memory = {
    peak_memory_kb = (match result.runs with
      | r :: _ -> r.memory_kb
      | [] -> None);
    estimated_working_set_kb = None;
    cache_friendly = None;
  } in

  let code = {
    binary_size_bytes = get_binary_size result.compile_result.executable;
    source_lines = count_source_lines result.benchmark.source_file;
    compile_time_ms = result.compile_result.compile_time_ms;
  } in

  {
    benchmark = result.benchmark;
    throughput = result.mean_throughput;
    vectorization;
    memory = Some memory;
    code;
  }

(** Print vectorization comparison *)
let print_vectorization_report metrics_list =
  Printf.printf "\n";
  Printf.printf "=============================================================\n";
  Printf.printf "                 VECTORIZATION ANALYSIS                       \n";
  Printf.printf "=============================================================\n\n";

  (* Group by application *)
  let by_app = List.fold_left (fun acc m ->
    let app = m.benchmark.app in
    let existing = try List.assoc app acc with Not_found -> [] in
    (app, m :: existing) :: List.remove_assoc app acc
  ) [] metrics_list in

  List.iter (fun (app, app_metrics) ->
    Printf.printf "--- %s ---\n" (String.uppercase_ascii (app_to_dir app));
    Printf.printf "%-10s %8s %8s %10s %10s\n"
      "Language" "Vector%" "Width" "Vec Instr" "Scalar";
    Printf.printf "%s\n" (String.make 55 '-');

    List.iter (fun m ->
      match m.vectorization with
      | None ->
        Printf.printf "%-10s %8s %8s %10s %10s\n"
          (lang_to_dir m.benchmark.lang) "N/A" "-" "-" "-"
      | Some v ->
        let width_str = match v.vector_width with
          | Some w -> string_of_int w
          | None -> "-"
        in
        Printf.printf "%-10s %7.1f%% %8s %10d %10d\n"
          (lang_to_dir m.benchmark.lang)
          (v.vectorization_ratio *. 100.0)
          width_str
          v.vector_instructions
          v.scalar_instructions
    ) app_metrics;
    Printf.printf "\n"
  ) (List.rev by_app)

(** Print code metrics comparison *)
let print_code_metrics_report metrics_list =
  Printf.printf "\n";
  Printf.printf "=============================================================\n";
  Printf.printf "                    CODE METRICS                              \n";
  Printf.printf "=============================================================\n\n";

  let by_app = List.fold_left (fun acc m ->
    let app = m.benchmark.app in
    let existing = try List.assoc app acc with Not_found -> [] in
    (app, m :: existing) :: List.remove_assoc app acc
  ) [] metrics_list in

  List.iter (fun (app, app_metrics) ->
    Printf.printf "--- %s ---\n" (String.uppercase_ascii (app_to_dir app));
    Printf.printf "%-10s %12s %12s %12s\n"
      "Language" "Binary (KB)" "Source LOC" "Compile (ms)";
    Printf.printf "%s\n" (String.make 50 '-');

    List.iter (fun m ->
      let binary_str = match m.code.binary_size_bytes with
        | Some b -> Printf.sprintf "%.1f" (float_of_int b /. 1024.0)
        | None -> "N/A"
      in
      let loc_str = match m.code.source_lines with
        | Some l -> string_of_int l
        | None -> "N/A"
      in
      Printf.printf "%-10s %12s %12s %12.1f\n"
        (lang_to_dir m.benchmark.lang)
        binary_str
        loc_str
        m.code.compile_time_ms
    ) app_metrics;
    Printf.printf "\n"
  ) (List.rev by_app)

(** Theoretical best vectorization estimate *)
let theoretical_best_vectorization app =
  match app with
  | Particles -> 0.95      (* Almost fully vectorizable *)
  | Mandelbrot -> 0.70     (* Divergent iteration counts *)
  | DSP -> 0.98            (* Streaming, highly regular *)
  | Filters -> 0.90        (* Regular convolution *)
  | RayTracing -> 0.50     (* Divergent rays *)
  | NBody -> 0.85          (* Pair interactions *)
  | Inference -> 0.95      (* Matrix ops *)
  | Physics -> 0.75        (* Collision detection has branches *)

(** Print comparison to theoretical best *)
let print_theoretical_comparison metrics_list =
  Printf.printf "\n";
  Printf.printf "=============================================================\n";
  Printf.printf "            VECTORIZATION VS THEORETICAL BEST                 \n";
  Printf.printf "=============================================================\n\n";

  let by_app = List.fold_left (fun acc m ->
    let app = m.benchmark.app in
    let existing = try List.assoc app acc with Not_found -> [] in
    (app, m :: existing) :: List.remove_assoc app acc
  ) [] metrics_list in

  List.iter (fun (app, app_metrics) ->
    let theoretical = theoretical_best_vectorization app in
    Printf.printf "--- %s (Theoretical: %.0f%%) ---\n"
      (String.uppercase_ascii (app_to_dir app)) (theoretical *. 100.0);
    Printf.printf "%-10s %10s %10s %12s\n"
      "Language" "Actual" "Target" "Efficiency";
    Printf.printf "%s\n" (String.make 45 '-');

    List.iter (fun m ->
      match m.vectorization with
      | None ->
        Printf.printf "%-10s %10s %10s %12s\n"
          (lang_to_dir m.benchmark.lang) "N/A" "-" "-"
      | Some v ->
        let efficiency = v.vectorization_ratio /. theoretical *. 100.0 in
        Printf.printf "%-10s %9.1f%% %9.1f%% %11.1f%%\n"
          (lang_to_dir m.benchmark.lang)
          (v.vectorization_ratio *. 100.0)
          (theoretical *. 100.0)
          efficiency
    ) app_metrics;
    Printf.printf "\n"
  ) (List.rev by_app)
