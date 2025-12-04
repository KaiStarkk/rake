(** Report Generation Module

    This module generates reports in various formats (console, Markdown, JSON)
    from benchmark results.
*)

open Config

(** Report format options *)
type report_format =
  | Console
  | Markdown
  | Json
  | Html

(** Report content configuration *)
type report_config = {
  format : report_format;
  include_vectorization : bool;
  include_code_metrics : bool;
  include_theoretical : bool;
  output_file : string option;
}

(** Default report configuration *)
let default_report_config = {
  format = Console;
  include_vectorization = true;
  include_code_metrics = true;
  include_theoretical = true;
  output_file = None;
}

(** Generate JSON report *)
let generate_json (results : Runner.benchmark_result list) (metrics : Metrics.full_metrics list) =
  let result_to_json (r : Runner.benchmark_result) =
    `Assoc [
      ("app", `String (app_to_dir r.benchmark.app));
      ("lang", `String (lang_to_dir r.benchmark.lang));
      ("compiled", `Bool r.compile_result.success);
      ("mean_time_sec", `Float r.mean_time_sec);
      ("stddev_time_sec", `Float r.stddev_time_sec);
      ("min_time_sec", `Float r.min_time_sec);
      ("max_time_sec", `Float r.max_time_sec);
      ("throughput", match r.mean_throughput with
        | Some t -> `Float t
        | None -> `Null);
      ("compile_time_ms", `Float r.compile_result.compile_time_ms);
    ]
  in

  let metric_to_json (m : Metrics.full_metrics) =
    let vec_json = match m.vectorization with
      | None -> `Null
      | Some v -> `Assoc [
          ("ratio", `Float v.vectorization_ratio);
          ("width", match v.vector_width with Some w -> `Int w | None -> `Null);
          ("vector_instructions", `Int v.vector_instructions);
          ("scalar_instructions", `Int v.scalar_instructions);
        ]
    in
    `Assoc [
      ("app", `String (app_to_dir m.benchmark.app));
      ("lang", `String (lang_to_dir m.benchmark.lang));
      ("vectorization", vec_json);
      ("binary_size_bytes", match m.code.binary_size_bytes with
        | Some s -> `Int s | None -> `Null);
      ("source_lines", match m.code.source_lines with
        | Some l -> `Int l | None -> `Null);
    ]
  in

  `Assoc [
    ("timestamp", `String (string_of_float (Unix.gettimeofday ())));
    ("results", `List (List.map result_to_json results));
    ("metrics", `List (List.map metric_to_json metrics));
  ]

(** Generate Markdown report *)
(** Format timestamp as ISO-8601 string *)
let format_timestamp () =
  let tm = Unix.gmtime (Unix.gettimeofday ()) in
  Printf.sprintf "%04d-%02d-%02d %02d:%02d:%02d UTC"
    (tm.Unix.tm_year + 1900)
    (tm.Unix.tm_mon + 1)
    tm.Unix.tm_mday
    tm.Unix.tm_hour
    tm.Unix.tm_min
    tm.Unix.tm_sec

let generate_markdown (results : Runner.benchmark_result list) (metrics : Metrics.full_metrics list) config =
  let buf = Buffer.create 4096 in
  let add = Buffer.add_string buf in
  let addln s = add s; add "\n" in

  addln "# Rake Evaluation Arena Report";
  addln "";
  addln (Printf.sprintf "_Generated: %s_" (format_timestamp ()));
  addln "";

  (* Group results by app *)
  let by_app = List.fold_left (fun acc r ->
    let app = r.Runner.benchmark.app in
    let existing = try List.assoc app acc with Not_found -> [] in
    (app, r :: existing) :: List.remove_assoc app acc
  ) [] results in

  (* Performance Results *)
  addln "## Performance Results";
  addln "";

  List.iter (fun (app, app_results) ->
    addln (Printf.sprintf "### %s" (String.uppercase_ascii (app_to_dir app)));
    addln "";
    addln "| Language | Mean (s) | Stddev | Min (s) | Throughput |";
    addln "|----------|----------|--------|---------|------------|";

    let sorted = List.sort (fun r1 r2 ->
      compare r1.Runner.mean_time_sec r2.Runner.mean_time_sec
    ) app_results in

    List.iter (fun (r : Runner.benchmark_result) ->
      if r.compile_result.success && r.runs <> [] then begin
        let throughput_str = match r.mean_throughput with
          | Some t when t > 1e9 -> Printf.sprintf "%.2f G/s" (t /. 1e9)
          | Some t when t > 1e6 -> Printf.sprintf "%.2f M/s" (t /. 1e6)
          | Some t when t > 1e3 -> Printf.sprintf "%.2f K/s" (t /. 1e3)
          | Some t -> Printf.sprintf "%.2f /s" t
          | None -> "N/A"
        in
        addln (Printf.sprintf "| %s | %.4f | %.4f | %.4f | %s |"
          (lang_to_dir r.benchmark.lang)
          r.mean_time_sec
          r.stddev_time_sec
          r.min_time_sec
          throughput_str)
      end else
        addln (Printf.sprintf "| %s | FAILED | - | - | - |"
          (lang_to_dir r.benchmark.lang))
    ) sorted;
    addln ""
  ) (List.rev by_app);

  (* Vectorization Analysis *)
  if config.include_vectorization then begin
    addln "## Vectorization Analysis";
    addln "";

    let metrics_by_app = List.fold_left (fun acc m ->
      let app = m.Metrics.benchmark.app in
      let existing = try List.assoc app acc with Not_found -> [] in
      (app, m :: existing) :: List.remove_assoc app acc
    ) [] metrics in

    List.iter (fun (app, app_metrics) ->
      addln (Printf.sprintf "### %s" (String.uppercase_ascii (app_to_dir app)));
      addln "";
      addln "| Language | Vector% | Width | Vec Instr | Scalar |";
      addln "|----------|---------|-------|-----------|--------|";

      List.iter (fun (m : Metrics.full_metrics) ->
        match m.vectorization with
        | None ->
          addln (Printf.sprintf "| %s | N/A | - | - | - |"
            (lang_to_dir m.benchmark.lang))
        | Some v ->
          let width_str = match v.vector_width with
            | Some w -> string_of_int w
            | None -> "-"
          in
          addln (Printf.sprintf "| %s | %.1f%% | %s | %d | %d |"
            (lang_to_dir m.benchmark.lang)
            (v.vectorization_ratio *. 100.0)
            width_str
            v.vector_instructions
            v.scalar_instructions)
      ) app_metrics;
      addln ""
    ) (List.rev metrics_by_app)
  end;

  (* Code Metrics *)
  if config.include_code_metrics then begin
    addln "## Code Metrics";
    addln "";
    addln "| App | Language | Binary (KB) | Source LOC | Compile (ms) |";
    addln "|-----|----------|-------------|------------|--------------|";

    List.iter (fun (m : Metrics.full_metrics) ->
      let binary_str = match m.code.binary_size_bytes with
        | Some b -> Printf.sprintf "%.1f" (float_of_int b /. 1024.0)
        | None -> "N/A"
      in
      let loc_str = match m.code.source_lines with
        | Some l -> string_of_int l
        | None -> "N/A"
      in
      addln (Printf.sprintf "| %s | %s | %s | %s | %.1f |"
        (app_to_dir m.benchmark.app)
        (lang_to_dir m.benchmark.lang)
        binary_str
        loc_str
        m.code.compile_time_ms)
    ) metrics;
    addln ""
  end;

  Buffer.contents buf

(** Read file contents safely, returning None if file doesn't exist *)
let read_file_contents path =
  if Sys.file_exists path then
    try
      let ic = open_in path in
      let n = in_channel_length ic in
      let s = really_input_string ic n in
      close_in ic;
      Some s
    with _ -> None
  else
    None

(** Escape HTML special characters *)
let html_escape s =
  let buf = Buffer.create (String.length s * 2) in
  String.iter (fun c ->
    match c with
    | '<' -> Buffer.add_string buf "&lt;"
    | '>' -> Buffer.add_string buf "&gt;"
    | '&' -> Buffer.add_string buf "&amp;"
    | '"' -> Buffer.add_string buf "&quot;"
    | _ -> Buffer.add_char buf c
  ) s;
  Buffer.contents buf

(** Get language name for highlight.js *)
let hljs_lang = function
  | Config.Rake -> "plaintext"  (* Custom, will style as generic code *)
  | Config.C -> "c"
  | Config.Rust -> "rust"
  | Config.Zig -> "zig"
  | Config.Mojo -> "python"  (* Closest match *)
  | Config.Bend -> "haskell"  (* Closest match for functional style *)
  | Config.Odin -> "go"  (* Closest match *)

(** Generate HTML report with Chart.js graphs *)
let generate_html (results : Runner.benchmark_result list) (metrics : Metrics.full_metrics list) _config =
  let buf = Buffer.create 8192 in
  let add = Buffer.add_string buf in
  let addln s = add s; add "\n" in

  (* Group results by app *)
  let by_app = List.fold_left (fun acc r ->
    let app = r.Runner.benchmark.app in
    let existing = try List.assoc app acc with Not_found -> [] in
    (app, r :: existing) :: List.remove_assoc app acc
  ) [] results in

  (* HTML Header with Chart.js and highlight.js *)
  addln "<!DOCTYPE html>";
  addln "<html lang=\"en\">";
  addln "<head>";
  addln "  <meta charset=\"UTF-8\">";
  addln "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">";
  addln "  <title>Rake Evaluation Arena Report</title>";
  addln "  <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>";
  addln "  <link rel=\"stylesheet\" href=\"https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css\">";
  addln "  <script src=\"https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js\"></script>";
  addln "  <script src=\"https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/x86asm.min.js\"></script>";
  addln "  <script src=\"https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/llvm.min.js\"></script>";
  addln "  <style>";
  addln "    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }";
  addln "    .container { max-width: 1400px; margin: 0 auto; }";
  addln "    h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }";
  addln "    h2 { color: #555; margin-top: 40px; }";
  addln "    h3 { color: #666; margin-top: 20px; }";
  addln "    .card { background: white; border-radius: 8px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }";
  addln "    .chart-container { position: relative; height: 300px; margin: 20px 0; }";
  addln "    table { width: 100%; border-collapse: collapse; margin: 20px 0; }";
  addln "    th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }";
  addln "    th { background: #4CAF50; color: white; }";
  addln "    tr:hover { background: #f5f5f5; }";
  addln "    .status-ok { color: #4CAF50; font-weight: bold; }";
  addln "    .status-failed { color: #f44336; }";
  addln "    .timestamp { color: #999; font-size: 0.9em; }";
  addln "    .metric { display: inline-block; padding: 5px 10px; margin: 5px; background: #e3f2fd; border-radius: 4px; }";
  addln "    /* Code display styles */";
  addln "    .code-section { margin: 20px 0; }";
  addln "    .code-tabs { display: flex; gap: 10px; margin-bottom: 10px; }";
  addln "    .code-tab { padding: 8px 16px; background: #e0e0e0; border: none; border-radius: 4px 4px 0 0; cursor: pointer; font-weight: 500; }";
  addln "    .code-tab.active { background: #4CAF50; color: white; }";
  addln "    .code-content { display: none; }";
  addln "    .code-content.active { display: block; }";
  addln "    .code-wrapper { background: #f8f8f8; border: 1px solid #ddd; border-radius: 4px; overflow: hidden; }";
  addln "    .code-header { background: #e8e8e8; padding: 8px 12px; font-size: 12px; color: #666; border-bottom: 1px solid #ddd; }";
  addln "    pre { margin: 0; padding: 16px; overflow-x: auto; max-height: 500px; font-size: 13px; line-height: 1.4; }";
  addln "    code { font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Fira Code', monospace; }";
  addln "    .lang-badge { display: inline-block; padding: 2px 8px; background: #4CAF50; color: white; border-radius: 3px; font-size: 11px; margin-left: 10px; }";
  addln "    .comparison-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(600px, 1fr)); gap: 20px; }";
  addln "    .comparison-col { background: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }";
  addln "    .comparison-col h4 { margin-top: 0; color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 8px; }";
  addln "  </style>";
  addln "</head>";
  addln "<body>";
  addln "<div class=\"container\">";
  addln (Printf.sprintf "<h1>Rake Evaluation Arena Report</h1>");
  addln (Printf.sprintf "<p class=\"timestamp\">Generated: %s</p>" (format_timestamp ()));

  (* Summary stats *)
  let total = List.length results in
  let successful = List.filter (fun r -> r.Runner.compile_result.success && r.Runner.runs <> []) results in
  let num_successful = List.length successful in
  addln "<div class=\"card\">";
  addln "<h2>Summary</h2>";
  addln (Printf.sprintf "<p>Benchmarks run: <strong>%d</strong> | Successful: <strong>%d</strong></p>" total num_successful);
  addln "</div>";

  (* Performance charts for each app *)
  List.iteri (fun idx (app, app_results) ->
    let chart_id = Printf.sprintf "chart_%d" idx in
    addln "<div class=\"card\">";
    addln (Printf.sprintf "<h2>%s</h2>" (String.uppercase_ascii (app_to_dir app)));

    (* Table *)
    addln "<table>";
    addln "<tr><th>Language</th><th>Mean (s)</th><th>Stddev</th><th>Min (s)</th><th>Throughput</th><th>Status</th></tr>";

    let sorted = List.sort (fun r1 r2 ->
      compare r1.Runner.mean_time_sec r2.Runner.mean_time_sec
    ) app_results in

    List.iter (fun (r : Runner.benchmark_result) ->
      if r.compile_result.success && r.runs <> [] then begin
        let throughput_str = match r.mean_throughput with
          | Some t when t > 1e9 -> Printf.sprintf "%.2f G/s" (t /. 1e9)
          | Some t when t > 1e6 -> Printf.sprintf "%.2f M/s" (t /. 1e6)
          | Some t when t > 1e3 -> Printf.sprintf "%.2f K/s" (t /. 1e3)
          | Some t -> Printf.sprintf "%.2f /s" t
          | None -> "N/A"
        in
        addln (Printf.sprintf "<tr><td>%s</td><td>%.4f</td><td>%.4f</td><td>%.4f</td><td>%s</td><td class=\"status-ok\">OK</td></tr>"
          (lang_to_dir r.benchmark.lang)
          r.mean_time_sec
          r.stddev_time_sec
          r.min_time_sec
          throughput_str)
      end else
        addln (Printf.sprintf "<tr><td>%s</td><td>-</td><td>-</td><td>-</td><td>-</td><td class=\"status-failed\">FAILED</td></tr>"
          (lang_to_dir r.benchmark.lang))
    ) sorted;
    addln "</table>";

    (* Bar chart for throughput *)
    let valid_results = List.filter (fun r -> r.Runner.mean_throughput <> None && r.Runner.compile_result.success) sorted in
    if valid_results <> [] then begin
      addln (Printf.sprintf "<div class=\"chart-container\"><canvas id=\"%s\"></canvas></div>" chart_id);
      addln "<script>";
      addln (Printf.sprintf "new Chart(document.getElementById('%s'), {" chart_id);
      addln "  type: 'bar',";
      addln "  data: {";
      addln (Printf.sprintf "    labels: [%s],"
        (String.concat ", " (List.map (fun r -> Printf.sprintf "'%s'" (lang_to_dir r.Runner.benchmark.lang)) valid_results)));
      addln "    datasets: [{";
      addln "      label: 'Throughput (M/s)',";
      addln (Printf.sprintf "      data: [%s],"
        (String.concat ", " (List.map (fun r ->
          match r.Runner.mean_throughput with
          | Some t -> Printf.sprintf "%.2f" (t /. 1e6)
          | None -> "0"
        ) valid_results)));
      addln "      backgroundColor: ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336', '#00BCD4', '#795548'],";
      addln "      borderWidth: 1";
      addln "    }]";
      addln "  },";
      addln "  options: {";
      addln "    responsive: true,";
      addln "    maintainAspectRatio: false,";
      addln "    plugins: { legend: { display: false } },";
      addln "    scales: { y: { beginAtZero: true, title: { display: true, text: 'Million items/sec' } } }";
      addln "  }";
      addln "});";
      addln "</script>"
    end;
    addln "</div>"
  ) (List.rev by_app);

  (* Vectorization comparison chart *)
  let vec_data = List.filter_map (fun m ->
    match m.Metrics.vectorization with
    | Some v -> Some (m.benchmark, v.Metrics.vectorization_ratio)
    | None -> None
  ) metrics in

  if vec_data <> [] then begin
    addln "<div class=\"card\">";
    addln "<h2>Vectorization Comparison</h2>";
    addln "<div class=\"chart-container\"><canvas id=\"vec_chart\"></canvas></div>";
    addln "<script>";
    addln "new Chart(document.getElementById('vec_chart'), {";
    addln "  type: 'bar',";
    addln "  data: {";
    addln (Printf.sprintf "    labels: [%s],"
      (String.concat ", " (List.map (fun (b, _) ->
        Printf.sprintf "'%s/%s'" (app_to_dir b.app) (lang_to_dir b.lang)
      ) vec_data)));
    addln "    datasets: [{";
    addln "      label: 'Vectorization %',";
    addln (Printf.sprintf "      data: [%s],"
      (String.concat ", " (List.map (fun (_, r) -> Printf.sprintf "%.1f" (r *. 100.0)) vec_data)));
    addln "      backgroundColor: '#2196F3'";
    addln "    }]";
    addln "  },";
    addln "  options: {";
    addln "    responsive: true,";
    addln "    maintainAspectRatio: false,";
    addln "    scales: { y: { beginAtZero: true, max: 100, title: { display: true, text: 'Vectorization %' } } }";
    addln "  }";
    addln "});";
    addln "</script>";
    addln "</div>"
  end;

  (* Code Comparison Section *)
  let successful_results = List.filter (fun r -> r.Runner.compile_result.Compiler.success) results in
  if successful_results <> [] then begin
    addln "<div class=\"card\">";
    addln "<h2>Code Comparison</h2>";
    addln "<p>Compare source code, intermediate representation, and assembly output across implementations.</p>";

    (* Group by app *)
    let by_app = List.fold_left (fun acc r ->
      let app = r.Runner.benchmark.app in
      let existing = try List.assoc app acc with Not_found -> [] in
      (app, r :: existing) :: List.remove_assoc app acc
    ) [] successful_results in

    List.iter (fun (app, app_results) ->
      addln (Printf.sprintf "<h3>%s</h3>" (String.uppercase_ascii (app_to_dir app)));

      (* Source Code Comparison *)
      addln "<h4>Source Code</h4>";
      addln "<div class=\"comparison-grid\">";

      List.iter (fun (r : Runner.benchmark_result) ->
        let lang = r.benchmark.lang in
        let lang_name = lang_to_dir lang in

        match r.compile_result.source_file with
        | Some path ->
          (match read_file_contents path with
           | Some content ->
             addln (Printf.sprintf "<div class=\"comparison-col\">");
             addln (Printf.sprintf "<h4>%s <span class=\"lang-badge\">%s</span></h4>"
               (String.uppercase_ascii lang_name) (Filename.basename path));
             addln "<div class=\"code-wrapper\">";
             addln (Printf.sprintf "<pre><code class=\"language-%s\">%s</code></pre>"
               (hljs_lang lang) (html_escape content));
             addln "</div></div>"
           | None -> ())
        | None -> ()
      ) app_results;
      addln "</div>";

      (* IR Comparison *)
      let has_ir = List.exists (fun r -> r.Runner.compile_result.Compiler.ir_file <> None) app_results in
      if has_ir then begin
        addln "<h4>Intermediate Representation</h4>";
        addln "<div class=\"comparison-grid\">";

        List.iter (fun (r : Runner.benchmark_result) ->
          let lang = r.benchmark.lang in
          let lang_name = lang_to_dir lang in
          let ir_label = match lang with Config.Rake -> "MLIR" | _ -> "LLVM IR" in

          match r.compile_result.ir_file with
          | Some path ->
            (match read_file_contents path with
             | Some content ->
               addln (Printf.sprintf "<div class=\"comparison-col\">");
               addln (Printf.sprintf "<h4>%s %s</h4>" (String.uppercase_ascii lang_name) ir_label);
               addln "<div class=\"code-wrapper\">";
               addln (Printf.sprintf "<div class=\"code-header\">%s</div>" (Filename.basename path));
               addln (Printf.sprintf "<pre><code class=\"language-llvm\">%s</code></pre>"
                 (html_escape content));
               addln "</div></div>"
             | None -> ())
          | None -> ()
        ) app_results;
        addln "</div>"
      end;

      (* Assembly Comparison *)
      let has_asm = List.exists (fun r -> r.Runner.compile_result.Compiler.assembly <> None) app_results in
      if has_asm then begin
        addln "<h4>Assembly Output</h4>";
        addln "<p><em>Showing key vectorized sections. Look for <code>ymm</code> (AVX2 256-bit) or <code>xmm</code> (SSE 128-bit) registers.</em></p>";
        addln "<div class=\"comparison-grid\">";

        List.iter (fun (r : Runner.benchmark_result) ->
          let lang = r.benchmark.lang in
          let lang_name = lang_to_dir lang in

          match r.compile_result.assembly with
          | Some path ->
            (match read_file_contents path with
             | Some content ->
               (* Truncate very long assembly *)
               let truncated = if String.length content > 20000 then
                 (String.sub content 0 20000) ^ "\n\n... (truncated, full file: " ^ path ^ ")"
               else content in
               addln (Printf.sprintf "<div class=\"comparison-col\">");
               addln (Printf.sprintf "<h4>%s Assembly</h4>" (String.uppercase_ascii lang_name));
               addln "<div class=\"code-wrapper\">";
               addln (Printf.sprintf "<div class=\"code-header\">%s</div>" (Filename.basename path));
               addln (Printf.sprintf "<pre><code class=\"language-x86asm\">%s</code></pre>"
                 (html_escape truncated));
               addln "</div></div>"
             | None -> ())
          | None -> ()
        ) app_results;
        addln "</div>"
      end
    ) (List.rev by_app);

    addln "</div>"
  end;

  addln "</div>";
  addln "<script>hljs.highlightAll();</script>";
  addln "</body>";
  addln "</html>";

  Buffer.contents buf

(** Write report to file or stdout *)
let write_report content output_file =
  match output_file with
  | None -> print_string content
  | Some path ->
    let oc = open_out path in
    output_string oc content;
    close_out oc;
    Printf.printf "Report written to: %s\n" path

(** Generate full report *)
let generate_report config (results : Runner.benchmark_result list) (metrics : Metrics.full_metrics list) =
  match config.format with
  | Console ->
    Runner.print_results_summary results;
    if config.include_vectorization then
      Metrics.print_vectorization_report metrics;
    if config.include_code_metrics then
      Metrics.print_code_metrics_report metrics;
    if config.include_theoretical then
      Metrics.print_theoretical_comparison metrics

  | Markdown ->
    let content = generate_markdown results metrics config in
    write_report content config.output_file

  | Json ->
    let json = generate_json results metrics in
    let content = Yojson.Basic.pretty_to_string json in
    write_report content config.output_file

  | Html ->
    let content = generate_html results metrics config in
    write_report content config.output_file

(** Format timestamp for filenames *)
let format_file_timestamp () =
  let tm = Unix.gmtime (Unix.gettimeofday ()) in
  Printf.sprintf "%04d%02d%02d_%02d%02d%02d"
    (tm.Unix.tm_year + 1900)
    (tm.Unix.tm_mon + 1)
    tm.Unix.tm_mday
    tm.Unix.tm_hour
    tm.Unix.tm_min
    tm.Unix.tm_sec

(** Save results for historical tracking *)
let save_results ~results_dir results metrics =
  let timestamp = format_file_timestamp () in
  let json_file = Filename.concat results_dir (Printf.sprintf "results_%s.json" timestamp) in

  let json = generate_json results metrics in
  let oc = open_out json_file in
  output_string oc (Yojson.Basic.pretty_to_string json);
  close_out oc;

  Printf.printf "Results saved to: %s\n" json_file;
  json_file

(** Load historical results for comparison *)
let load_results json_file =
  let content = Yojson.Basic.from_file json_file in
  (* Returns raw JSON for now - could parse into proper types *)
  content

(** Compare two result sets *)
let compare_results old_json new_json =
  Printf.printf "\n";
  Printf.printf "=============================================================\n";
  Printf.printf "                  PERFORMANCE COMPARISON                      \n";
  Printf.printf "=============================================================\n\n";

  let open Yojson.Basic.Util in

  let old_results = old_json |> member "results" |> to_list in
  let new_results = new_json |> member "results" |> to_list in

  Printf.printf "%-12s %-10s %12s %12s %10s\n"
    "App" "Lang" "Old (s)" "New (s)" "Change";
  Printf.printf "%s\n" (String.make 60 '-');

  List.iter (fun new_r ->
    let app = new_r |> member "app" |> to_string in
    let lang = new_r |> member "lang" |> to_string in
    let new_time = new_r |> member "mean_time_sec" |> to_float in

    (* Find matching old result *)
    let old_time_opt = List.find_map (fun old_r ->
      let old_app = old_r |> member "app" |> to_string in
      let old_lang = old_r |> member "lang" |> to_string in
      if old_app = app && old_lang = lang then
        Some (old_r |> member "mean_time_sec" |> to_float)
      else None
    ) old_results in

    match old_time_opt with
    | None ->
      Printf.printf "%-12s %-10s %12s %12.4f %10s\n"
        app lang "N/A" new_time "NEW"
    | Some old_time ->
      let change = (new_time -. old_time) /. old_time *. 100.0 in
      let change_str =
        if change < -1.0 then Printf.sprintf "%.1f%% faster" (-.change)
        else if change > 1.0 then Printf.sprintf "%.1f%% slower" change
        else "~same"
      in
      Printf.printf "%-12s %-10s %12.4f %12.4f %10s\n"
        app lang old_time new_time change_str
  ) new_results
