(** Multi-language Compiler Module

    This module handles compilation of benchmarks across different
    languages. Each language has its own compilation pipeline.
*)

open Config

(** Compilation result *)
type compile_result = {
  success : bool;
  executable : string option;
  assembly : string option;      (** Path to .s file if generated *)
  ir_file : string option;       (** Path to IR file (MLIR for Rake, LLVM IR for others) *)
  source_file : string option;   (** Path to original source file *)
  compile_time_ms : float;
  stdout : string;
  stderr : string;
}

(** Compiler configuration *)
type compiler_config = {
  eval_dir : string;
  build_dir : string;
  rake_compiler : string;        (** Path to rake compiler *)
  cc : string;                   (** C compiler (gcc/clang) *)
  rustc : string option;         (** Rust compiler if available *)
  zig : string option;           (** Zig compiler if available *)
  mojo : string option;          (** Mojo compiler if available *)
  bend : string option;          (** Bend compiler if available *)
  odin : string option;          (** Odin compiler if available *)
  optimize : bool;               (** Enable optimizations *)
  target_features : string list; (** e.g. ["avx2"; "fma"] *)
}

(** Default compiler configuration *)
let default_compiler_config ~eval_dir = {
  eval_dir;
  build_dir = Filename.concat eval_dir "_build/arena";
  rake_compiler = "rake";  (* Assumes in PATH or will be set *)
  cc = "gcc";
  rustc = None;
  zig = None;
  mojo = None;
  bend = None;
  odin = None;
  optimize = true;
  target_features = ["native"];
}

(** Run a shell command and capture output *)
let run_command cmd =
  let start = Unix.gettimeofday () in
  let stdout_file = Filename.temp_file "stdout" ".txt" in
  let stderr_file = Filename.temp_file "stderr" ".txt" in
  let full_cmd = Printf.sprintf "%s >%s 2>%s" cmd stdout_file stderr_file in
  let exit_code = Unix.system full_cmd in
  let elapsed = (Unix.gettimeofday () -. start) *. 1000.0 in
  let read_file f =
    let ic = open_in f in
    let n = in_channel_length ic in
    let s = really_input_string ic n in
    close_in ic;
    Sys.remove f;
    s
  in
  let stdout = read_file stdout_file in
  let stderr = read_file stderr_file in
  let success = match exit_code with Unix.WEXITED 0 -> true | _ -> false in
  (success, stdout, stderr, elapsed)

(** Check if a command exists *)
let command_exists cmd =
  let (success, _, _, _) = run_command (Printf.sprintf "which %s" cmd) in
  success

(** Detect available compilers *)
let detect_compilers ~eval_dir =
  let base = default_compiler_config ~eval_dir in
  let cc = if command_exists "clang" then "clang" else "gcc" in
  let rustc = if command_exists "rustc" then Some "rustc" else None in
  let zig = if command_exists "zig" then Some "zig" else None in
  let mojo = if command_exists "mojo" then Some "mojo" else None in
  let bend = if command_exists "bend" then Some "bend" else None in
  let odin = if command_exists "odin" then Some "odin" else None in
  (* Try to find rake compiler relative to eval_dir or use dune-built version *)
  let rake_compiler =
    let candidates = [
      Filename.concat eval_dir "../_build/default/bin/main.exe";
      "_build/default/bin/main.exe";
      "rake";
    ] in
    try
      List.find (fun c ->
        let exists = try Unix.access c [Unix.X_OK]; true with _ -> false in
        exists || command_exists c
      ) candidates
    with Not_found -> "rake"
  in
  { base with cc; rustc; zig; mojo; bend; odin; rake_compiler }

(** Ensure build directory exists *)
let ensure_build_dir config app lang =
  let app_dir = Filename.concat config.build_dir (app_to_dir app) in
  let lang_dir = Filename.concat app_dir (lang_to_dir lang) in
  let _ = Unix.system (Printf.sprintf "mkdir -p %s" lang_dir) in
  lang_dir

(** Compile Rake source to native executable *)
let compile_rake config (benchmark : Config.benchmark) =
  match benchmark.source_file with
  | None -> { success = false; executable = None; assembly = None; ir_file = None; source_file = None;
              compile_time_ms = 0.0; stdout = ""; stderr = "No source file" }
  | Some source ->
    let build_dir = ensure_build_dir config benchmark.app Rake in
    let basename = Filename.basename source |> Filename.remove_extension in
    let mlir_file = Filename.concat build_dir (basename ^ ".mlir") in
    let llvm_file = Filename.concat build_dir (basename ^ ".ll") in
    let obj_file = Filename.concat build_dir (basename ^ ".o") in
    let asm_file = Filename.concat build_dir (basename ^ ".s") in
    let exe_file = Filename.concat build_dir basename in
    let harness_dir = Filename.dirname (Filename.dirname source) in
    let harness = Filename.concat harness_dir "harness.c" in

    (* Step 1: Rake -> MLIR *)
    let cmd1 = Printf.sprintf "%s %s -o %s"
      config.rake_compiler source mlir_file in
    let (ok1, out1, err1, t1) = run_command cmd1 in
    if not ok1 then
      { success = false; executable = None; assembly = None; ir_file = None; source_file = None;
        compile_time_ms = t1; stdout = out1; stderr = err1 }
    else

    (* Step 2: MLIR -> LLVM dialect *)
    let cmd2 = Printf.sprintf
      "mlir-opt %s --convert-vector-to-llvm --convert-math-to-llvm \
       --convert-arith-to-llvm --convert-func-to-llvm \
       --reconcile-unrealized-casts -o %s.lowered.mlir"
      mlir_file mlir_file in
    let (ok2, out2, err2, t2) = run_command cmd2 in
    if not ok2 then
      { success = false; executable = None; assembly = None; ir_file = None; source_file = None;
        compile_time_ms = t1 +. t2; stdout = out1 ^ out2; stderr = err1 ^ err2 }
    else

    (* Step 3: MLIR -> LLVM IR *)
    let cmd3 = Printf.sprintf
      "mlir-translate --mlir-to-llvmir %s.lowered.mlir -o %s"
      mlir_file llvm_file in
    let (ok3, out3, err3, t3) = run_command cmd3 in
    if not ok3 then
      { success = false; executable = None; assembly = None; ir_file = None; source_file = None;
        compile_time_ms = t1 +. t2 +. t3;
        stdout = out1 ^ out2 ^ out3; stderr = err1 ^ err2 ^ err3 }
    else

    (* Step 4: LLVM IR -> Object + Assembly *)
    let opt_flags = if config.optimize then "-O3" else "-O0" in
    let cmd4 = Printf.sprintf "llc %s -filetype=obj %s -o %s && llc %s -filetype=asm %s -o %s"
      opt_flags llvm_file obj_file opt_flags llvm_file asm_file in
    let (ok4, out4, err4, t4) = run_command cmd4 in
    if not ok4 then
      { success = false; executable = None; assembly = None; ir_file = None; source_file = None;
        compile_time_ms = t1 +. t2 +. t3 +. t4;
        stdout = out1 ^ out2 ^ out3 ^ out4;
        stderr = err1 ^ err2 ^ err3 ^ err4 }
    else

    (* Step 5: Link with harness if exists *)
    let total_time = ref (t1 +. t2 +. t3 +. t4) in
    let final_exe = ref None in
    let all_stdout = ref (out1 ^ out2 ^ out3 ^ out4) in
    let all_stderr = ref (err1 ^ err2 ^ err3 ^ err4) in

    if Sys.file_exists harness then begin
      (* Use -mavx2 explicitly since NixOS strips -march=native for reproducibility *)
      let simd_flags = "-mavx2 -mfma" in
      (* Link harness.c with the Rake object file *)
      let cmd5 = Printf.sprintf "%s %s -O3 %s %s %s -o %s -lm"
        config.cc opt_flags simd_flags harness obj_file exe_file in
      let (ok5, out5, err5, t5) = run_command cmd5 in
      total_time := !total_time +. t5;
      all_stdout := !all_stdout ^ out5;
      all_stderr := !all_stderr ^ err5;
      if ok5 then final_exe := Some exe_file
    end else begin
      final_exe := Some obj_file  (* No harness, just return object *)
    end;

    { success = !final_exe <> None;
      executable = !final_exe;
      assembly = Some asm_file;
      ir_file = Some mlir_file;
      source_file = Some source;
      compile_time_ms = !total_time;
      stdout = !all_stdout;
      stderr = !all_stderr }

(** Compile C source *)
let compile_c config (benchmark : Config.benchmark) =
  match benchmark.source_file with
  | None -> { success = false; executable = None; assembly = None; ir_file = None; source_file = None;
              compile_time_ms = 0.0; stdout = ""; stderr = "No source file" }
  | Some source ->
    let build_dir = ensure_build_dir config benchmark.app C in
    let basename = Filename.basename source |> Filename.remove_extension in
    let exe_file = Filename.concat build_dir basename in
    let asm_file = Filename.concat build_dir (basename ^ ".s") in
    let ll_file = Filename.concat build_dir (basename ^ ".ll") in

    let opt_flags = if config.optimize then "-O3 -ffast-math" else "-O0" in
    (* Use explicit AVX2 flags since NixOS strips -march=native *)
    let simd_flags = "-mavx2 -mfma" in

    (* Compile to executable *)
    let cmd1 = Printf.sprintf "%s %s %s -fopenmp-simd %s -o %s -lm"
      config.cc opt_flags simd_flags source exe_file in
    let (ok1, out1, err1, t1) = run_command cmd1 in

    (* Also generate assembly for analysis *)
    let cmd2 = Printf.sprintf "%s %s %s -fopenmp-simd -S %s -o %s"
      config.cc opt_flags simd_flags source asm_file in
    let (ok2, out2, err2, t2) = run_command cmd2 in

    (* Also generate LLVM IR for comparison (if using clang) *)
    let cmd3 = Printf.sprintf "%s %s %s -fopenmp-simd -S -emit-llvm %s -o %s 2>/dev/null"
      config.cc opt_flags simd_flags source ll_file in
    let _ = run_command cmd3 in

    { success = ok1;
      executable = if ok1 then Some exe_file else None;
      assembly = if ok2 then Some asm_file else None;
      ir_file = if Sys.file_exists ll_file then Some ll_file else None;
      source_file = Some source;
      compile_time_ms = t1 +. t2;
      stdout = out1 ^ out2;
      stderr = err1 ^ err2 }

(** Compile Rust source *)
let compile_rust config (benchmark : Config.benchmark) =
  match config.rustc, benchmark.source_file with
  | None, _ -> { success = false; executable = None; assembly = None; ir_file = None; source_file = None;
                 compile_time_ms = 0.0; stdout = ""; stderr = "Rust not available" }
  | _, None -> { success = false; executable = None; assembly = None; ir_file = None; source_file = None;
                 compile_time_ms = 0.0; stdout = ""; stderr = "No source file" }
  | Some _, Some source ->
    let build_dir = ensure_build_dir config benchmark.app Rust in
    let basename = Filename.basename source |> Filename.remove_extension in

    (* Check if this is a Cargo project or standalone file *)
    let source_dir = Filename.dirname source in
    let cargo_toml = Filename.concat source_dir "Cargo.toml" in

    if Sys.file_exists cargo_toml then begin
      (* Cargo project *)
      let cmd = Printf.sprintf "cd %s && RUSTFLAGS='-C target-cpu=native' cargo build --release --quiet"
        source_dir in
      let (ok, out, err, t) = run_command cmd in
      let exe = Filename.concat source_dir "target/release/particles" in  (* TODO: detect name *)
      let final_exe = Filename.concat build_dir basename in
      if ok && Sys.file_exists exe then begin
        let _ = Unix.system (Printf.sprintf "cp %s %s" exe final_exe) in
        { success = true; executable = Some final_exe; assembly = None;
          ir_file = None; source_file = Some source;
          compile_time_ms = t; stdout = out; stderr = err }
      end else
        { success = false; executable = None; assembly = None; ir_file = None; source_file = Some source;
          compile_time_ms = t; stdout = out; stderr = err }
    end else begin
      (* Standalone file *)
      let exe_file = Filename.concat build_dir basename in
      let opt_flags = if config.optimize then "-C opt-level=3" else "" in
      let cmd = Printf.sprintf "rustc %s -C target-cpu=native %s -o %s"
        opt_flags source exe_file in
      let (ok, out, err, t) = run_command cmd in
      { success = ok;
        executable = if ok then Some exe_file else None;
        assembly = None;
        ir_file = None;
        source_file = Some source;
        compile_time_ms = t;
        stdout = out;
        stderr = err }
    end

(** Compile Zig source *)
let compile_zig config (benchmark : Config.benchmark) =
  match config.zig, benchmark.source_file with
  | None, _ -> { success = false; executable = None; assembly = None; ir_file = None; source_file = None;
                 compile_time_ms = 0.0; stdout = ""; stderr = "Zig not available" }
  | _, None -> { success = false; executable = None; assembly = None; ir_file = None; source_file = None;
                 compile_time_ms = 0.0; stdout = ""; stderr = "No source file" }
  | Some _, Some source ->
    let build_dir = ensure_build_dir config benchmark.app Zig in
    let basename = Filename.basename source |> Filename.remove_extension in
    let exe_file = Filename.concat build_dir basename in

    let opt_flag = if config.optimize then "-OReleaseFast" else "" in
    let cmd = Printf.sprintf "zig build-exe %s %s -femit-bin=%s"
      opt_flag source exe_file in
    let (ok, out, err, t) = run_command cmd in

    { success = ok;
      executable = if ok then Some exe_file else None;
      assembly = None;
      ir_file = None;
      source_file = Some source;
      compile_time_ms = t;
      stdout = out;
      stderr = err }

(** Compile Mojo source *)
let compile_mojo config (benchmark : Config.benchmark) =
  match config.mojo, benchmark.source_file with
  | None, _ -> { success = false; executable = None; assembly = None; ir_file = None; source_file = None;
                 compile_time_ms = 0.0; stdout = ""; stderr = "Mojo not available" }
  | _, None -> { success = false; executable = None; assembly = None; ir_file = None; source_file = None;
                 compile_time_ms = 0.0; stdout = ""; stderr = "No source file" }
  | Some _, Some source ->
    let build_dir = ensure_build_dir config benchmark.app Mojo in
    let basename = Filename.basename source |> Filename.remove_extension in
    let exe_file = Filename.concat build_dir basename in

    let cmd = Printf.sprintf "mojo build %s -o %s" source exe_file in
    let (ok, out, err, t) = run_command cmd in

    { success = ok;
      executable = if ok then Some exe_file else None;
      assembly = None;
      ir_file = None;
      source_file = Some source;
      compile_time_ms = t;
      stdout = out;
      stderr = err }

(** Compile Bend source *)
let compile_bend config (benchmark : Config.benchmark) =
  match config.bend, benchmark.source_file with
  | None, _ -> { success = false; executable = None; assembly = None; ir_file = None; source_file = None;
                 compile_time_ms = 0.0; stdout = ""; stderr = "Bend not available" }
  | _, None -> { success = false; executable = None; assembly = None; ir_file = None; source_file = None;
                 compile_time_ms = 0.0; stdout = ""; stderr = "No source file" }
  | Some _, Some source ->
    let build_dir = ensure_build_dir config benchmark.app Bend in
    let basename = Filename.basename source |> Filename.remove_extension in
    let exe_file = Filename.concat build_dir basename in

    (* Bend compiles to HVM and then native code *)
    let cmd = Printf.sprintf "bend compile %s -o %s" source exe_file in
    let (ok, out, err, t) = run_command cmd in

    { success = ok;
      executable = if ok then Some exe_file else None;
      assembly = None;
      ir_file = None;
      source_file = Some source;
      compile_time_ms = t;
      stdout = out;
      stderr = err }

(** Compile Odin source *)
let compile_odin config (benchmark : Config.benchmark) =
  match config.odin, benchmark.source_file with
  | None, _ -> { success = false; executable = None; assembly = None; ir_file = None; source_file = None;
                 compile_time_ms = 0.0; stdout = ""; stderr = "Odin not available" }
  | _, None -> { success = false; executable = None; assembly = None; ir_file = None; source_file = None;
                 compile_time_ms = 0.0; stdout = ""; stderr = "No source file" }
  | Some _, Some source ->
    let build_dir = ensure_build_dir config benchmark.app Odin in
    let basename = Filename.basename source |> Filename.remove_extension in
    let exe_file = Filename.concat build_dir basename in
    let source_dir = Filename.dirname source in

    let opt_flag = if config.optimize then "-o:speed" else "-o:none" in
    let cmd = Printf.sprintf "odin build %s -file %s -out:%s"
      opt_flag source_dir exe_file in
    let (ok, out, err, t) = run_command cmd in

    { success = ok;
      executable = if ok then Some exe_file else None;
      assembly = None;
      ir_file = None;
      source_file = Some source;
      compile_time_ms = t;
      stdout = out;
      stderr = err }

(** Compile a benchmark using the appropriate compiler *)
let compile config (benchmark : Config.benchmark) =
  match benchmark.lang with
  | Rake -> compile_rake config benchmark
  | C -> compile_c config benchmark
  | Rust -> compile_rust config benchmark
  | Zig -> compile_zig config benchmark
  | Mojo -> compile_mojo config benchmark
  | Bend -> compile_bend config benchmark
  | Odin -> compile_odin config benchmark

(** Compile all ready benchmarks *)
let compile_all config benchmarks =
  let ready = ready_benchmarks benchmarks in
  List.map (fun b ->
    Printf.printf "Compiling %s/%s... " (app_to_dir b.app) (lang_to_dir b.lang);
    flush stdout;
    let result = compile config b in
    Printf.printf "%s (%.1f ms)\n"
      (if result.success then "OK" else "FAILED")
      result.compile_time_ms;
    (b, result)
  ) ready
