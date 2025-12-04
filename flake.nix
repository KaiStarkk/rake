{
  description = "Rake - a vector-first programming language";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        ocamlPackages = pkgs.ocaml-ng.ocamlPackages_4_14;
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = with ocamlPackages; [
            # Core OCaml
            ocaml
            dune_3
            findlib

            # Rake compiler deps
            menhir
            ppx_deriving

            # Eval arena deps
            yojson
            cmdliner

            # Dev tools
            ocaml-lsp
            ocamlformat
          ] ++ (with pkgs; [
            # MLIR/LLVM toolchain for compilation pipeline
            llvmPackages.mlir
            llvmPackages.llvm

            # Benchmarking tools
            hyperfine
            time

            # Competitor compilers (optional, for eval arena)
            gcc
            rustc
            cargo
            zig
            # mojo  # Not in nixpkgs yet
            # bend  # Not in nixpkgs yet
            odin
          ]);
        };
      });
}
