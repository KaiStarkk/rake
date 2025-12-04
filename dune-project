(lang dune 3.0)
(using menhir 2.1)
(name rake)
(version 0.1.0)
(generate_opam_files true)

(package
 (name rake)
 (synopsis "The rake vector-first programming language")
 (depends
  (ocaml (>= 4.14))
  (menhir (>= 20220210))
  (ppx_deriving (>= 5.0))
  (yojson (>= 2.0))
  (cmdliner (>= 1.1))))
