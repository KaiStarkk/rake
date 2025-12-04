(lang dune 3.20.2)
(using menhir)
(name rake)
(version 0.1.0)
(generate_opam_files true)

(package
 (name rake)
 (synopsis "The rake vector-first programming language")
 (depends
  (ocaml (>= 5.3.0))
  (menhir (>= 20250903))
  (ppx_deriving (>= 6.1.1))
  (yojson (>= 2.2.2))
  (cmdliner (>= 1.3.0))))
