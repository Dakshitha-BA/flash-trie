# Security Policy

## Supported versions

Security fixes are applied to the `main` branch.

## Reporting a vulnerability

Please do **not** report security vulnerabilities through public GitHub issues.

Report them privately through GitHub's private vulnerability reporting:

1. Go to the [Security tab](https://github.com/Dakshitha-BA/flash-trie/security)
   of this repository.
2. Select **Report a vulnerability**.

Please include as much of the following as you can, so the report can be
triaged quickly:

- The type of issue (e.g. buffer overflow, out-of-bounds read, integer
  overflow, denial of service).
- Full paths of the source files related to the issue.
- The affected commit or tag.
- Any special configuration required to reproduce (CUDA version, GPU
  architecture, CMake flags).
- Step-by-step reproduction instructions, plus proof-of-concept input if you
  have one.
- The impact of the issue, including how an attacker might exploit it.

## Scope

FlashTrie parses trie indexes and top-K proposal files, and executes CUDA
kernels over them. Loading an index or input file from an untrusted source is
outside the intended threat model — treat those inputs as trusted. Memory-safety
issues reachable from inputs your own application generates are in scope.
