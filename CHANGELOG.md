# Changelog

Versions follow the `VERSION` file. A release tag is `v` plus that version.

## 0.1.0 - 2026-11-18

First installable release of the Python library.

- `pip install .` builds the CUDA extension and installs the `flashtrie` package.
- `flashtrie.__version__` reports the version from `VERSION`.
- `examples/quickstart.py` is the import example for other teams.
- Tagged releases install with
  `pip install "flashtrie @ git+https://github.com/Dakshitha-BA/flash-trie.git@v0.1.0"`.
