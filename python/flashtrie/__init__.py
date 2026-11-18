"""FlashTrie: GPU-accelerated constrained beam search.

Install from a checkout or a version tag, then import this package. Do not
put the build directory on ``sys.path``.

    pip install .
    pip install "flashtrie @ git+https://github.com/Dakshitha-BA/flash-trie.git@v0.1.0"
"""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    __version__ = version("flashtrie")
except PackageNotFoundError:
    _version_file = Path(__file__).resolve().parents[2] / "VERSION"
    __version__ = _version_file.read_text(encoding="utf-8").strip()

_EXTENSION_EXPORTS = (
    "Trie",
    "tbs",
    "bs_tbs_gpu",
    "load_topk_keys_and_scores",
)

__all__ = [*_EXTENSION_EXPORTS, "__version__"]


def __getattr__(name):
    if name not in _EXTENSION_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        from . import _core
    except ImportError as exc:
        raise ImportError(
            "flashtrie is importable, but the compiled CUDA extension is "
            "missing. On a machine with nvcc, install it from the repository "
            "root with: pip install ."
        ) from exc
    value = getattr(_core, name)
    globals()[name] = value
    return value
