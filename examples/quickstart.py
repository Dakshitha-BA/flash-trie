"""Import FlashTrie and run one constrained beam search.

Install the package first, from the repository root, on a machine with the
CUDA toolkit:

    pip install .

Pin a release instead of a checkout when another team depends on it:

    pip install "flashtrie @ git+https://github.com/Dakshitha-BA/flash-trie.git@v0.1.0"

Then run this file from anywhere:

    python examples/quickstart.py
"""

import sys


def main():
    try:
        import flashtrie
    except ImportError as exc:
        raise SystemExit(
            "Could not import flashtrie. From the repository root:\n"
            "    pip install .\n"
            "Or pin a release:\n"
            '    pip install "flashtrie @ git+https://github.com/'
            'Dakshitha-BA/flash-trie.git@v0.1.0"'
        ) from exc

    print(f"flashtrie {flashtrie.__version__}")

    # Tokenized identifiers. Each inner list is one key in the constraint trie.
    keys = [[10, 25, 3], [10, 25, 91], [7, 40, 1]]
    trie = flashtrie.Trie()
    trie.build(keys)
    print(f"built {trie.num_keys()} keys, {trie.total_size()} bytes")

    # One row per decoding step. Rows must be sorted by token id.
    beams = [8, 8, 8]
    topk_id = [[7, 10, 99], [25, 40, 99], [1, 3, 91]]
    topk_logp = [[-0.2, -0.1, -8.0], [-0.1, -0.2, -8.0], [-0.4, -0.1, -0.2]]

    use_gpu = True
    try:
        trie.init_tbs_gpu(
            max_batch_size=1,
            max_num_topk=8,
            max_num_out_beams=8,
            max_num_inter_beams=8 * len(beams),
        )
    except Exception as exc:
        print(f"GPU workspace unavailable ({exc}); using the CPU path")
        use_gpu = False

    if use_gpu:
        sequences, scores, elapsed_ms, _phases = flashtrie.bs_tbs_gpu(
            trie,
            1,
            [beams],
            [topk_id],
            [topk_logp],
            token_logp_threshold=-10.0,
            sents_logp_threshold=-40.0,
            length_norm=5.0,
            early_exit=False,
        )
        sequences, scores = sequences[0], scores[0]
        path = "GPU"
    else:
        sequences, scores, elapsed_ms = flashtrie.tbs(
            trie,
            beams,
            topk_id,
            topk_logp,
            token_logp_threshold=-10.0,
            sents_logp_threshold=-40.0,
            length_norm=5.0,
            early_exit=False,
            use_gpu=False,
        )
        path = "CPU"

    print(f"{path} search {elapsed_ms:.2f} ms, {len(sequences)} candidates")
    for sequence, score in zip(sequences, scores):
        print(list(sequence), float(score))
    return 0


if __name__ == "__main__":
    sys.exit(main())
