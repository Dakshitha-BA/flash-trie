#!/usr/bin/env python3
"""Generate a self-contained example workload for FlashTrie.

Produces everything the example scripts need, with no proprietary model or
keyword library:

  1. ``trie.bin``   a FlashTrie constraint index built from synthetic keys.
  2. ``topk.tsv``   a top-K proposal file in the format consumed by
                    ``run_trie_search.py``.
  3. ``keys.txt``   the synthetic key set, for inspection (with --write-keys).

The synthetic workload mimics the shape of a generative-retrieval decoding
problem: a library of ``--num-keys`` identifiers, each a sequence of
``--key-length`` token IDs over a vocabulary of ``--vocab-size``, and a
per-query grid of top-K token proposals of shape ``(key_length, topk)``.

Proposal grids are built by sampling a handful of "seed" keys, unioning their
tokens at each step, and filling the rest of each row with random distractors.
This guarantees that (a) valid paths through the trie exist, so beam search
returns results, and (b) most proposals are invalid, so the trie has to prune
- which is the behaviour the benchmark measures.

Examples
--------
Single file, one wide K that every sweep point can slice down::

    python make_example_data.py --out-dir data --topk 1000

One file per K, matching the sweep drivers' --onemap convention::

    python make_example_data.py --out-dir data \\
        --per-topk-files 100,200,400,600,800,1000
"""

import argparse
import json
import os
import random

from flashtrie_bench import import_flashtrie


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "data"),
        help="Directory to write trie.bin, topk.tsv and keys.txt into")
    p.add_argument("--num-keys", type=int, default=1000000,
                   help="Number of synthetic keys in the constraint library")
    p.add_argument("--key-length", type=int, default=8,
                   help="Maximum tokens per key; individual keys are shorter, "
                        "following --length-weights")
    p.add_argument("--length-weights", default="2,10,45,25,10,5,2,1",
                   help="Relative frequency of key lengths 1, 2, 3, ... The "
                        "default approximates a real tokenized keyword "
                        "library, where most keys are 2-4 tokens")
    p.add_argument("--vocab-size", type=int, default=262144,
                   help="Size of the synthetic token vocabulary")
    p.add_argument("--num-queries", type=int, default=2000,
                   help="Number of queries to emit per top-K file")
    p.add_argument("--topk", type=int, default=100,
                   help="Proposals per decoding step (single-file mode)")
    p.add_argument("--per-topk-files", default=None,
                   help="Comma-separated K values. Writes "
                        "topk/output_topk{K}.tsv for each K, the layout the "
                        "sweep drivers consume in --onemap and directory "
                        "mode. Example: 100,200,400,600,800,1000")
    p.add_argument("--seeds-per-query", type=int, default=128,
                   help="Valid keys seeded into each query's proposal grid")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--write-keys", action="store_true",
                   help="Also write the plain-text key list to keys.txt")
    return p.parse_args()


def make_keys(rng, num_keys, key_length, vocab_size, length_weights):
    """Sample unique keys of varying length, sharing a pool of prefixes.

    Real tokenized keyword libraries are dominated by short keys (2-4 tokens)
    that share prefixes heavily, and the trie's branching comes from that
    sharing. Keys are drawn the same way here: a shared 2-token stem followed
    by a sampled continuation.
    """
    if vocab_size < 2:
        raise SystemExit("--vocab-size must be at least 2")

    weights = length_weights[:key_length]
    if not weights or sum(weights) <= 0:
        raise SystemExit("--length-weights must contain a positive value")
    lengths = list(range(1, len(weights) + 1))

    hi = vocab_size - 1
    # A shared stem pool is what creates deep, branchy prefixes.
    num_stems = max(1, num_keys // 64)
    stems = [(rng.randint(0, hi), rng.randint(0, hi))
             for _ in range(num_stems)]

    keys = set()
    # Bound the loop so an infeasible vocab/length combination cannot hang.
    for _ in range(num_keys * 20):
        if len(keys) >= num_keys:
            break
        length = rng.choices(lengths, weights=weights, k=1)[0]
        stem = stems[rng.randrange(num_stems)]
        key = tuple(stem[:length])
        key += tuple(rng.randint(0, hi) for _ in range(length - len(key)))
        keys.add(key)

    if len(keys) < num_keys:
        print(f"warning: generated {len(keys)} unique keys of the "
              f"{num_keys} requested; raise --vocab-size or --key-length")
    return sorted(keys)


def make_topk_row(rng, seed_tokens, topk, vocab_size):
    """Build one proposal row, returned token-ID-sorted like real data.

    Seed tokens (which continue a valid key) receive the head of the
    log-probability distribution; random distractors receive the tail.
    """
    scored = {}
    for tok in seed_tokens:
        if tok not in scored:
            scored[tok] = -0.2 - 2.0 * rng.random()

    hi = vocab_size - 1
    guard = 0
    while len(scored) < topk and guard < topk * 50:
        guard += 1
        tok = rng.randint(0, hi)
        if tok not in scored:
            scored[tok] = -4.0 - 8.0 * rng.random()
    # Pad deterministically if the vocabulary cannot fill the row.
    filler = 0
    while len(scored) < topk:
        while filler in scored:
            filler += 1
        scored[filler] = -12.0

    ids = sorted(scored)[:topk]
    return ids, [round(scored[t], 4) for t in ids]


def write_topk_file(path, rng, keys, key_length, topk, vocab_size,
                    num_queries, seeds_per_query):
    """Write one ``query<TAB>json`` file with (key_length, topk) grids."""
    with open(path, "w") as fh:
        for q in range(num_queries):
            seeds = [rng.choice(keys)
                     for _ in range(min(seeds_per_query, len(keys)))]
            indices, probs = [], []
            for step in range(key_length):
                # Keys are shorter than the grid, so only the seeds still
                # running at this step contribute a valid continuation.
                ids, logps = make_topk_row(
                    rng, [s[step] for s in seeds if step < len(s)],
                    topk, vocab_size)
                indices.append(ids)
                probs.append(logps)
            record = {"probs": probs, "indices": indices}
            fh.write(f"query_{q}\t{json.dumps(record)}\n")
    return os.path.getsize(path)


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    flashtrie = import_flashtrie()

    print(f"Generating {args.num_keys} keys "
          f"(max length {args.key_length}, vocab {args.vocab_size}) ...")
    weights = [float(w) for w in args.length_weights.split(",") if w.strip()]
    keys = make_keys(rng, args.num_keys, args.key_length, args.vocab_size,
                     weights)
    dist = {}
    for key in keys:
        dist[len(key)] = dist.get(len(key), 0) + 1
    print("  length distribution: "
          + "  ".join(f"{k}:{v}" for k, v in sorted(dist.items())))

    trie_path = os.path.join(args.out_dir, "trie.bin")
    print(f"Building trie -> {trie_path}")
    trie = flashtrie.Trie()
    trie.build([list(k) for k in keys])
    trie.save(trie_path)
    print(f"  #keys:  {trie.num_keys()}")
    print(f"  #nodes: {trie.num_nodes()}")
    print(f"  size:   {trie.total_size() / (1 << 20):.2f} MiB")

    if args.write_keys:
        keys_path = os.path.join(args.out_dir, "keys.txt")
        with open(keys_path, "w") as fh:
            for key in keys:
                fh.write(" ".join(str(t) for t in key) + "\n")
        print(f"Wrote {keys_path}")

    if args.per_topk_files:
        topk_values = [int(v) for v in args.per_topk_files.split(",")
                       if v.strip()]
        topk_dir = os.path.join(args.out_dir, "topk")
        os.makedirs(topk_dir, exist_ok=True)
        print(f"Generating {args.num_queries} queries per K "
              f"-> {topk_dir}/output_topk{{K}}.tsv")
        for topk in topk_values:
            path = os.path.join(topk_dir, f"output_topk{topk}.tsv")
            size = write_topk_file(path, rng, keys, args.key_length, topk,
                                   args.vocab_size, args.num_queries,
                                   args.seeds_per_query)
            print(f"  K={topk:<6} {size / (1 << 20):8.2f} MiB  "
                  f"{os.path.basename(path)}")
        print("\nDone. Try the beam-width sweep:")
        print("  python run_trie_search_BWsweep.py \\")
        print(f"      --trie-path {trie_path} \\")
        print(f"      --data-path '{topk_dir}/output_topk*.tsv' \\")
        print("      --file-values "
              + " ".join(str(v) for v in topk_values) + " \\")
        print("      --onemap --batch-size 1 --output-dir out/bw_sweep")
        return

    topk_path = os.path.join(args.out_dir, "topk.tsv")
    print(f"Generating {args.num_queries} queries -> {topk_path}")
    size = write_topk_file(topk_path, rng, keys, args.key_length, args.topk,
                           args.vocab_size, args.num_queries,
                           args.seeds_per_query)
    print(f"  shape:  ({args.key_length}, {args.topk}) per query")
    print(f"  size:   {size / (1 << 20):.2f} MiB")
    print("\nDone. Try:")
    print("  python run_trie_search.py \\")
    print(f"      --trie-path {trie_path} \\")
    print(f"      --data-path {topk_path} \\")
    print("      --output-path out/example.tsv \\")
    print(f"      --beam-size {args.topk} --topk-tokens {args.topk}")


if __name__ == "__main__":
    main()
