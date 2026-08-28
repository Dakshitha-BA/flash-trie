#!/usr/bin/env python3
"""Run FlashTrie constrained beam search over a top-K proposal file.

Reads a ``query<TAB>json`` TSV, runs GPU-resident constrained beam search
against a FlashTrie index, writes the decoded sequences, and reports latency
percentiles.

Timing methodology
------------------
``--num-runs`` passes are made over the dataset. Pass 1 is a warm-up: its
decoded output is written, but its latency is discarded. Passes 2..N are
timed. Percentiles are computed within each measured pass and then averaged
across passes, so the reported deviation is run-to-run variance.

For stable measurements, pin the process to one NUMA node, e.g.

    numactl --cpunodebind=0 --membind=0 taskset -c 0-7 \\
        python run_trie_search.py ...

Example
-------
    python run_trie_search.py \\
        --trie-path data/trie.bin \\
        --data-path data/topk.tsv \\
        --output-path out/example.tsv \\
        --beam-size 100 --topk-tokens 8 --batch-size 1 \\
        --superset-topk --latency-out out/latency.tsv
"""

import argparse
import os
import sys
import time

import numpy as np

from flashtrie_bench import (
    FlashTrieSearcher,
    iter_batches,
    read_topk_tsv,
    summarize_latency,
)

PERCENTILES = (50, 90, 95, 99)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # I/O
    p.add_argument("--trie-path", required=True,
                   help="Path to the FlashTrie index (built by "
                        "make_example_data.py or tools/marisa-build)")
    p.add_argument("--data-path", required=True,
                   help="Top-K proposal TSV: 'query<TAB>json' per line")
    p.add_argument("--output-path", default=None,
                   help="Where to write decoded sequences (optional)")
    p.add_argument("--latency-out", default=None,
                   help="Write a one-row latency summary TSV here")
    p.add_argument("--phase-out", default=None,
                   help="Write the per-phase GPU timing breakdown here. "
                        "Requires flashtrie built with -DTBS_PROFILE")

    # Search configuration
    p.add_argument("--beam-size", type=int, default=100,
                   help="Beam width B (default: 100)")
    p.add_argument("--topk-tokens", type=int, default=8,
                   help="Proposals K considered per step (default: 8)")
    p.add_argument("--superset-topk", action="store_true",
                   help="Data file has K > --topk-tokens; keep the leading "
                        "--topk-tokens columns so one wide file serves every "
                        "sweep point")
    p.add_argument("--token-logp-threshold", type=float, default=-10.0,
                   help="Drop proposals below this log-probability")
    p.add_argument("--sent-logp-threshold", type=float, default=-40.0,
                   help="Drop hypotheses below this cumulative score")
    p.add_argument("--length-norm", type=float, default=5.0,
                   help="Length-normalisation exponent for final scores")
    p.add_argument("--early-exit", action="store_true",
                   help="Stop a beam once no hypothesis can improve")
    p.add_argument("--trie-start-token", default=None,
                   help="Force a token at step 0 with beam width 1. Pass an "
                        "integer to force that token, or 'lcid' to take it "
                        "from each record's lcid field. Omit (default) when "
                        "the trie stores raw token sequences")

    # Execution
    p.add_argument("--batch-size", type=int, default=1,
                   help="Requests per kernel launch (default: 1)")
    p.add_argument("--max-seq-length", type=int, default=64,
                   help="Sets max_num_inter_beams = beam_size * this")
    p.add_argument("--cpu", action="store_true",
                   help="Run the CPU reference path instead of the GPU "
                        "kernel (useful as a correctness/speedup baseline)")
    p.add_argument("--num-runs", type=int, default=11,
                   help="Total passes over the dataset; pass 1 is warm-up "
                        "(default: 11)")
    p.add_argument("--max-queries", type=int, default=None,
                   help="Read at most this many queries")
    p.add_argument("--logging-steps", type=int, default=1000,
                   help="Log progress every N batches")

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.num_runs < 1:
        raise SystemExit("--num-runs must be >= 1")

    start_token = args.trie_start_token
    if start_token is not None and start_token != "lcid":
        try:
            start_token = int(start_token)
        except ValueError:
            raise SystemExit(
                "--trie-start-token must be an integer or 'lcid'")
        if start_token < 0:
            start_token = None

    print(f"Reading {args.data_path} ...")
    dataset = list(read_topk_tsv(args.data_path, args.max_queries))
    if not dataset:
        raise SystemExit(f"No queries read from {args.data_path}")
    print(f"  {len(dataset)} queries")

    searcher = FlashTrieSearcher(
        trie_path=args.trie_path,
        beam_size=args.beam_size,
        topk_tokens=args.topk_tokens,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        token_logp_threshold=args.token_logp_threshold,
        sent_logp_threshold=args.sent_logp_threshold,
        length_norm=args.length_norm,
        early_exit=args.early_exit,
        use_gpu=not args.cpu,
        superset_topk=args.superset_topk,
        trie_start_token=start_token,
    )

    print(f"\nBeam width      : {args.beam_size}")
    print(f"Top-K tokens    : {args.topk_tokens}")
    print(f"Batch size      : {args.batch_size}")
    print(f"Device          : {'CPU (reference)' if args.cpu else 'GPU'}")
    print(f"Runs            : {args.num_runs} "
          f"(1 warm-up + {args.num_runs - 1} measured)\n")

    output_file = None
    if args.output_path:
        os.makedirs(
            os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
        output_file = open(args.output_path, "w")

    per_run_batch_ms = []
    per_run_throughput = []
    outputs_per_query = []
    grand_start = time.time()

    try:
        for run in range(1, args.num_runs + 1):
            warmup = run == 1
            searcher.reset_timers()
            run_outputs = []
            run_start = time.time()

            for step, batch in enumerate(
                    iter_batches(dataset, args.batch_size)):
                records = [record for _, record in batch]
                tokens, scores = searcher.search(records)

                if warmup:
                    for (query, _), toks in zip(batch, tokens):
                        run_outputs.append(len(toks))
                        if output_file is not None:
                            seqs = [" ".join(str(t) for t in s) for s in toks]
                            output_file.write(
                                f"{query}\t{len(toks)}\t"
                                f"{' | '.join(seqs)}\n")

                if args.logging_steps and step and \
                        step % args.logging_steps == 0:
                    print(f"  run {run}: {step} batches")

            batch_ms = [ms for ms, _ in searcher.batch_times]
            run_wall = time.time() - run_start
            label = "warm-up" if warmup else f"run {run - 1}"
            print(f"  {label:>9}: {np.mean(batch_ms):7.3f} ms/batch  "
                  f"({run_wall:.1f}s wall)")

            if warmup:
                outputs_per_query = run_outputs
            else:
                per_run_batch_ms.append(batch_ms)
                per_run_throughput.append(
                    len(dataset) / run_wall if run_wall else 0.0)
    finally:
        if output_file is not None:
            output_file.close()
            print(f"\nOutputs written to: {args.output_path}")

    print(f"\n{'=' * 56}")
    print("Latency summary")
    print(f"{'=' * 56}")
    if outputs_per_query:
        print(f"Mean outputs/query    : {np.mean(outputs_per_query):.2f}")

    if not per_run_batch_ms:
        print("No measured runs (use --num-runs >= 2 to collect latency).")
        return 0

    stats = summarize_latency(per_run_batch_ms, PERCENTILES)
    mean_qps = float(np.mean(per_run_throughput))
    std_qps = (float(np.std(per_run_throughput, ddof=1))
               if len(per_run_throughput) > 1 else 0.0)

    print(f"Measured runs         : {stats['num_measured_runs']}")
    print(f"Batches per run       : {len(per_run_batch_ms[0])}")
    print(f"Mean latency/batch    : {stats['mean_ms']:.3f} "
          f"+/- {stats['std_ms']:.3f} ms")
    for perc in PERCENTILES:
        print(f"  P{perc:<3}                : "
              f"{stats[f'p{perc}_ms']:.3f} "
              f"+/- {stats[f'std_p{perc}_ms']:.3f} ms")
    print(f"Mean throughput       : {mean_qps:.1f} +/- {std_qps:.1f} queries/s")
    print(f"Total elapsed         : {time.time() - grand_start:.1f}s")

    if args.latency_out:
        os.makedirs(
            os.path.dirname(os.path.abspath(args.latency_out)), exist_ok=True)
        cols = ["beam", "topk", "batch_size", "batches_per_run",
                "num_measured_runs", "mean_ms", "std_ms"]
        vals = [args.beam_size, args.topk_tokens, args.batch_size,
                len(per_run_batch_ms[0]), stats["num_measured_runs"],
                f"{stats['mean_ms']:.3f}", f"{stats['std_ms']:.3f}"]
        for perc in PERCENTILES:
            cols += [f"p{perc}_ms", f"std_p{perc}_ms"]
            vals += [f"{stats[f'p{perc}_ms']:.3f}",
                     f"{stats[f'std_p{perc}_ms']:.3f}"]
        cols += ["mean_qps", "std_qps"]
        vals += [f"{mean_qps:.3f}", f"{std_qps:.3f}"]
        with open(args.latency_out, "w") as fh:
            fh.write("\t".join(cols) + "\n")
            fh.write("\t".join(str(v) for v in vals) + "\n")
        print(f"Latency written to    : {args.latency_out}")

    if args.phase_out:
        if not searcher.phase_times:
            print("No phase timings recorded; rebuild flashtrie with "
                  "-DTBS_PROFILE=ON to enable --phase-out.")
        else:
            names = ["expansion", "validation", "selection",
                     "grid_overhead", "kernel"]
            totals = [0.0] * len(names)
            requests = 0
            for summed, count in searcher.phase_times:
                for i, value in enumerate(summed):
                    totals[i] += value
                requests += count
            means = [t / requests if requests else 0.0 for t in totals]
            os.makedirs(
                os.path.dirname(os.path.abspath(args.phase_out)),
                exist_ok=True)
            with open(args.phase_out, "w") as fh:
                fh.write("beam\ttopk\tbatch_size\t"
                         + "\t".join(f"{n}_ms" for n in names) + "\n")
                fh.write(f"{args.beam_size}\t{args.topk_tokens}\t"
                         f"{args.batch_size}\t"
                         + "\t".join(f"{m:.4f}" for m in means) + "\n")
            print("Phase breakdown (ms/request): "
                  + "  ".join(f"{n}={m:.3f}" for n, m in zip(names, means)))
            print(f"Phases written to     : {args.phase_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
