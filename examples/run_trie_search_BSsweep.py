#!/usr/bin/env python3
"""Batch-size sweep driver for FlashTrie.

Runs each ``beam:topk`` configuration across a list of batch sizes and
collects the results into one latency summary TSV. Use this to measure how
throughput scales as more requests share a kernel launch.

*Single-file mode* (default). One wide TSV, explicit configurations::

    python run_trie_search_BSsweep.py \\
        --trie-path data/trie.bin \\
        --data-path data/topk.tsv \\
        --bw-topk-pairs 100:8,600:8,1000:8 \\
        --batch-sizes 1,2,4,8,16,32 \\
        --superset-topk \\
        --output-dir out/bs_sweep

*Directory mode*. A directory of per-top-K files named ``output_topk{K}.tsv``;
K is taken from the filename and used as both the beam width and the top-K,
matching the beam-width sweep's ``--onemap`` convention::

    python run_trie_search_BSsweep.py \\
        --trie-path data/trie.bin \\
        --data-path data/topk \\
        --topk-values 100,500,1000 \\
        --batch-sizes 1,4,16 \\
        --output-dir out/bs_sweep
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "run_trie_search.py")


def parse_int_list(text, flag):
    values = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            values.append(int(chunk))
        except ValueError as exc:
            raise SystemExit(
                f"{flag}: could not parse '{chunk}' as an integer") from exc
    if not values:
        raise SystemExit(f"{flag} did not yield any values")
    return values


def parse_pairs(text):
    """Parse a comma-separated list of ``beam[:topk]`` entries."""
    pairs = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        beam_s, _, topk_s = chunk.partition(":")
        topk_s = topk_s or beam_s
        try:
            pairs.append((int(beam_s), int(topk_s)))
        except ValueError as exc:
            raise SystemExit(
                f"--bw-topk-pairs: could not parse '{chunk}' as "
                "beam[:topk]") from exc
    if not pairs:
        raise SystemExit("--bw-topk-pairs did not yield any configurations")
    return pairs


def discover_directory_inputs(data_path, file_pattern, topk_values):
    """Find output_topk{K}.tsv files and read K out of each filename."""
    found = []
    for path in sorted(glob.glob(os.path.join(data_path, file_pattern))):
        match = re.search(r"topk(\d+)", os.path.basename(path))
        if not match:
            print(f"  skipping {os.path.basename(path)}: no topk<N> in name")
            continue
        topk = int(match.group(1))
        if topk_values and topk not in topk_values:
            continue
        found.append((path, topk))
    if not found:
        raise SystemExit(
            f"No files matching '{file_pattern}' under {data_path}")
    found.sort(key=lambda item: item[1])
    return found


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--trie-path", required=True,
                   help="Path to the FlashTrie index")
    p.add_argument("--data-path", required=True,
                   help="A single top-K TSV, or a directory of "
                        "output_topk{K}.tsv files")
    p.add_argument("--output-dir", default=os.path.join(HERE, "out", "bs"),
                   help="Directory for per-configuration outputs")
    p.add_argument("--batch-sizes", default="1,2,4,8,16,32",
                   help="Comma-separated batch sizes (default: "
                        "1,2,4,8,16,32)")
    p.add_argument("--bw-topk-pairs", default=None,
                   help="Comma-separated 'beam[:topk]' entries. Required "
                        "when --data-path is a single file")
    p.add_argument("--file-pattern", default="output_topk*.tsv",
                   help="Glob for directory mode "
                        "(default: output_topk*.tsv)")
    p.add_argument("--topk-values", default=None,
                   help="Comma-separated top-K values to keep in directory "
                        "mode")
    p.add_argument("--superset-topk", action="store_true",
                   help="Pass --superset-topk to each run")
    p.add_argument("--latency-report", default=None,
                   help="Collect every configuration into one summary TSV "
                        "(default: <output-dir>/latency_summary.tsv)")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip configurations whose latency file already "
                        "exists")

    # Passed straight through to run_trie_search.py.
    p.add_argument("--token-logp-threshold", type=float, default=-10.0)
    p.add_argument("--sent-logp-threshold", type=float, default=-40.0)
    p.add_argument("--length-norm", type=float, default=5.0)
    p.add_argument("--early-exit", action="store_true")
    p.add_argument("--max-seq-length", type=int, default=64)
    p.add_argument("--num-runs", type=int, default=11)
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--logging-steps", type=int, default=1000)
    p.add_argument("--cpu", action="store_true",
                   help="Run the CPU reference path instead of the GPU kernel")
    p.add_argument("--phase-timing", action="store_true",
                   help="Also request the per-phase GPU breakdown "
                        "(needs flashtrie built with -DTBS_PROFILE)")

    return p.parse_args(argv)


def build_runs(args):
    """Return (input_path, beam, topk, batch_size) tuples."""
    batch_sizes = parse_int_list(args.batch_sizes, "--batch-sizes")

    if os.path.isdir(args.data_path):
        if args.bw_topk_pairs:
            raise SystemExit(
                "--bw-topk-pairs applies to single-file mode; "
                "--data-path is a directory")
        keep = (parse_int_list(args.topk_values, "--topk-values")
                if args.topk_values else None)
        configs = [(path, topk, topk) for path, topk in
                   discover_directory_inputs(
                       args.data_path, args.file_pattern, keep)]
    else:
        if not os.path.isfile(args.data_path):
            raise SystemExit(f"Input file not found: {args.data_path}")
        if not args.bw_topk_pairs:
            raise SystemExit(
                "--bw-topk-pairs is required when --data-path is a file")
        configs = [(args.data_path, beam, topk)
                   for beam, topk in parse_pairs(args.bw_topk_pairs)]

    return [(path, beam, topk, bs)
            for path, beam, topk in configs
            for bs in batch_sizes]


def main(argv=None):
    args = parse_args(argv)
    runs = build_runs(args)
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = args.latency_report or os.path.join(
        args.output_dir, "latency_summary.tsv")

    print("=" * 60)
    print("FlashTrie batch-size sweep")
    print("=" * 60)
    print(f"Trie        : {args.trie_path}")
    print(f"Output dir  : {args.output_dir}")
    print(f"Total runs  : {len(runs)}")
    for path, beam, topk, bs in runs:
        print(f"  {os.path.basename(path):<28} beam={beam:<6} "
              f"topk={topk:<6} bs={bs}")
    print("=" * 60)

    grand_start = time.time()
    summary_rows = []
    header = None
    failed = []

    for path, beam, topk, bs in runs:
        tag = f"bw{beam}_tk{topk}_bs{bs}"
        output_path = os.path.join(args.output_dir, f"{tag}.tsv")
        latency_path = os.path.join(args.output_dir, f"latency_{tag}.tsv")
        phase_path = os.path.join(args.output_dir, f"phase_{tag}.tsv")

        print(f"\n--- beam={beam} topk={topk} batch={bs} ---")

        if args.skip_existing and os.path.isfile(latency_path):
            print("    (skipped: latency file already exists)")
        else:
            cmd = [
                sys.executable, RUNNER,
                "--trie-path", args.trie_path,
                "--data-path", path,
                "--output-path", output_path,
                "--latency-out", latency_path,
                "--beam-size", str(beam),
                "--topk-tokens", str(topk),
                "--batch-size", str(bs),
                "--max-seq-length", str(args.max_seq_length),
                "--token-logp-threshold", str(args.token_logp_threshold),
                "--sent-logp-threshold", str(args.sent_logp_threshold),
                "--length-norm", str(args.length_norm),
                "--num-runs", str(args.num_runs),
                "--logging-steps", str(args.logging_steps),
            ]
            if args.superset_topk:
                cmd.append("--superset-topk")
            if args.early_exit:
                cmd.append("--early-exit")
            if args.cpu:
                cmd.append("--cpu")
            if args.max_queries is not None:
                cmd += ["--max-queries", str(args.max_queries)]
            if args.phase_timing:
                cmd += ["--phase-out", phase_path]

            if subprocess.run(cmd).returncode != 0:
                print(f"    FAILED: beam={beam} topk={topk} batch={bs}")
                failed.append((beam, topk, bs))
                continue

        if os.path.isfile(latency_path):
            with open(latency_path) as fh:
                lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
            if len(lines) >= 2:
                if header is None:
                    header = "input\t" + lines[0]
                summary_rows.append(
                    f"{os.path.basename(path)}\t{lines[1]}")

    if header and summary_rows:
        with open(report_path, "w") as fh:
            fh.write(header + "\n")
            fh.write("\n".join(summary_rows) + "\n")
        print(f"\nLatency summary written to: {report_path}")

    print(f"\nSweep finished in {time.time() - grand_start:.1f}s "
          f"({len(summary_rows)}/{len(runs)} configurations succeeded)")
    if failed:
        print("Failed configurations:")
        for beam, topk, bs in failed:
            print(f"  beam={beam}  topk={topk}  batch={bs}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
