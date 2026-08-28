#!/usr/bin/env python3
"""Beam-width / top-K sweep driver for FlashTrie.

Invokes ``run_trie_search.py`` once per configuration and collects every
result into a single latency summary TSV.

Two input modes
---------------
*Glob mode* (used for the paper's main sweep). Put a single ``*`` in
``--data-path`` and list the substitutions in ``--file-values``. With
``--onemap`` the i-th file is paired with the i-th configuration, and when
``--beam-topk`` is omitted the file value itself becomes both the beam width
and the top-K, so ``output_topk300.tsv`` runs at ``beam=topk=300``::

    python run_trie_search_BWsweep.py \\
        --trie-path data/trie.bin \\
        --data-path 'data/topk/output_topk*.tsv' \\
        --file-values 100 200 300 400 500 600 700 800 900 1000 \\
        --onemap \\
        --output-dir out/bw_sweep \\
        --batch-size 1

*Single-file mode*. Point ``--data-path`` at one wide file and enumerate the
configurations explicitly; ``--superset-topk`` slices each row down::

    python run_trie_search_BWsweep.py \\
        --trie-path data/trie.bin \\
        --data-path data/topk.tsv \\
        --beam-topk 100:8 200:8 400:8 600:8 800:8 1000:8 \\
        --superset-topk \\
        --output-dir out/bw_sweep

Without ``--onemap`` the driver runs the cartesian product of inputs and
configurations.
"""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "run_trie_search.py")


def parse_beam_topk(entries):
    """Parse ``beam[:topk]`` entries; topk defaults to the beam width."""
    pairs = []
    for entry in entries:
        text = entry.strip()
        if not text:
            continue
        if ":" in text:
            beam_s, topk_s = text.split(":", 1)
        else:
            beam_s = topk_s = text
        try:
            pairs.append((int(beam_s), int(topk_s)))
        except ValueError as exc:
            raise SystemExit(
                f"--beam-topk: could not parse '{entry}' as beam[:topk]"
            ) from exc
    if not pairs:
        raise SystemExit("--beam-topk did not yield any configurations")
    return pairs


def expand_inputs(data_path, file_values):
    """Expand a single ``*`` in data_path over file_values."""
    if "*" not in data_path:
        if file_values:
            raise SystemExit(
                "--file-values requires a '*' placeholder in --data-path")
        return [(data_path, None)]
    if data_path.count("*") != 1:
        raise SystemExit("--data-path must contain exactly one '*'")
    if not file_values:
        raise SystemExit("--data-path contains '*' but --file-values is empty")
    return [(data_path.replace("*", value), value) for value in file_values]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--trie-path", required=True,
                   help="Path to the FlashTrie index")
    p.add_argument("--data-path", required=True,
                   help="Top-K TSV, or a template containing one '*'")
    p.add_argument("--file-values", nargs="+", default=None,
                   help="Substitutions for the '*' in --data-path")
    p.add_argument("--output-dir", default=os.path.join(HERE, "out", "bw"),
                   help="Directory for per-configuration outputs")
    p.add_argument("--beam-topk", nargs="+", default=None,
                   help="Configurations as 'beam[:topk]'. Required unless "
                        "--onemap is set")
    p.add_argument("--onemap", action="store_true",
                   help="Pair inputs with configurations positionally "
                        "instead of taking their cartesian product")
    p.add_argument("--superset-topk", action="store_true",
                   help="Pass --superset-topk to each run")
    p.add_argument("--latency-report", default=None,
                   help="Collect every configuration into one summary TSV "
                        "(default: <output-dir>/latency_summary.tsv)")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip configurations whose latency file already "
                        "exists, so an interrupted sweep can be resumed")

    # Passed straight through to run_trie_search.py.
    p.add_argument("--token-logp-threshold", type=float, default=-10.0)
    p.add_argument("--sent-logp-threshold", type=float, default=-40.0)
    p.add_argument("--length-norm", type=float, default=5.0)
    p.add_argument("--early-exit", action="store_true")
    p.add_argument("--batch-size", type=int, default=1)
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
    inputs = expand_inputs(args.data_path, args.file_values)

    missing = [path for path, _ in inputs if not os.path.isfile(path)]
    if missing:
        raise SystemExit("Input files not found:\n  " + "\n  ".join(missing))

    runs = []
    if args.onemap:
        if args.beam_topk:
            pairs = parse_beam_topk(args.beam_topk)
            if len(pairs) != len(inputs):
                raise SystemExit(
                    f"--onemap needs one --beam-topk entry per input "
                    f"({len(inputs)} inputs, {len(pairs)} entries)")
            for (path, value), (beam, topk) in zip(inputs, pairs):
                runs.append((path, value, beam, topk))
        else:
            for path, value in inputs:
                if value is None:
                    raise SystemExit(
                        "--onemap without --beam-topk requires --file-values")
                try:
                    beam = int(value)
                except ValueError as exc:
                    raise SystemExit(
                        "--onemap without --beam-topk requires numeric "
                        f"--file-values, got '{value}'") from exc
                runs.append((path, value, beam, beam))
    else:
        if not args.beam_topk:
            raise SystemExit("--beam-topk is required unless --onemap is set")
        for path, value in inputs:
            for beam, topk in parse_beam_topk(args.beam_topk):
                runs.append((path, value, beam, topk))
    return runs


def main(argv=None):
    args = parse_args(argv)
    runs = build_runs(args)
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = args.latency_report or os.path.join(
        args.output_dir, "latency_summary.tsv")

    print("=" * 60)
    print("FlashTrie beam-width sweep")
    print("=" * 60)
    print(f"Trie        : {args.trie_path}")
    print(f"Output dir  : {args.output_dir}")
    print(f"Batch size  : {args.batch_size}")
    print(f"Total runs  : {len(runs)}")
    for path, _, beam, topk in runs:
        print(f"  {os.path.basename(path):<28} beam={beam:<6} topk={topk}")
    print("=" * 60)

    grand_start = time.time()
    summary_rows = []
    header = None
    failed = []

    for path, value, beam, topk in runs:
        stem = value if value is not None else os.path.splitext(
            os.path.basename(path))[0]
        tag = f"{stem}_bw{beam}_tk{topk}"
        output_path = os.path.join(args.output_dir, f"{tag}.tsv")
        latency_path = os.path.join(args.output_dir, f"latency_{tag}.tsv")
        phase_path = os.path.join(args.output_dir, f"phase_{tag}.tsv")

        print(f"\n--- {os.path.basename(path)} | beam={beam} topk={topk} ---")

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
                "--batch-size", str(args.batch_size),
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
                print(f"    FAILED: beam={beam} topk={topk}")
                failed.append((path, beam, topk))
                continue

        if os.path.isfile(latency_path):
            with open(latency_path) as fh:
                lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
            if len(lines) >= 2:
                if header is None:
                    header = "input\tfile_value\t" + lines[0]
                summary_rows.append(
                    f"{os.path.basename(path)}\t{value or ''}\t{lines[1]}")

    if header and summary_rows:
        with open(report_path, "w") as fh:
            fh.write(header + "\n")
            fh.write("\n".join(summary_rows) + "\n")
        print(f"\nLatency summary written to: {report_path}")

    print(f"\nSweep finished in {time.time() - grand_start:.1f}s "
          f"({len(summary_rows)}/{len(runs)} configurations succeeded)")
    if failed:
        print("Failed configurations:")
        for path, beam, topk in failed:
            print(f"  {os.path.basename(path)}  beam={beam}  topk={topk}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
