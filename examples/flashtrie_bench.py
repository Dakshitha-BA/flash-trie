"""Benchmarking helpers layered over the compiled ``flashtrie`` extension.

This module deliberately exposes only the API that the C++/CUDA sources in
this repository actually export:

``Trie``
    ``load`` / ``mmap`` / ``build`` / ``save`` / ``clear``,
    ``num_keys`` / ``num_nodes`` / ``num_tries`` / ``total_size``,
    ``init_tbs_gpu``
``tbs``
    Single-request constrained beam search (CPU or GPU).
``bs_tbs_gpu``
    Batched GPU constrained beam search.

Everything the example scripts need is built on top of these.
"""

import json
import math
import os
import sys
import time

import numpy as np

__all__ = [
    "import_flashtrie",
    "FlashTrieSearcher",
    "read_topk_tsv",
    "iter_batches",
    "summarize_latency",
]


def import_flashtrie():
    """Import the compiled ``flashtrie`` extension.

    Searches ``FLASHTRIE_BUILD_DIR`` if set, otherwise ``<repo>/build/tools``.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build_dir = os.environ.get(
        "FLASHTRIE_BUILD_DIR", os.path.join(repo_root, "build", "tools"))
    if build_dir and build_dir not in sys.path:
        sys.path.insert(0, build_dir)
    try:
        import flashtrie
    except ImportError as exc:
        raise SystemExit(
            f"Could not import the 'flashtrie' extension module ({exc}).\n"
            f"Looked in: {build_dir}\n"
            "Build it first:\n"
            "    cmake -B build -DCMAKE_BUILD_TYPE=Release\n"
            "    cmake --build build -j\n"
            "or point FLASHTRIE_BUILD_DIR at the directory containing "
            "flashtrie*.so.") from exc
    return flashtrie


def read_topk_tsv(path, max_queries=None):
    """Read a ``query<TAB>json`` top-K proposal file.

    The JSON object per line holds ``probs`` and ``indices``, both ``(T, K)``,
    and an optional ``lcid`` prefix token. Yields ``(query, record)`` pairs.
    """
    with open(path, "r") as fh:
        for i, line in enumerate(fh):
            if max_queries is not None and i >= max_queries:
                break
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                raise ValueError(
                    f"{path}:{i + 1}: expected 'query<TAB>json', got "
                    f"{len(parts)} column(s)")
            yield parts[0], json.loads(parts[1])


def iter_batches(items, batch_size):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


class FlashTrieSearcher:
    """Constrained beam search over a FlashTrie index.

    Parameters mirror the decoding knobs in the paper: ``beam_size`` is the
    beam width B, ``topk_tokens`` is the number of proposals K considered per
    step, and the two thresholds prune proposals and hypotheses respectively.
    """

    def __init__(self, trie_path, beam_size=100, topk_tokens=8,
                 batch_size=4, max_seq_length=64,
                 token_logp_threshold=-10.0, sent_logp_threshold=-40.0,
                 length_norm=5.0, early_exit=False, use_gpu=True,
                 superset_topk=False, trie_start_token=None, verbose=True):
        self.ab = import_flashtrie()
        self.beam_size = beam_size
        self.topk_tokens = topk_tokens
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.token_logp_threshold = token_logp_threshold
        self.sent_logp_threshold = sent_logp_threshold
        self.length_norm = length_norm
        self.early_exit = early_exit
        self.use_gpu = use_gpu
        self.superset_topk = superset_topk
        # None      -> no forced first step (the trie keys are raw token
        #              sequences; this is the default)
        # "lcid"    -> take the forced token from each record's "lcid" field
        # <int>     -> force this token at step 0 for every request
        self.trie_start_token = trie_start_token

        # (batch_wall_ms, num_requests) per call
        self.batch_times = []
        # (summed_phase_vector, num_requests) per call; nonzero only when the
        # extension is built with -DTBS_PROFILE
        self.phase_times = []

        if verbose:
            print(f"Loading trie from {trie_path} ...")
        start = time.time()
        self.trie = self.ab.Trie()
        self.trie.load(trie_path)
        if verbose:
            print(f"  loaded in {time.time() - start:.2f}s")
            print(f"  #keys:  {self.trie.num_keys()}")
            print(f"  #nodes: {self.trie.num_nodes()}")
            print(f"  size:   {self.trie.total_size() / (1 << 20):.2f} MiB")

        if self.use_gpu:
            # Size the persistent GPU workspace for the widest call this run
            # can make. max_seq_length covers the forced LCID prefix step.
            self.trie.init_tbs_gpu(
                self.batch_size,
                self.topk_tokens,
                max_num_out_beams=self.beam_size,
                max_num_inter_beams=self.beam_size * self.max_seq_length,
            )

    def _prepare(self, record):
        """Turn one JSON record into (beams, topk_id, topk_logp) lists."""
        topk_logp = np.asarray(record["probs"], dtype=np.single)
        topk_id = np.asarray(record["indices"], dtype=np.uintc)

        if self.trie_start_token == "lcid":
            lcid = record.get("lcid", None)
        else:
            lcid = self.trie_start_token

        if topk_logp.shape != topk_id.shape:
            raise ValueError(
                f"probs {topk_logp.shape} and indices {topk_id.shape} "
                "must have the same shape")

        # A single wide file can serve every sweep point. Select by
        # probability rather than by column position, because proposal files
        # are stored token-ID-sorted, not probability-sorted.
        if self.superset_topk and topk_id.shape[1] > self.topk_tokens:
            keep = np.argpartition(
                -topk_logp, self.topk_tokens - 1, axis=1)[:, : self.topk_tokens]
            topk_logp = np.take_along_axis(topk_logp, keep, axis=1)
            topk_id = np.take_along_axis(topk_id, keep, axis=1)
        if topk_id.shape[1] != self.topk_tokens:
            raise ValueError(
                f"data has K={topk_id.shape[1]} but --topk-tokens="
                f"{self.topk_tokens}; pass --superset-topk to slice a wider "
                "file down")

        # The kernel's binary child search requires token-ID-sorted rows.
        order = np.argsort(topk_id, axis=-1)
        topk_id = np.take_along_axis(topk_id, order, axis=-1)
        topk_logp = np.take_along_axis(topk_logp, order, axis=-1)

        if lcid is not None:
            # Forced first step: only the prefix token is reachable, so give
            # it probability 1 (log 0) and mask every other column. Filler
            # tokens must stay distinct from the prefix and the row must stay
            # sorted, because the kernel binary-searches each row.
            filler = [t for t in range(self.topk_tokens + 1) if t != lcid]
            prefix_ids = np.array(
                [lcid] + filler[: self.topk_tokens - 1], dtype=np.uintc)
            prefix_logp = np.full(
                [self.topk_tokens], -math.inf, dtype=np.single)
            prefix_logp[0] = 0.0
            prefix_order = np.argsort(prefix_ids)
            prefix_ids = prefix_ids[prefix_order]
            prefix_logp = prefix_logp[prefix_order]
            topk_logp = np.concatenate([prefix_logp[None, :], topk_logp], 0)
            topk_id = np.concatenate([prefix_ids[None, :], topk_id], 0)

        beams = np.full([topk_id.shape[0]], self.beam_size, dtype=np.uintc)
        if lcid is not None:
            beams[0] = 1

        return (beams.tolist(), topk_id.tolist(), topk_logp.tolist(),
                lcid is not None)

    def search(self, records):
        """Run constrained beam search over a batch of JSON records.

        Returns ``(batch_tokens, batch_scores)``. Wall time and, when the
        extension is built with ``-DTBS_PROFILE``, the per-phase GPU
        breakdown are accumulated on the instance.
        """
        batch_beams, batch_id, batch_logp, had_lcid = [], [], [], []
        for record in records:
            beams, ids, logps, lcid_used = self._prepare(record)
            batch_beams.append(beams)
            batch_id.append(ids)
            batch_logp.append(logps)
            had_lcid.append(lcid_used)

        if self.use_gpu:
            out_sent, out_logp, wall_ms, phases = self.ab.bs_tbs_gpu(
                trie=self.trie,
                batch_size=len(batch_id),
                batch_beams=batch_beams,
                batch_topk_id=batch_id,
                batch_topk_logp=batch_logp,
                token_logp_threshold=self.token_logp_threshold,
                sents_logp_threshold=self.sent_logp_threshold,
                length_norm=self.length_norm,
                early_exit=self.early_exit,
            )
        else:
            # The CPU reference path is single-request; loop to form a batch.
            out_sent, out_logp, phases = [], [], []
            start = time.time()
            for beams, ids, logps in zip(batch_beams, batch_id, batch_logp):
                sent, logp, _ = self.ab.tbs(
                    trie=self.trie,
                    beams=beams,
                    topk_id=ids,
                    topk_logp=logps,
                    token_logp_threshold=self.token_logp_threshold,
                    sents_logp_threshold=self.sent_logp_threshold,
                    length_norm=self.length_norm,
                    early_exit=self.early_exit,
                    use_gpu=False,
                )
                out_sent.append(sent)
                out_logp.append(logp)
            wall_ms = (time.time() - start) * 1000.0

        self.batch_times.append((wall_ms, len(batch_id)))
        if phases and any(any(p) for p in phases):
            summed = [sum(p[k] for p in phases) for k in range(len(phases[0]))]
            self.phase_times.append((summed, len(phases)))

        # Strip the forced prefix step back off the emitted sequences.
        for i, used in enumerate(had_lcid):
            if used:
                out_sent[i] = [sent[1:] for sent in out_sent[i]]

        return out_sent, [np.asarray(s) for s in out_logp]

    def reset_timers(self):
        self.batch_times = []
        self.phase_times = []


def summarize_latency(per_run_batch_ms, percentiles=(50, 90, 95, 99)):
    """Aggregate per-run batch latencies into mean +/- std statistics.

    ``per_run_batch_ms`` is a list (one entry per measured run) of lists of
    per-batch wall times in milliseconds. Statistics are computed within each
    run and then averaged across runs, so the reported deviation is
    run-to-run variance rather than within-run query variance.
    """
    run_means = [float(np.mean(r)) for r in per_run_batch_ms if len(r)]
    stats = {
        "num_measured_runs": len(run_means),
        "mean_ms": float(np.mean(run_means)) if run_means else 0.0,
        "std_ms": float(np.std(run_means, ddof=1)) if len(run_means) > 1
                  else 0.0,
    }
    for perc in percentiles:
        vals = [float(np.percentile(r, perc))
                for r in per_run_batch_ms if len(r)]
        stats[f"p{perc}_ms"] = float(np.mean(vals)) if vals else 0.0
        stats[f"std_p{perc}_ms"] = (
            float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
    return stats
