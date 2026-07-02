import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

from tqdm import tqdm

from scorer.composite import reasoning_string, score_candidate


def load_candidates(path: str):
    p = Path(path)
    open_fn = gzip.open if p.suffix == ".gz" else open

    with open_fn(p, "rt", encoding="utf-8") as f:
        raw = f.read().strip()

    if raw.startswith("["):
        yield from json.loads(raw)
    else:
        for line in raw.splitlines():
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(
        description="India Runs — Intelligent Candidate Ranker"
    )
    parser.add_argument(
        "--candidates",
        required=True,
        help="Path to candidates.jsonl (or .json array, or .gz)",
    )
    parser.add_argument("--out", required=True, help="Output CSV path for submission")
    parser.add_argument(
        "--top",
        type=int,
        default=100,
        help="Number of top candidates to output (default: 100)",
    )
    args = parser.parse_args()

    results = []
    eliminated = 0
    elimination_reasons = {}

    print(f"\nLoading candidates from: {args.candidates}")
    print("Running three-pass evidence ranker...\n")

    for cand in tqdm(load_candidates(args.candidates), desc="Scoring", unit="cand"):
        r = score_candidate(cand)
        if r is None:
            eliminated += 1
        else:
            results.append(r)

    print(f"\n{'─' * 50}")
    print(f"Total processed : {eliminated + len(results):,}")
    print(f"Eliminated      : {eliminated:,}")
    print(f"Scored          : {len(results):,}")
    print(f"{'─' * 50}")

    if len(results) < args.top:
        print(
            f"\nWARNING: only {len(results)} candidates survived filters. "
            f"Need {args.top}. Loosen Pass 1 thresholds if this is the full dataset.",
            file=sys.stderr,
        )

    results.sort(key=lambda x: (-x["raw_score"], x["candidate_id"]))
    top = results[: args.top]

    max_raw = max((r["raw_score"] for r in top), default=1.0)
    scale = max(max_raw, 1.0)
    for r in top:
        r["display_score"] = r["raw_score"] / scale

    # Sort again using display scores rounded to 4 decimals to ensure tie-breaks are strictly ascending by candidate_id
    top.sort(key=lambda x: (-round(x["display_score"], 4), x["candidate_id"]))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, r in enumerate(top, 1):
            writer.writerow(
                [
                    r["candidate_id"],
                    rank,
                    f"{r['display_score']:.4f}",
                    reasoning_string(r),
                ]
            )

    print(f"\n✔  Top {len(top)} written to: {out_path}")

    print("\n── Top 10 Preview ──────────────────────────────────────────────")
    print(f"{'Rank':<5} {'ID':<15} {'Score':<7} {'Title':<35} {'YoE':<5} {'Mult':<5}")
    print("─" * 80)
    for r in top[:10]:
        p = r["candidate"]["profile"]
        dim = r["dimensions"]
        print(
            f"{top.index(r) + 1:<5} "
            f"{r['candidate_id']:<15} "
            f"{r['display_score']:<7.4f} "
            f"{str(p.get('current_title', ''))[:34]:<35} "
            f"{p.get('years_of_experience', 0):<5.1f} "
            f"{r['multiplier']:<5.2f}"
        )
    print()

    scores = [round(r["display_score"], 4) for r in top]
    violations = sum(1 for i in range(1, len(scores)) if scores[i] > scores[i - 1])
    if violations:
        print(
            f"WARNING: {violations} score ordering violations found. "
            f"Run validate_submission.py before uploading.",
            file=sys.stderr,
        )
    else:
        print("Score ordering check: PASSED (non-increasing)")

    print(f"\nNext step: python validate_submission.py {out_path}")


if __name__ == "__main__":
    main()
