import json
import gzip
from pathlib import Path
from scorer.composite import score_candidate

def precompute():
    path = Path("candidates.jsonl")
    if not path.exists():
        path = Path("candidates.jsonl.gz")
    if not path.exists():
        path = Path("candidates_sample.jsonl")
        
    print(f"Loading and scoring from {path}...")
    
    open_fn = gzip.open if path.suffix == ".gz" else open
    results = []
    eliminated_count = 0
    total_processed = 0
    
    with open_fn(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_processed += 1
            c = json.loads(line)
            scored = score_candidate(c)
            if scored is None:
                eliminated_count += 1
            else:
                results.append(scored)
                
    # Sort by raw score and candidate_id (standard sorting)
    results.sort(key=lambda x: (-x["raw_score"], x["candidate_id"]))
    
    # Scale display scores
    if results:
        max_raw = max((r["raw_score"] for r in results), default=1.0)
        scale = max(max_raw, 1.0)
        for r in results:
            r["display_score"] = r["raw_score"] / scale
        # Tie-breaker sort
        results.sort(key=lambda x: (-round(x["display_score"], 4), x["candidate_id"]))
        
    print(f"Total processed: {total_processed}, Scored: {len(results)}, Eliminated: {eliminated_count}")
    
    # Keep top 200 candidates
    top_200 = results[:200]
    
    # Save metadata along with candidates
    data = {
        "total_processed": total_processed,
        "eliminated_count": eliminated_count,
        "scored_count": len(results),
        "top_results": top_200
    }
    
    out_path = Path("precomputed_top_candidates.json")
    with open(out_path, "w", encoding="utf-8") as out_f:
        json.dump(data, out_f, indent=2, ensure_ascii=False)
        
    print(f"Successfully saved {len(top_200)} top candidates to {out_path}")

if __name__ == "__main__":
    precompute()
