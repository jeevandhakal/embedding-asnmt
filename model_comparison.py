import json
import argparse
import numpy as np
from pathlib import Path

def load_embeddings(path: Path) -> dict[str, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: np.asarray(v, dtype=np.float32) for k, v in data.items()}

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def rank_list(emb: dict[str, np.ndarray], me: str) -> list[str]:
    if me not in emb:
        raise KeyError(f"'{me}' not found in embeddings keys.")
    me_vec = emb[me]
    scores = []
    for name, vec in emb.items():
        if name == me:
            continue
        scores.append((name, cosine_sim(me_vec, vec)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [n for n, _ in scores]

def spearman_rho(order1: list[str], order2: list[str]) -> float:
    common = [x for x in order1 if x in set(order2)]
    n = len(common)
    if n < 2:
        return float("nan")
    pos1 = {name: i + 1 for i, name in enumerate(order1)}
    pos2 = {name: i + 1 for i, name in enumerate(order2)}
    d2_sum = 0.0
    for name in common:
        d = pos1[name] - pos2[name]
        d2_sum += d * d
    return 1.0 - (6.0 * d2_sum) / (n * (n * n - 1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb1", required=True)
    ap.add_argument("--emb2", required=True)
    ap.add_argument("--me", required=True)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    e1 = load_embeddings(Path(args.emb1))
    e2 = load_embeddings(Path(args.emb2))

    o1 = rank_list(e1, args.me)
    o2 = rank_list(e2, args.me)

    rho = spearman_rho(o1, o2)

    print("=== Embedding Sensitivity Tests ===")
    print(f"Me: {args.me}")
    print(f"Spearman rho: {rho:.4f}\n")

    topk = max(1, args.topk)
    print(f"Top-{topk} model1:", o1[:topk])
    print(f"Top-{topk} model2:", o2[:topk])

    overlap = [x for x in o1[:topk] if x in set(o2[:topk])]
    print(f"Top-{topk} overlap ({len(overlap)}):", overlap)

if __name__ == "__main__":
    main()
