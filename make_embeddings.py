import argparse
import json
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to classmates CSV")
    ap.add_argument("--model", required=True, help="Model name (SentenceTransformer)")
    ap.add_argument("--out", required=True, help="Output JSON embeddings file")
    args = ap.parse_args()

    df = pd.read_csv(Path(args.csv))

    # Expected columns in your CSV: Name, Description
    df["Name"] = df["Name"].astype(str).str.strip()
    df["Description"] = df["Description"].astype(str).fillna("").str.strip()

    texts = df["Description"].tolist()

    model = SentenceTransformer(args.model)
    vectors = model.encode(texts, normalize_embeddings=True)

    embeddings = {name: vec.tolist() for name, vec in zip(df["Name"], vectors)}
    Path(args.out).write_text(json.dumps(embeddings, ensure_ascii=False), encoding="utf-8")

    print(f"Saved {len(embeddings)} embeddings to {args.out} using model={args.model}")


if __name__ == "__main__":
    main()
