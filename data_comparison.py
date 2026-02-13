
import json
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

def load_embeddings(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    # Define paths
    old_path = Path("embeddings/minilm.json")
    new_path = Path("embeddings/modified_minilm.json")

    # Load embeddings
    if not old_path.exists():
        print(f"Error: {old_path} not found.")
        return
    if not new_path.exists():
        print(f"Error: {new_path} not found.")
        return

    old_emb = load_embeddings(old_path)
    new_emb = load_embeddings(new_path)

    print(f"{'Name':<30} | {'Cosine Similarity':<20}")
    print("-" * 55)

    results = []

    # Iterate through names present in both files
    for name in old_emb:
        if name in new_emb:
            vec1 = np.array(old_emb[name]).reshape(1, -1)
            vec2 = np.array(new_emb[name]).reshape(1, -1)
            
            # Compute Cosine Similarity
            sim = cosine_similarity(vec1, vec2)[0][0]
            results.append((name, sim))
        else:
            # Name only in old
            pass
            
    # Sort results by similarity (ascending) to easily spot the modified ones
    results.sort(key=lambda x: x[1])

    for name, sim in results:
        # Highlight if similarity is significantly less than 1.0 (indicating modification)
        prefix = ">> " if sim < 0.9999 else "   " 
        print(f"{prefix}{name:<27} | {sim:.6f}")

if __name__ == "__main__":
    main()
