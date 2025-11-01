"""
Basic TF-IDF similarity: docs/reference/* vs. outputs/* (chat logs, commit messages, generated text).
Run:
  python scripts/check_copilot_usage.py --docs docs/reference --targets outputs
"""
import os
import argparse
import glob

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _SKLEARN_OK = True
except Exception as _e:  # pylint: disable=broad-except
    _SKLEARN_OK = False
    _SKLEARN_ERR = _e


def load_texts(path: str):
    texts = []
    names = []
    for fp in glob.glob(os.path.join(path, "**/*.*"), recursive=True):
        if os.path.isdir(fp):
            continue
        ext = os.path.splitext(fp)[1].lower()
        if ext in (".pdf",):
            # skip binary; prefer extracted txt
            continue
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                s = f.read()
        except Exception:
            s = ""
        if s.strip():
            texts.append(s)
            names.append(fp)
    return names, texts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--docs", default="docs/reference")
    p.add_argument("--targets", default="outputs")
    args = p.parse_args()

    if not _SKLEARN_OK:
        print("Error: scikit-learn is required for this script.")
        print("Import error:", _SKLEARN_ERR)
        return 1

    doc_names, docs = load_texts(args.docs)
    tgt_names, tgts = load_texts(args.targets)
    if not docs or not tgts:
        print("No docs or targets found. Ensure files exist under", args.docs, args.targets)
        return 0

    vec = TfidfVectorizer(max_features=5000, stop_words="english")
    A = vec.fit_transform(docs + tgts)
    D = A[: len(docs)]
    T = A[len(docs) :]
    sims = cosine_similarity(T, D)

    # for each target, print top doc matches
    for i, tname in enumerate(tgt_names):
        row = sims[i]
        top_idx = row.argsort()[::-1][:5]
        print(f"\nTarget: {tname}\nTop doc matches:")
        for j in top_idx:
            print(f"  {doc_names[j]}  sim={row[j]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
