"""
Quick checks on public_tickets.jsonl before training:
class balance, length leakage, a TF-IDF baseline on the test set, and samples.

Usage: python scripts/check_training_data.py
"""
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

from app.config import PUBLIC_TICKETS_PATH, TICKET_CATEGORIES, TICKETS_PATH


def read(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    rows = read(PUBLIC_TICKETS_PATH)
    test = read(TICKETS_PATH)
    texts = [r["text"] for r in rows]
    labels = [r["category"] for r in rows]

    print("== 1. Balance and word counts (median [p10-p90])")
    groups = defaultdict(list)
    for r in rows:
        n = len(r["text"].split())
        groups[r["category"]].append(n)
        groups[(r["category"], r["source"])].append(n)

    def fmt(xs):
        q = np.percentile(xs, [10, 50, 90]).astype(int)
        return f"n={len(xs):<5} {q[1]:>3} [{q[0]}-{q[2]}]"

    for cat in TICKET_CATEGORIES:
        print(f"  {cat:<10} {'(all)':<11} {fmt(groups[cat])}")
        for key in sorted(k for k in groups if isinstance(k, tuple) and k[0] == cat):
            print(f"  {'':<10} {key[1]:<11} {fmt(groups[key])}")
    print(f"  {'fitness':<10} {'(test)':<11} {fmt([len(r['text'].split()) for r in test])}")

    conflicts = [t for t, c in Counter(t.lower() for t in texts).items() if c > 1]
    print(f"  duplicate texts: {len(conflicts)}")

    print(f"\n== 2. Length-only baseline (5-fold CV, chance = {1 / len(TICKET_CATEGORIES):.0%})")
    X_len = np.log1p([[len(t.split()), len(t)] for t in texts])
    acc = cross_val_score(LogisticRegression(max_iter=1000), X_len, labels, cv=5).mean()
    print(f"  accuracy from length alone: {acc:.1%}")

    print("\n== 3. TF-IDF + logistic regression")
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=2000, C=4)
    cv = cross_val_score(clf, X, labels, cv=5).mean()
    print(f"  public data 5-fold CV accuracy: {cv:.1%}")
    clf.fit(X, labels)
    preds = clf.predict(vec.transform([r["text"] for r in test]))
    print("  held-out fitness tickets:")
    print(classification_report([r["category"] for r in test], preds,
                                labels=TICKET_CATEGORIES, zero_division=0))
    for r, p in zip(test, preds):
        if p != r["category"]:
            print(f"  MISS {r['category']:<9} -> {p:<9} {r['text']}")

    print("\n== 4. Random samples")
    rng = random.Random(0)
    by_group = defaultdict(list)
    for r in rows:
        by_group[(r["category"], r["source"])].append(r["text"])
    for key in sorted(by_group):
        for t in rng.sample(by_group[key], 3):
            print(f"  {key[0]:<9} {key[1]:<10} | {t[:140]}")


if __name__ == "__main__":
    main()
