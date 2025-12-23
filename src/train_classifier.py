# src/train_classifier.py
import json
from pathlib import Path
from typing import List, Tuple

import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

DATA_PATH = Path("../data/wiki_topic_dataset.jsonl")


def load_dataset() -> Tuple[List[str], List[str]]:
    texts, labels = [], []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
            labels.append(obj["topic"])
    return texts, labels


def main():
    texts, labels = load_dataset()
    print("Total samples:", len(texts))

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    # TF-IDF feature extraction
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        stop_words="english"
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Classifier
    clf = LogisticRegression(
        max_iter=2000,
        n_jobs=-1
    )
    clf.fit(X_train_vec, y_train)

    # Evaluation
    y_pred = clf.predict(X_test_vec)
    print("=== Classification report ===")
    print(classification_report(y_test, y_pred))

    # Save model and vectorizer
    Path("models").mkdir(exist_ok=True)
    joblib.dump(vectorizer, "../models/tfidf_vectorizer.joblib")
    joblib.dump(clf, "../models/topic_classifier_logreg.joblib")
    print("[DONE] Saved vectorizer and classifier to models/")


if __name__ == "__main__":
    main()
