# src/analyze.py
import re
from urllib.parse import urlparse
from requests.exceptions import HTTPError, RequestException
import joblib
import requests
from bs4 import BeautifulSoup


# ===== Helpers =====

def is_url(s: str) -> bool:
    try:
        r = urlparse(s)
        return all([r.scheme, r.netloc])
    except ValueError:
        return False

USER_AGENT = "NLP-Topic-Classifier-StudentProject/1.0 (mailto:youremail@example.com)"

def fetch_text_from_url(url: str) -> str:
    try:
        resp = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": USER_AGENT},
        )
        resp.raise_for_status()
    except HTTPError as e:
        print(f"[ERROR] HTTP error while fetching {url}: {e}")
        return ""
    except RequestException as e:
        print(f"[ERROR] Request error while fetching {url}: {e}")
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")
    paragraphs = soup.find_all("p")
    text = "\n".join(p.get_text(" ", strip=True) for p in paragraphs)
    text = re.sub(r"\s+", " ", text)
    return text.strip()



vectorizer = joblib.load("../models/tfidf_vectorizer.joblib")
clf = joblib.load("../models/topic_classifier_logreg.joblib")

def classify_topic(text: str) -> dict:
    X = vectorizer.transform([text])
    probs = clf.predict_proba(X)[0]  # shape (n_classes,)

    top_idx = probs.argmax()
    top_label = clf.classes_[top_idx]
    top_prob = float(probs[top_idx])

    all_probs = {cls: float(p) for cls, p in zip(clf.classes_, probs)}

    return {
        "label": top_label,      # Always the highest probability class
        "top_label": top_label,
        "top_prob": top_prob,
        "all_probs": all_probs,
    }


def analyze(input_str: str):
    if is_url(input_str):
        text = fetch_text_from_url(input_str)
        source_type = "url"
    else:
        text = input_str
        source_type = "text"

    if len(text) < 200:
        raise ValueError("Text too short for classification.")

    topic_info = classify_topic(text)

    return {
        "source_type": source_type,
        "topic": topic_info["label"],
        "topic_details": topic_info,
    }

if __name__ == "__main__":
    example = "https://en.wikipedia.org/wiki/Mutation"
    result = analyze(example)
    print("Source type:", result["source_type"])
    print("Predicted topic:", result["topic"])
    print("Topic details:", result["topic_details"])