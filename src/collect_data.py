# src/collect_data.py
import os
import json
from typing import Dict, List

import wikipediaapi

TOPIC_PAGES: Dict[str, List[str]] = {
    "Physics": [
        "Physics",
        "Classical mechanics",
        "Quantum mechanics",
        "Thermodynamics",
        "Statistical mechanics",
        "Electromagnetism",
        "Classical electromagnetism",
        "Optics",
        "Special relativity",
        "General relativity",
        "Astrophysics",
        "Cosmology",
        "Nuclear physics",
        "Particle physics",
        "Quantum field theory",
        "Condensed matter physics",
        "Solid-state physics",
        "Fluid mechanics",
        "Plasma (physics)",
        "Acoustics"
    ],
    "Computer Science": [
        "Computer science",
        "Algorithm",
        "Data structure",
        "Computational complexity theory",
        "Operating system",
        "Computer network",
        "Internet",
        "Database",
        "Relational database",
        "Computer architecture",
        "Computer programming",
        "Programming language",
        "Software engineering",
        "Artificial intelligence",
        "Machine learning",
        "Deep learning",
        "Computer graphics",
        "Human–computer interaction",
        "Distributed computing",
        "Information security",
        "Cryptography"
    ],
    "Biology": [
        "Biology",
        "Cell (biology)",
        "DNA",
        "RNA",
        "Gene",
        "Genome",
        "Protein",
        "Enzyme",
        "Metabolism",
        "Photosynthesis",
        "Evolution",
        "Natural selection",
        "Genetics",
        "Mitosis",
        "Meiosis",
        "Immune system",
        "Nervous system",
        "Human anatomy",
        "Microorganism",
        "Bacteria",
        "Virus"
    ],
    "Economics": [
        "Economics",
        "Microeconomics",
        "Macroeconomics",
        "Supply and demand",
        "Market (economics)",
        "Price",
        "Inflation",
        "Deflation",
        "Gross domestic product",
        "Unemployment",
        "Monetary policy",
        "Fiscal policy",
        "Interest rate",
        "Exchange rate",
        "International trade",
        "Comparative advantage",
        "Economic growth",
        "Business cycle",
        "Public finance",
        "Game theory",
        "Behavioral economics",
        "Finance"
    ],
    "History": [
        "History",
        "Ancient history",
        "Middle Ages",
        "Early modern period",
        "Modern history",
        "Renaissance",
        "Industrial Revolution",
        "Roman Empire",
        "Byzantine Empire",
        "Ottoman Empire",
        "French Revolution",
        "American Revolution",
        "Russian Revolution",
        "American Civil War",
        "World War I",
        "World War II",
        "Cold War",
        "Colonialism",
        "Decolonization",
        "Vietnam War"
    ],
    "Other": [
        "Association football",
        "Basketball",
        "Tennis",
        "K-pop",
        "Hip hop music",
        "Pop music",
        "Rock music",
        "Jazz",
        "Painting",
        "Sculpture",
        "Cinema",
        "Film",
        "Television",
        "Video game",
        "Anime",
        "Manga",
        "Fashion",
        "Cuisine",
        "Tourism",
        "Social media",
        "YouTube"
    ]
}


# 2. Init Wikipedia API client
wiki = wikipediaapi.Wikipedia(
    user_agent="NLP-Topic-Summarizer-StudentProject/1.0 (mailto:youremail@example.com)",
    language="en",
    extract_format=wikipediaapi.ExtractFormat.WIKI,
)

os.makedirs("data", exist_ok=True)
OUT_PATH = "../data/wiki_topic_dataset.jsonl"

def fetch_page_text(title: str) -> str:
    page = wiki.page(title)
    if not page.exists():
        print(f"[WARN] Page does not exist: {title}")
        return ""
    return page.text


def main():
    n_samples = 0
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for topic, titles in TOPIC_PAGES.items():
            for title in titles:
                print(f"[INFO] Fetching '{title}' for topic '{topic}'...")
                text = fetch_page_text(title)

                if not text or len(text) < 500:
                    print(f"[WARN] Skipping '{title}' (too short or empty)")
                    continue

                record = {
                    "title": title,
                    "topic": topic,
                    "text": text
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_samples += 1

    print(f"[DONE] Saved {n_samples} samples to {OUT_PATH}")


if __name__ == "__main__":
    main()
