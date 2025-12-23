# 📚 NLP Topic Classifier

A Machine Learning project that classifies text or websites into academic categories using Natural Language Processing (NLP).

This tool can analyze raw text or scrape content from a given URL to determine its primary topic (Physics, Computer Science, Biology, Economics, History, or Other) with a confidence score.

## 🚀 Features

* **Text Classification:** Classifies input text into 6 distinct categories.
* **URL Support:** Automatically scrapes and processes text from valid URLs (e.g., Wikipedia, news sites).
* **Confidence Visualization:** Displays the probability distribution for all potential topics.
* **Web UI:** Includes a user-friendly interface built with Streamlit.

## 🛠️ Tech Stack

* **Language:** Python 3.x
* **Machine Learning:** Scikit-Learn (Logistic Regression, TF-IDF)
* **NLP:** TfidfVectorizer (N-grams: 1,2)
* **Web Scraping:** BeautifulSoup4, Requests
* **Dataset API:** Wikipedia-API
* **UI:** Streamlit

## 📂 Project Structure

```text
├── data/
│   └── wiki_topic_dataset.jsonl       # Raw dataset collected from Wikipedia
├── models/
│   ├── tfidf_vectorizer.joblib        # Saved TF-IDF vocabulary
│   └── topic_classifier_logreg.joblib # Trained Logistic Regression model
├── src/
│   ├── collect_data.py                # Script to download Wikipedia articles
│   ├── train_classifier.py            # Script to train and save the model
│   ├── analyze.py                     # Core inference logic
│   └── app.py                         # Streamlit Web UI
└── README.md
