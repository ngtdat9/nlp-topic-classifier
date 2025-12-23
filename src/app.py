# src/app.py
import streamlit as st
import pandas as pd
from analyze import analyze  # Importing your existing function

st.set_page_config(page_title="Topic Classifier", page_icon="📚")

# --- UI Header ---
st.title("📚 NLP Topic Classifier")
st.markdown("""
This tool classifies text or URLs into 6 categories: 
**Physics, CS, Biology, Economics, History, or Other.**
""")

# --- Input Area ---
st.write("### Input")
input_text = st.text_area(
    "Enter a URL or paste a paragraph of text:",
    height=150,
    placeholder="paste text or URL here..."
)

# --- Analysis Logic ---
if st.button("Analyze Topic", type="primary"):
    if not input_text:
        st.warning("Please enter some text or a URL first.")
    else:
        try:
            with st.spinner("Analyzing..."):
                result = analyze(input_text)

                # Extract data
                topic = result["topic"]
                probs = result["topic_details"]["all_probs"]
                source = result["source_type"]

            # --- Display Results ---
            st.success("Analysis Complete!")

            # 1. Main Topic Badge
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric(label="Predicted Topic", value=topic)
                st.caption(f"Source: {source.upper()}")

            # 2. Probability Chart
            with col2:
                st.subheader("Confidence Scores")
                # Convert dict to DataFrame for a nice bar chart
                df = pd.DataFrame(list(probs.items()), columns=["Topic", "Probability"])
                df = df.sort_values(by="Probability", ascending=True)  # Sort for chart
                st.bar_chart(df.set_index("Topic"), color="#4CAF50")

        except ValueError as e:
            st.error(f"Error: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

# --- Sidebar info ---
with st.sidebar:
    st.header("About")
    st.info("This project uses TF-IDF and Logistic Regression to classify academic topics.")