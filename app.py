import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_REPO = "ahmedsoomro/news-summarizer-t5-small"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_REPO, low_cpu_mem_usage=True)
    return tokenizer, model

tokenizer, model = load_model()

st.title("📰 News Text Summarizer")
st.write("Paste any news/article below and generate an AI summary.")

# User input box
text = st.text_area("Enter text here:", height=250)

if st.button("Generate Summary"):
    if not text.strip():
        st.warning("Please enter some text!")
    else:
        with st.spinner("Generating summary..."):

            # Prefix required for T5 models
            input_text = "summarize: " + text

            # Tokenize input
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)

            # Inference without gradients (reduces memory usage)
            with torch.no_grad():
                summary_ids = model.generate(
                    inputs["input_ids"],
                    max_length=120,
                    min_length=30,
                    num_beams=4,
                    length_penalty=2.0,
                    early_stopping=True
                )

            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        st.subheader("Summary:")
        st.write(summary)
