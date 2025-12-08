import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import html

# Page config
st.set_page_config(page_title="AI News Summarizer", page_icon="📰", layout="centered")

# Custom CSS for styling
st.markdown("""
<style>
    .title {
        text-align: center;
        font-size: 32px;
        font-weight: bold;
    }
    .sub {
        text-align: center;
        color: #555;
    }
    .summary-box {
        background: #f1f5f9;
        padding: 15px;
        border-radius: 8px;
        font-size: 16px;
        margin-top: 10px;
        color: #1f2937;
        line-height: 1.6;
        border-left: 4px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_name = "ahmedsoomro/news-summarizer-t5-small"
    # Use slow tokenizer to avoid fast tokenizer compatibility issues
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    return tokenizer, model

tokenizer, model = load_model()

st.markdown('<p class="title">📰 AI News Summarizer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub">Paste news text and get an instant AI summary.</p>', unsafe_allow_html=True)

text = st.text_area("✒ Enter Article Text:", height=250)

if st.button("Generate Summary"):
    if text.strip():
        st.info("⏳ Generating summary, please wait...")
        inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)

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

        st.subheader("📌 Summary:")
        # Escape HTML special characters and display the summary
        escaped_summary = html.escape(summary)
        st.markdown(f"<div class='summary-box'>{escaped_summary}</div>", unsafe_allow_html=True)
    else:
        st.warning("⚠ Please enter text to summarize.")
