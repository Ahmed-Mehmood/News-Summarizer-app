# import streamlit as st
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# import torch

# # Load model and tokenizer
# @st.cache_resource
# def load_model():
#     tokenizer = AutoTokenizer.from_pretrained("saved_model")
#     model = AutoModelForSeq2SeqLM.from_pretrained("saved_model")
#     return tokenizer, model

# tokenizer, model = load_model()

# st.title("📰 Text Summarizer")
# st.write("Enter an article below and get a concise summary!")

# # User input
# input_text = st.text_area("Paste your text here:", height=250)

# if st.button("Generate Summary"):
#     if input_text.strip() == "":
#         st.warning("Please enter some text!")
#     else:
#         # Tokenize input
#         inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
#         # Generate summary
#         summary_ids = model.generate(
#             inputs["input_ids"],
#             max_length=100,
#             min_length=30,
#             num_beams=4,
#             early_stopping=True
#         )
#         summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#         st.subheader("Summary:")
#         st.write(summary)


import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_REPO = "ahmedsoomro/news-summarizer-t5-small"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_REPO)
    return tokenizer, model

tokenizer, model = load_model()

st.title("News Summarizer App")

text = st.text_area("Paste your news article here")

if st.button("Summarize"):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=120)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    st.write("### Summary")
    st.write(summary)
