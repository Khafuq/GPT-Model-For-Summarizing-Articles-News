import re
import streamlit as st 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))


@st.cache_data()
def load_model():
    model_name = "csebuetnlp/mT5_m2o_arabic_crossSum"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def generate_summary(article_text, tokenizer, model):
    input_ids = tokenizer(
        [WHITESPACE_HANDLER(article_text)],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    )["input_ids"]

    output_ids = model.generate(
        input_ids=input_ids,
        max_length=84,
        no_repeat_ngram_size=2,
        num_beams=4
    )[0]

    summary = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return summary

st.set_page_config(layout="wide", page_title="Tajreed App", page_icon="📝", initial_sidebar_state="expanded")

def add_logo():
    st.sidebar.image("/Users/khfuq/Desktop/Deployment./FullLogo_Transparent (1) (1).png", use_column_width=True)

add_logo()

def main():
    st.title("TAJREED")
    st.markdown('##### **Summarizing Arabic Articles**')
    


    article_text = st.text_area("Enter text to Summarize:")

    if st.button("Summarize"):
        if len(article_text.strip()) == 0:
            st.warning("Please enter some text to summarize.")
        else:
            tokenizer, model = load_model()
            summary = generate_summary(article_text, tokenizer, model)
            st.subheader("Summary:")
            st.write(summary)

if __name__ == "__main__":
    main()


base="dark"
primaryColor="#7bd0ea"
backgroundColor="#29174a"
secondaryBackgroundColor="#3a2f5c"
font="serif"
