import sys
import types
import torch
import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration, BartTokenizer, BartForConditionalGeneration

# ==== Workaround for torch.classes inspection issue ====
torch.classes = types.SimpleNamespace()

# ==== Load T5 ====
@st.cache_resource
def load_t5_model():
    t5_path = r"E:\MTechCSE\Study\Sem2\NLP\Assignment\Assignment_3\Final\Model\T5\t5_clan"
    t5_tokenizer = T5Tokenizer.from_pretrained(t5_path)
    t5_model = T5ForConditionalGeneration.from_pretrained(t5_path)
    t5_model.eval()
    return t5_model, t5_tokenizer

# ==== Load BART ====
@st.cache_resource
def load_bart_model():
    bart_path = r"E:\MTechCSE\Study\Sem2\NLP\Assignment\Assignment_3\Final\Model\BART\bart_model_output\final"
    bart_tokenizer = BartTokenizer.from_pretrained(bart_path)
    bart_model = BartForConditionalGeneration.from_pretrained(bart_path)
    bart_model.eval()
    return bart_model, bart_tokenizer

# ==== Inference Functions ====
def generate_t5(text, model, tokenizer):
    input_text = "normalize: " + text
    encoding = tokenizer(
        input_text, return_tensors="pt", truncation=True, padding=True, max_length=256
    )
    with torch.no_grad():
        outputs = model.generate(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            max_length=128,
            num_beams=4,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_bart(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=128,
            num_beams=4,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ==== Streamlit App Layout ====
st.set_page_config(page_title="Claim Normalization", layout="centered")
st.title("🧠 Social Media Claim Normalization")

st.markdown("Enter a messy social media post below, and select the model(s) to use for normalization.")

# Text input
user_input = st.text_area("✍️ Messy Claim:", height=180)

# Model selection
selected_models = st.multiselect(
    "Select model(s) to generate normalized claim:",
    options=["T5", "BART"],
    default=["T5"]
)

# Submit button
if st.button("Generate Normalized Claim"):
    if not user_input.strip():
        st.warning("Please enter a messy claim.")
    elif not selected_models:
        st.warning("Please select at least one model.")
    else:
        with st.spinner("Generating..."):
            if "T5" in selected_models:
                t5_model, t5_tokenizer = load_t5_model()
                try:
                    t5_output = generate_t5(user_input, t5_model, t5_tokenizer)
                    st.success("T5 Output:")
                    st.write(t5_output)
                except Exception as e:
                    st.error(f"T5 model failed: {e}")

            if "BART" in selected_models:
                bart_model, bart_tokenizer = load_bart_model()
                try:
                    bart_output = generate_bart(user_input, bart_model, bart_tokenizer)
                    st.success("BART Output:")
                    st.write(bart_output)
                except Exception as e:
                    st.error(f"BART model failed: {e}")
