import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

st.title("TinyLlama Chat UI")

# ===== TinyLlama Setup =====
@st.cache_resource
def load_model():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,   # GPU ke liye, CPU-only me float32
        device_map="auto"
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

generator = load_model()
# ===========================

# User Input
user_input = st.text_area("Enter your prompt:")

if st.button("Generate"):
    if user_input.strip() != "":
        with st.spinner("Generating response..."):
            output = generator(user_input, max_length=200, do_sample=True, top_k=10)
            st.success("Response:")
            st.write(output[0]['generated_text'])
