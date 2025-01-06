import streamlit as st

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Header
st.write("<h3 align='center'>Explaining Medical Terms in Hausa Machine Translation</h3>", unsafe_allow_html=True)

st.write("""
Using SBERT and Text-To-Text Transfer Transformer for Explaining Medical Terms in Hausa Machine Translation.
""")


st.image("images/repo-cover.jpg")


st.write("""
### Problem Statement
<p align="justify">
    Khoong et. al. 2019 found that common medical discharge information was incorrectly translated by Google Translate 8% of the time for Spanish and 19% for Chinese and that 2% of those translations could cause clinically significant harm for Spanish and 8% for Chinese. Patil & Davies, 2014, found that only 45% of common medical phrases were correctly translated to two African languages. <br/> Coupled with identified problems, existing MT leaves the translation of medical terms to whatever the patient can infer resulting into incomplete information.
</p> 
""", unsafe_allow_html=True)


st.write("""
### Aim
<p align="justify">
    The aim of this study is to bridge the gap in English-Hausa medical translation by combining SBERT and Text-To-Text Transfer Transformer for the explanation of medical terms in English-Hausa machine translation.
</p> 
""", unsafe_allow_html=True)


translation_model_name = "xgboost-lover/t5-small-english-hausa-translation"

emt_df = pd.read_csv("emt_df_llama.csv")


def create_medical_term_embeddings(emt_df):
    """
    Create embeddings for medical terms using Sentence Transformers
    """
    # Initialize the SentenceTransformer model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create embeddings for medical terms
    medical_terms = emt_df['Medical Term'].tolist()
    medical_embeddings = embedding_model.encode(medical_terms)
    
    return embedding_model, medical_terms, medical_embeddings


def find_similar_medical_terms(input_text, embedding_model, medical_terms, medical_embeddings, threshold=0.7):
    """
    Find similar medical terms in input text by comparing embeddings
    """
    # Split input text into words and phrases (up to 3 words)
    words = input_text.split()
    phrases = []

    # Single words
    phrases.extend(words)

    # Two-word phrases
    for i in range(len(words)-1):
        phrases.append(f"{words[i]} {words[i+1]}")

    # Three-word phrases
    for i in range(len(words)-2):
        phrases.append(f"{words[i]} {words[i+1]} {words[i+2]}")

    # Create embeddings for input phrases
    phrase_embeddings = embedding_model.encode(phrases)

    # Find similar terms
    matched_terms = set()
    for phrase, phrase_emb in zip(phrases, phrase_embeddings):
        # Calculate similarity with all medical terms
        similarities = cosine_similarity([phrase_emb], medical_embeddings)[0]

        # Find matches above the threshold
        matches = np.where(similarities > threshold)[0]
        for match_idx in matches:
            matched_terms.add(medical_terms[match_idx])

    return list(matched_terms)


def load_translation_model(model_name):
    """
    Load translation model and tokenizer from the Hugging Face Hub
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def translate_text(input_text, model, tokenizer):
    """
    Translate input text using the loaded Hugging Face model
    """
    inputs = tokenizer("translate English to Hausa: " + input_text, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text


def provide_explanations(matched_terms, emt_df, hausa_translation_model, tokenizer):
    """
    Provide explanations for matched medical terms
    """
    explanations = []
    for term in matched_terms:
        # Find the English explanation for the term
        row = emt_df[emt_df['Medical Term'] == term]
        if not row.empty:
            english_explanation = row['llama_explanation'].values[0]
            
            # Translate the explanation to Hausa
            hausa_translation = translate_text(english_explanation, hausa_translation_model, tokenizer)
            explanations.append(f"{term}: {hausa_translation}")
    
    return explanations


def run_inference_pipeline(input_text, emt_df, translation_model_name):
    """
    End-to-end inference pipeline
    """
    # Load models
    translation_model, tokenizer = load_translation_model(translation_model_name)
    embedding_model, medical_terms, medical_embeddings = create_medical_term_embeddings(emt_df)

    # Detect medical terms
    matched_terms = find_similar_medical_terms(input_text, embedding_model, medical_terms, medical_embeddings)

    # Translate the input text
    translated_text = translate_text(input_text, translation_model, tokenizer)

    # Provide explanations for detected medical terms
    explanations = provide_explanations(matched_terms, emt_df, translation_model, tokenizer)

    # Output the results
    print("Original Text:")
    print(input_text)
    print("\nTranslated Text:")
    print(translated_text)
    print("\nMedical Explanation:")
    for explanation in explanations:
        print(explanation)

    return translated_text, explanations


st.write("""### Inference""")


# Input text
input_text = st.text_area("Enter text to translate to Hausa:")

# Button and spinner
if st.button("Translate"):
    if input_text:
        with st.spinner("Translating, please wait..."):
            translated_text, explanations = run_inference_pipeline(input_text, emt_df, translation_model_name)
        st.success("Translation complete!")
        st.write("#### == Original Text ====")
        st.write(input_text)
        st.write("#### == Translated Text ====")
        st.write(translated_text)
        st.write("#### == Relevant Medical Terms and Explanations ====")
        for explanation in explanations:
            st.write(explanation)
