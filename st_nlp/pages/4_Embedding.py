import streamlit as st
import plotly_express as px
from cache_func import load_w2v, load_glove, get_similarity

w2v = load_w2v()
glove = load_glove()

st.title("Embeddings")
st.subheader("Word2Vec Training")

st.write("We used the gensim library to train a Word2Vec on our tokenized description tokens.")

st.subheader("GloVe Model Augmentation")

st.write("We used a pretrained GloVe model to add the weights to our trained Word2Vec to better it.")

st.subheader("Test the Similarity of the Two Models for Comparison")

request1 = st.text_input('token')
if st.button("Get similarity for Word2Vec"):
    col1, col2 = st.columns(2)
    with col1:
        try:
            st.write(get_similarity(w2v, request1))
        except:
            st.write("An error has occured, the word is not in the model")
    with col2:
        try:
            st.write(get_similarity(glove, request1))
        except:
            st.write("An error has occured, the word is not in the model")

st.subheader("Word2Vec Visualisation")

st.subheader("Augmented Model Visualisation")



st.subheader("Tensorboard")