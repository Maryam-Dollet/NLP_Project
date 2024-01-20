import streamlit as st
import plotly_express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import pandas as pd


from cache_func import load_w2v, load_glove, get_similarity, get_PCA, get_TSNE

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

st.subheader("PCA Visualisation")

st.markdown("#### PCA Word2Vec")

result_df1 = get_PCA(w2v)

fig = go.Figure(data=go.Scatter(x=result_df1['x'],
                                y=result_df1['y'],
                                mode='markers',
                                text=result_df1['word'])) # hover text goes here

fig.update_layout(width=1500 ,height=1000)
st.plotly_chart(fig)

st.markdown("#### PCA Augmented Model")

result_df2 = get_PCA(glove)

fig = go.Figure(data=go.Scatter(x=result_df2['x'],
                                y=result_df2['y'],
                                mode='markers',
                                text=result_df2['word'])) # hover text goes here

fig.update_layout(width=1500 ,height=1000)
st.plotly_chart(fig)

st.subheader("TSNE Visualisation")

st.markdown("#### TSNE Word2Vec")

result_df3 = get_PCA(w2v)

fig = go.Figure(data=go.Scatter(x=result_df3['x'],
                                y=result_df3['y'],
                                mode='markers',
                                text=result_df3['word'])) # hover text goes here

fig.update_layout(width=1500 ,height=1000)
st.plotly_chart(fig)

st.markdown("#### TSNE Augmented Model")

result_df4 = get_PCA(glove)

fig = go.Figure(data=go.Scatter(x=result_df4['x'],
                                y=result_df4['y'],
                                mode='markers',
                                text=result_df4['word'])) # hover text goes here

fig.update_layout(width=1500 ,height=1000)
st.plotly_chart(fig)


st.subheader("Tensorboard")

st.markdown("#### Word2Vec")

st.markdown("#### Augmented Model")