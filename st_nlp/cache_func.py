import streamlit as st
import pandas as pd
from gensim.models import Word2Vec


@st.cache_data
def load_companies_translated():
    return pd.read_csv("data/company_desc_translated.csv", sep=";").dropna(ignore_index=True)


@st.cache_data
def load_reviews_sample():
    return pd.read_csv("data/reviews_sample_translated.csv", sep=";").dropna(ignore_index=True)


@st.cache_data
def load_category():
    return pd.read_csv("data/category_data.csv", sep=";").dropna(ignore_index=True)

@st.cache_data
def load_reviews_sample2():
    return pd.read_csv("data/reviews_sample.csv")

@st.cache_data
def load_w2v():
    w2v_model = Word2Vec.load('models/w2v_company_desc_model')
    return w2v_model

def get_similarity(model, token):
    return model.wv.most_similar(positive=[token])

@st.cache_data
def load_glove():
    w2v_model = Word2Vec.load('models/glove_transfer')
    return w2v_model

@st.cache_data
def load_doc2vec():
    w2v_model = Word2Vec.load('models/d2v.model')
    return w2v_model