import streamlit as st
import pandas as pd
from cache_func import load_company_tagged, load_doc2vec, find_similar_doc

st.title("Semantic Search Using Doc2Vec")

df = load_company_tagged()

st.dataframe(df)