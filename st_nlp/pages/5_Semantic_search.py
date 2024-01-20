import streamlit as st
import pandas as pd
from cache_func import load_company_tagged, load_doc2vec, find_similar_doc

df = load_company_tagged()
d2v = load_doc2vec()

st.title("Semantic Search Using Doc2Vec")

st.dataframe(df)

sentence = st.text_input('text')
if st.button("Get similar doc:"):
    best_match = find_similar_doc(d2v, sentence, df)
    best_match = [int(x) for x in best_match]
    # st.write(best_match)
    st.dataframe(df[df['tag'].isin(best_match[:5])])