import streamlit as st
from streamlit import components
from cache_func import load_corpus, LDA

corpus = load_corpus()

st.title("Topic Modeling")

st.subheader("LDA")

lda, html_string = LDA(corpus, 10)


st.write(lda.print_topics())

components.v1.html(html_string, width=1300, height=800)
