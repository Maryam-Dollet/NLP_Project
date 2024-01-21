import streamlit as st
from streamlit import components
import plotly.express as px
from cache_func import load_corpus, LDA

corpus = load_corpus()

st.title("Topic Modeling")

st.subheader("LDA")

lda, html_string, topic_words = LDA(corpus, 10)

with st.expander("Raw LDA"):
    st.write(lda.print_topics())

topic_words["pos"] = topic_words["topic"].apply(lambda x: 1 if x%2 == 0 else 2)
# st.write(topic_words.sort_values("topic"))

col1, col2 = st.columns(2)

with col1:
    topics1 = topic_words[topic_words['pos']== 1]
    for n in topics1['topic'].unique():
        topics_f = topics1[topics1['topic'] == n]
        fig = px.bar(topics_f.sort_values("value"), x="value", y="word", title=f'topic {n}')
        st.plotly_chart(fig)

with col2:
    topics2 = topic_words[topic_words['pos']== 2]
    for n in topics2['topic'].unique():
        topics_f = topics2[topics2['topic'] == n]
        fig = px.bar(topics_f.sort_values("value"), x="value", y="word", title=f'topic {n}')
        st.plotly_chart(fig)

st.subheader("pyLDAvis Visualization")
components.v1.html(html_string, width=1300, height=800)
