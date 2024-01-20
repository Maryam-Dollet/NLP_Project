import streamlit as st
from cache_func import load_reviews_sample2, load_companies_translated, load_reviews_sample

df = load_reviews_sample2()
df2 = load_companies_translated()
df3 = load_reviews_sample()

st.title("Data Processing")

st.subheader("First Cleaning")

st.write("In the descriptions and the reviews there are emojis and other punctuations which make the data unclean. This is why we need to clean it.")
st.write("To do this, we used the clean-text library: https://pypi., use_container_width=Trueorg/project/clean-text/")

st.write()
st.write("After cleaning the columns company description and reviews we chose to translate them.")

st.dataframe(df[["company_name", "description", "score"]], use_container_width=True)

st.subheader("Translation: Company Description")

st.write("The library used to translate the texts: https://pypi.org/project/deep-translator/")
st.write("First we translated the french company description to english, we took the unique company names from the reviews data and so had to translate maximum 12996 descriptions without taking into account the pages where the description is lacking. To translate wihtout errors we had to drop the rows without description data")

st.dataframe(df2)

st.subheader("Translation: Company reviews")
st.write("There are in total 235503 reviews without taking into account the None values. There is also an imbalance in the number of reviews per category. To reduce the problem we randomly chose the same number of reviews per category")

st.dataframe(df3)
