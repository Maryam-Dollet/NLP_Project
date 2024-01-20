import streamlit as st
import pandas as pd
from cache_func import load_companies_translated, load_reviews_sample, load_category

st.set_page_config(layout="wide")

df = pd.concat([load_companies_translated(), load_category()], axis=1)
df2 = load_reviews_sample()

st.title("Data Gathering: Web Scraping french companies on Trustpilot")

st.write("files: scrap_trust_pilot.ipynb and scrap_trust_pilot_2.ipynb")

st.write("Using Beautiful Soup to scrap the website first we extracted the links to the categories listed in this link:")

st.write("https://fr.trustpilot.com/categories")

st.markdown("#### Here are the categories extracted")

st.dataframe(df["category"].unique(), use_container_width=True)

st.markdown("#### Sample of the reviews we extracted")

st.dataframe(df2)
