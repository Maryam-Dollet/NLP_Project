import streamlit as st
import pandas as pd
from cache_func import load_companies_translated, load_reviews_sample, load_category

st.set_page_config(layout="wide")

df = load_companies_translated() 
df2 = load_reviews_sample()

st.title("Data Gathering: Web Scraping french companies on Trustpilot")

st.write("files: scrap_trust_pilot.ipynb and scrap_trust_pilot_2.ipynb")

st.write("Using Beautiful Soup to scrap the website first we extracted the links to the categories listed in this link:")

st.write("https://fr.trustpilot.com/categories")

st.markdown("#### Here are the categories extracted")
a = pd.concat([pd.DataFrame(df["category"].unique(), columns=["category"]), load_category()], axis=1)

st.dataframe(a, use_container_width=True, column_config={"url":st.column_config.LinkColumn("App URL")})

st.write("Next, we used the links and number of pages extracted to extract all the links of each company")

st.write("Then, we extracted the data on each company site using the links extracted previously")

st.markdown("#### Sample of the reviews we extracted")

st.dataframe(df2.iloc[:, :-2])
