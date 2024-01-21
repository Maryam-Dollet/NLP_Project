import streamlit as st
import pandas as pd
import time
import random
from cache_func import load_company_tagged, load_doc2vec, find_similar_doc

df = load_company_tagged()
d2v = load_doc2vec()

st.title("Semantic Search Using Doc2Vec")

st.write("To be able to make a similarity search of documents with a sentence, we trained a Doc2Vec with each description of a company having a tag. This enables us to search the company that matches the best the request")

st.dataframe(df)

# sentence = st.text_input('text')
# if st.button("Get similar doc:"):
#     best_match = find_similar_doc(d2v, sentence, df)
#     best_match = [int(x) for x in best_match]
#     st.dataframe(df[df['tag'].isin(best_match[:5])])


# prompt = st.chat_input("Say something")
# if prompt:
#     st.write(f"User has sent the following prompt: {prompt}")
#     best_match = find_similar_doc(d2v, prompt, df)
#     best_match = [int(x) for x in best_match]

#     df_match = df[df['tag'].isin(best_match[:5])]

#     for index, row in df_match.iterrows():
#         st.write(row["company_name"])
#         st.write(row["description_en"])

st.markdown("## Chatbot Test")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
    for response in message["responses"]:
        with st.chat_message(message["role"]):
            st.markdown(response)
# Accept user input
if prompt := st.chat_input("What are your needs?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt, "responses": []})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    best_match = find_similar_doc(d2v, prompt, df)
    best_match = [int(x) for x in best_match]
    df_match = df[df['tag'].isin(best_match[:3])]

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        assistant_response = f"Here are the best matches found for {prompt}: "
        # Simulate stream of response with milliseconds delay
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)


    responses = []
    for index, row in df_match.iterrows():
        time.sleep(0.5)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            response1 = ""
            assistant_response = row["company_name"] + " : " + row["description_en"]
            for chunk in assistant_response.split():
                response1 += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(response1 + "▌")
            message_placeholder.markdown(response1)
            responses.append(response1)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response, "responses": responses})