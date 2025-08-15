import streamlit as st
import requests

# Change this if you're running app on a different host/port
API_URL = "http://localhost:8000/ask"

st.set_page_config(page_title="iOS 26 Q&A Bot", layout="wide")

st.title("Chat with iOS 26 Bot")
st.markdown("Ask anything about **upcoming iOS 26** and get grounded answers with sources.")

# Input
question = st.text_input("Your question about iOS 26:", placeholder="E.g., What's new in beta 6?")
ask_button = st.button("Ask")

if ask_button and question.strip():
    with st.spinner("Searching…"):
        try:
            resp = requests.post(API_URL, json={"question": question})
            resp.raise_for_status()
            data = resp.json()
            answer = data.get("answer", "")
            sources = data.get("sources", [])
        except Exception as e:
            st.error(f"Error fetching answer: {e}")
            st.stop()

    st.subheader("Answer")
    st.markdown(answer)

    if sources:
        st.subheader("Sources (ranked):")
        for src in sources:
            st.markdown(f"- [{src['title']}]({src['url']}) — *{src.get('date','')}* (score: {src['score']:.2f})")
