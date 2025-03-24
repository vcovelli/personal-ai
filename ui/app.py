import streamlit as st
import requests

st.set_page_config(page_title="Local AI Assistant", layout="wide")
st.title("🧠 Local AI Assistant")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

model = st.selectbox("Choose a model", ["mistral", "llama3", "tinyllama"], index=0)

# Input box
if prompt := st.chat_input("Ask anything..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Send to FastAPI on desktop
    try:
        res = requests.post(
            "http://localhost:8000/chat",
            json={"prompt": prompt, "model": model},
            timeout=30
        )
        ai_msg = res.json()["response"]
    except Exception as e:
        ai_msg = f"⚠️ Error: {e}"

    st.chat_message("assistant").markdown(ai_msg)
    st.session_state.messages.append({"role": "assistant", "content": ai_msg})
