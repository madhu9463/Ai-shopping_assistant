import streamlit as st
from rag_pipeline import load_rag_chain, ask

# Streamlit App Configuration
st.set_page_config(page_title="E-commerce Assistant", page_icon="🛒", layout="centered")
st.title("🛒 E-commerce Shopping Assistant")
st.markdown("Hello! I'm your AI shopping assistant. What product are you looking for today?")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load RAG Chain (Cached so it only runs once)

@st.cache_resource(show_spinner=False)
def load_resources():
    with st.spinner("Đang tải dữ liệu cửa hàng và AI... Vui lòng đợi nhé."):
        chain = load_rag_chain()
    return chain

try:
    rag_chain = load_resources()
except Exception as e:
    st.error(f"	Error loading data: {e}")
    st.stop()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("VD: Recommend a blue cotton shirt for summer"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Retrieve and Generate

    with st.spinner("Searching for products and generating a response..."):
        try:
            result = ask(rag_chain, prompt)
            answer = result["answer"]

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(answer)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"An error occurred: {e}")
