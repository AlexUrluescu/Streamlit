import streamlit as st
import os
from dotenv import load_dotenv

from aiClass import AskChat

ai = AskChat()

load_dotenv()

os.environ.get("OPENAI_API_KEY")
os.environ.get("OPENAI_API_BASE")
os.environ.get("OPENAI_API_VERSION")

st.image("img.png")
st.subheader("LLM QESTION-ANSWERING APPLICATION")

with st.sidebar:
    api_key = st.text_input("OpenAI API Key:", type="password")
    api_base = st.text_input("OpenAI Base:", type="default")
    api_version = st.text_input("OpenAI Version:", type="default")

    if api_key and api_base and api_version:
        OPENAI_API_KEY = api_key
        OPENAI_API_BASE = api_base
        OPENAI_API_VERSION = api_version

    
    uploaded_file = st.file_uploader("Upload a file:", type=["pdf", "docx", "txt"])

    chunk_size = st.number_input("Chunk size", min_value=100, max_value=2048, value=512)

    number_relevant_chunks = st.number_input("Number of relevant chunks", min_value = 1, max_value=20, value=3)
    add_data = st.button("Add Data")

    if uploaded_file and add_data:
        with st.spinner("Reading, chunking and embedding file ..."):
            bytes_data = uploaded_file.read()
            file_name = os.path.join("./", uploaded_file.name)
            with open(file_name, "wb") as f:
                f.write(bytes_data)

            data = ai.load_documents(file_name)
            chunks = ai.chunk_data(data, chunk_size=chunk_size)
            st.write(f"Chunk size: {chunk_size}, Chunks: {len(chunks)}")

            tokens, embedding_cost = ai.calculate_embedding_cost(chunks)
            st.write(f"Embedding cost: ${embedding_cost:.4f}")

            vector_store = ai.create_embeddings(chunks)
            st.session_state.vs = vector_store
            st.success("File uploaded, chunked and embedded suceessfully")


question = st.text_input("Ask a question about the content of your file:")

if question:
    if "vs" in st.session_state:
        vector_store = st.session_state.vs
        st.write(f"Number of relevant chunks: {number_relevant_chunks}")
        answer = ai.ask_ang_get_answer( vector_store, question, number_relevant_chunks )
        st.text_area("LLM Answer: ", value=answer)