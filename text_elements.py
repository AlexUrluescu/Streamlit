import streamlit as st

st.markdown("## Hello world! - markdown")
st.title("Hello world! - title")
st.header("Hello world! - header")
st.subheader("Hello world! - subheader")
st.caption("Hello world! - caption")
st.code(""" def load_documents(self, file):
        name, extension = os.path.splitext(file)

        if extension == ".pdf":
            print(f"Loading file")
            loader = PyPDFLoader(file)
 - code""")

st.latex("\int a x^2 \ dx")