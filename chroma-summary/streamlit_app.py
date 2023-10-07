import os, tempfile
import streamlit as st
from langchain.llms.openai import OpenAI
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader

# Streamlit app
st.subheader('上传文档总结')

# Get OpenAI API key and source document input
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API key", value="", type="password")
    st.caption("*If you don't have an OpenAI API key, get it [here](https://platform.openai.com/account/api-keys).*")
source_doc = st.file_uploader("Source Document", label_visibility="collapsed", type="pdf")

# If the 'Summarize' button is clicked
if st.button("Summarize"):
    # Validate inputs
    if not openai_api_key.strip() or not source_doc:
        st.error(f"Please provide the missing fields.")
    else:
        try:
            with st.spinner('Please wait...'):
              # Save uploaded file temporarily to disk, load and split the file into pages, delete temp file
              with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                  tmp_file.write(source_doc.read())
              loader = PyPDFLoader(tmp_file.name)
              pages = loader.load_and_split()
              os.remove(tmp_file.name)

              # Create embeddings for the pages and insert into Chroma database
              embeddings=OpenAIEmbeddings(openai_api_key=openai_api_key)
              vectordb = Chroma.from_documents(pages, embeddings)

              # Initialize the OpenAI module, load and run the summarize chain
              llm=OpenAI(temperature=0, openai_api_key=openai_api_key, model_name="gpt-3.5-turbo-16k")
              chain = load_summarize_chain(llm, chain_type="stuff")
              search = vectordb.similarity_search(" ")
              summary = chain.run(input_documents=search, question="用中文撰写一段500字左右的文档总结，用中文回答我的问题。")

              st.success(summary)
        except Exception as e:
            st.exception(f"An error occurred: {e}")
