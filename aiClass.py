
import os
# from dotenv import load_dotenv
# load_dotenv()

from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA

from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
import tiktoken


class AskChat():
    # def __init__(self):
        # self.OPENAI_API_TYPE = "azure"
        # self.OPENAI_API_VERSION = os.environ.get("OPENAI_API_VERSION")
        # self.OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
        # self.OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE")        
        # self.OPENAI_API_VERSION = "2023-03-15-preview"
        # self.OPENAI_API_KEY = "9ac347d13e834f288a2076ff9c7b418a"
        # self.OPENAI_API_BASE = "https://sbzdfopenai.openai.azure.com/"
    

        # self.database = database
        # self.query = query

    
    def answering(self, database, query):
        OPENAI_API_KEY = "9ac347d13e834f288a2076ff9c7b418a"

        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


        db3 = Chroma(persist_directory= database, embedding_function=embedding_function)


        retriever = db3.as_retriever()
        # relevant_docs = retriever.get_relevant_documents(self.query)
        # # print(relevant_docs)

        llm = AzureChatOpenAI(deployment_name="gpt35-uif54579", openai_api_key="9ac347d13e834f288a2076ff9c7b418a", openai_api_base=self.OPENAI_API_BASE, openai_api_version=self.OPENAI_API_VERSION)

        chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever = retriever, return_source_documents = True)

        llm_response = chain(query)

        return llm_response['result']


    # This function get the file and create Documents
    def load_documents(self, file):
        name, extension = os.path.splitext(file)

        if extension == ".pdf":
            print(f"Loading {file}")
            loader = PyPDFLoader(file)

        
        elif extension == ".docx":
            print(f"Loading {file}")
            loader = Docx2txtLoader(file)

        elif extension == ".txt":
            print(f"Loading {file}")
            loader = TextLoader(file)

        else:
            print("This document is not supported")


        data = loader.load()
        return data


    # This function get the Documents and create Chunks
    def chunk_data(self, data, chunk_size, chunk_overlap = 0):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(data)

        return chunks


    # This function receive the chunks and create Embeddings into a Vector Store
    def create_embeddings(self, chunks, openai_api_base1, openai_api_key1):

        # OPENAI_API_KEY = "9ac347d13e834f288a2076ff9c7b418a"
        print(openai_api_base1)
        print(openai_api_key1)
        

        embeddings = OpenAIEmbeddings(
            deployment="adagptmodel",
            model="text-embedding-ada-002",
            openai_api_type="azure",
            # openai_api_base=openai_api_base1,
            # openai_api_key=openai_api_key1,            
            openai_api_base="https://sbzdfopenai.openai.azure.com/",
            openai_api_key = "9ac347d13e834f288a2076ff9c7b418a",
            chunk_size = 1
        )

        vector_store = Chroma.from_documents(chunks, embeddings)

        return vector_store


    def ask_and_get_answer(self, vector_store, question, kwargs, api_base, api_key, api_version):

        # OPENAI_API_KEY = "9ac347d13e834f288a2076ff9c7b418a"
        # OPENAI_API_BASE = "https://sbzdfopenai.openai.azure.com/"
        # OPENAI_API_VERSION = "2023-03-15-preview"

        llm = AzureChatOpenAI(
            deployment_name = "gpt35-uif54579",
            model_name = "gpt-35-turbo",
            openai_api_key= api_key,
            openai_api_base = api_base,
            openai_api_version = api_version
        )

        retriever = vector_store.as_retriever(search_type = "similarity", search_kwargs = {"k": kwargs})

        chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever= retriever)

        answer = chain.run(question)

        return answer


    # This function calculate the cost of the embeddings
    def calculate_embedding_cost(self, texts):
        
        enc = tiktoken.encoding_for_model("text-embedding-ada-002")
        total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])

        return total_tokens, total_tokens / 1000 * 0.4