import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

DOCUMENTS_PATH = "./docs"
VECTOR_STORE_PERSIST_PATH = "vector_data"

def load_chunk_persist_pdf() -> Chroma:
    documents = []
    for file in os.listdir(DOCUMENTS_PATH):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(DOCUMENTS_PATH, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunked_documents = text_splitter.split_documents(documents)

    vector_db = Chroma.from_documents(
        documents=chunked_documents,
        embedding=OpenAIEmbeddings(),
        persist_directory=VECTOR_STORE_PERSIST_PATH
    )
    vector_db.persist()

    return vector_db