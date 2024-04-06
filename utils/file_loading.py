import os
import mimetypes
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

def create_text_splitter(size=1000, overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return text_splitter

def add_metadata(docs, name):
    for doc in docs:
        if "page" in doc.metadata:
            doc.metadata["source"] = f"{name}, page {doc.metadata['page']}"
        else:
            doc.metadata["source"] = name

async def load_document_to_db(db, file, name):
    # load documents
    loader = create_loader(file)
    
    if loader:
        documents = loader.load()
        text_splitter = create_text_splitter()
        # split documents
        docs = text_splitter.split_documents(documents)
        # add name of file to the metadata
        add_metadata(docs, name)
        # add the document
        await db.aadd_documents(docs)

def create_loader(file):
    extension = os.path.splitext(file)[1]
    mime_type = mimetypes.guess_type(file)[0]
    print(extension, mime_type)

    if extension == ".pdf":
        return PyPDFLoader(file)
    elif extension == ".txt":
        return TextLoader(file)
    elif extension == ".docx":
        return Docx2txtLoader(file)
    else:
        return None