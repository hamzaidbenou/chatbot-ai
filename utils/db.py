import faiss
from langchain_community.vectorstores.redis import Redis
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore

def create_faiss_db(embeddings):
    # Initialize the vectorstore as empty
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    db = FAISS(embeddings, index, InMemoryDocstore({}), {})
    db.save_local(folder_path="./database/faiss_db", index_name="docs")
    return db

def connect_faiss_db(embeddings):
    # remove the comment tag just the first time running the app
    # create_faiss_db()
    # create connection to our faiss vector database
    db = FAISS.load_local(folder_path="./database/faiss_db", embeddings=embeddings, 
                          index_name="docs", allow_dangerous_deserialization=True)
    return db

def connect_redis_db(embeddings):
    # create connection to our redis vector database
    schema = { "text": [{"name": "name"}, {"name": "source"}] }
    db = Redis(redis_url="redis://localhost:6379", index_name="docs", embedding=embeddings, index_schema=schema)
    return db

def connect_db(type="redis"):
    if type == "faiss":
        return connect_faiss_db()
    else:
        return connect_redis_db()