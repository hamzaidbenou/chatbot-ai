import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# read env variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def get_llm():
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)
    return llm

def get_embedding():
    return OpenAIEmbeddings()

def create_chat_memory(memory_key="chat_history", output_key="answer"):
    memory = ConversationBufferMemory(return_messages=True, memory_key=memory_key, output_key=output_key)
    return memory

def create_chain(llm, db, chain_type="stuff", search_type="similarity", search_kwargs={"k": 3}):
    # create memory provider
    memory = create_chat_memory()
    # define retriever
    retriever = db.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
    # create a chatbot chain.
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        chain_type=chain_type, 
        retriever=retriever,
        memory = memory,
        return_source_documents=True,
        return_generated_question=True,
        verbose=True
    )
    return qa