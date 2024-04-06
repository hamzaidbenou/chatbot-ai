import chainlit as cl
from dotenv import load_dotenv
from utils.db import connect_db
from utils.file_loading import load_document_to_db
from utils.models import create_chain, get_embedding, get_llm

load_dotenv()

@cl.on_chat_start
async def start_chat():
    try:
        # get llm model
        llm = get_llm()
        # get embedding model
        embeddings = get_embedding()
        # connect to vector database
        # add "faiss" as param to use FAISS db
        db = connect_db(embeddings)
        cl.user_session.set("db", db)
        # create the conversational retrieval chain
        if llm and db:
            # search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.8, "k": 3}
            chain = create_chain(llm=llm, db=db)
            cl.user_session.set("chain", chain)
    except Exception as ex:
        print("Connection to vector db : An exception occurred")
        print(ex)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    db = cl.user_session.get("db")

    if db:
        if message.elements:
            for element_doc in message.elements:
                if hasattr(element_doc, "path"):
                    await load_document_to_db(db, element_doc.path, getattr(element_doc, "name", ""))

    if chain:
        res = await chain.ainvoke(message.content)
        answer = res["answer"]
        source_documents = res["source_documents"]

        text_elements = []

        if source_documents:
            for source_idx, source_doc in enumerate(source_documents):
                source_name = f"source_{source_idx}"
                if "source" in source_doc.metadata and source_doc.metadata["source"]:
                    source_name = source_doc.metadata["source"]
                # Create the text element referenced in the message
                text_elements.append(
                    cl.Text(content=source_doc.page_content, name=source_name)
                )
            source_names = [text_el.name for text_el in text_elements]

            if source_names:
                answer += f"\n\nSources: {', '.join(source_names)}"
            else:
                answer += "\n\nNo sources found"

        await cl.Message(content=answer, elements=text_elements).send()