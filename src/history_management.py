from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import VectorStoreRetrieverMemory
from langchain_community.vectorstores import FAISS


embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")


user_memories = {}


def get_user_memory(session_id: str,collection_name):
    """Retrieve chat history for the given session from FAISS or create a new one."""
    chat_memory_store = FAISS.load_local(f"data/{collection_name}_history", embeddings, allow_dangerous_deserialization=True)
    global user_memories#, chat_memory_store

    if session_id in user_memories:
        return user_memories[session_id]

    all_messages = chat_memory_store.similarity_search("", k=1000)
    session_messages = [doc.metadata for doc in all_messages if doc.metadata.get("session_id") == session_id]

    session_memory = VectorStoreRetrieverMemory(
        retriever=chat_memory_store.as_retriever(search_kwargs={"k": 3}),
        memory_key="chat_history",
        return_messages=True,
        input_key="input"
    )

    for entry in session_messages:
        session_memory.save_context({"input": entry["input"]}, {"output": entry["output"]})

    user_memories[session_id] = session_memory
    return session_memory  

import uuid
from langchain_core.documents import Document

def store_chat_history(session_id: str, user_query: str, bot_response: str,collection_name):
    """Store user query and bot response with a unique doc_id in FAISS."""

    #global chat_memory_store
    chat_memory_store = FAISS.load_local(f"data/{collection_name}_history", embeddings, allow_dangerous_deserialization=True)

    unique_doc_id = str(uuid.uuid4())

    chat_doc = Document(
        page_content=user_query,
        metadata={
            "doc_id": unique_doc_id,
            "session_id": session_id,
            "input": user_query,
            "output": bot_response
        }
    )

    chat_memory_store.add_documents(documents=[chat_doc],ids = [unique_doc_id])
    chat_memory_store.save_local(f"data/{collection_name}_history")


def retrieve_chat_history(session_id: str,collection_name):
    """Retrieve last 3 stored chat messages for a session from FAISS."""
    #global chat_memory_store
    chat_memory_store = FAISS.load_local(f"data/{collection_name}_history", embeddings, allow_dangerous_deserialization=True)
    chat_docs = chat_memory_store.similarity_search("", k=1000)
    session_messages = [doc.metadata for doc in chat_docs if doc.metadata.get("session_id") == session_id]

    return session_messages[-3:]  

def clear_chat_history(session_id: str,collection_name):
    """Delete chat history **ONLY** for a specific session ID from FAISS."""

    #global chat_memory_store  

    chat_memory_store = FAISS.load_local(f"data/{collection_name}_history", embeddings, allow_dangerous_deserialization=True)

    chat_docs_with_scores = chat_memory_store.similarity_search_with_score("", k=1000)

    chat_docs = [doc for doc, _ in chat_docs_with_scores] 
    print("chatdocs")
    print(type(chat_docs))
    doc_ids_to_delete = [
        doc.metadata.get("doc_id") for doc in chat_docs 
        if doc.metadata.get("session_id") == session_id and doc.metadata.get("doc_id") is not None
    ]

    if doc_ids_to_delete:
        print(f"Deleting {len(doc_ids_to_delete)} documents for session_id: {session_id}")
        chat_memory_store.delete(doc_ids_to_delete)
        chat_memory_store.save_local(f"data/{collection_name}_history")
    else:
        print(f"No matching records found for session_id: {session_id}")