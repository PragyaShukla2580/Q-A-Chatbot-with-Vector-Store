from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.pydantic_v1 import BaseModel as BM
from langchain_core.tools import tool
from src.history_management import *
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, END
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

class RetrieveRequest(BaseModel):
    query: str
    collection_name: str


class CustomState(BM):
    messages: list
    session_id: str
    collection_name: str

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")

reranker = SentenceTransformer("BAAI/bge-reranker-v2-m3")

llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key="ADD YOUR API KEY",
    model_name="meta-llama/llama-3.1-70b-instruct:free",
    temperature = 0
)


def rerank_chunks(query, chunks, top_k=5):
    """Rerank retrieved chunks based on semantic relevance to the query."""
    if not chunks:
        return []

    query_doc_pairs = [f"Query: {query}\nDocument: {chunk.page_content}" for chunk in chunks]

    scores = reranker.encode(query_doc_pairs, convert_to_tensor=True).tolist()

    sorted_chunks = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    
    return [chunk for chunk, _ in sorted_chunks[:top_k]] 


@tool(response_format="content_and_artifact")
def retrieve(data: RetrieveRequest):
    """
    Retrieve relevant documents from the FAISS index based on the query.

    Args:
        data (dict): A dictionary containing "query" and "collection_name".

    Returns:
        list: List of retrieved documents.
    """

    query = data.query
    collection_name = data.collection_name
    """Retrieve relevant information."""
    new_vector_store = FAISS.load_local(f"data/{collection_name}_vector_store", embeddings, allow_dangerous_deserialization=True)
    retrieved_docs = new_vector_store.similarity_search(query, k=10)
    top_5_docs = rerank_chunks(query, retrieved_docs, top_k=5)
    sources = [doc.metadata.get("source", "Unknown") for doc in top_5_docs]
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}")
        for doc in top_5_docs
    )
    return serialized, sources

def query_or_respond(state: CustomState):
    """Retrieve past 3 queries from FAISS memory and generate a response."""
    #query = state.messages[-1].content  
    last_message = state.messages[-1]

    if isinstance(last_message, dict):
        query = last_message.get("content", "")
    else:
        query = last_message.content
    session_id = state.session_id 
    collection_name = state.collection_name
    print(session_id)

    past_interactions = retrieve_chat_history(session_id,collection_name)
    all_questions = ""
    for i in past_interactions:
        all_questions = all_questions + " " + i['input']
    #past_interactions = [i for i in past_interactions if i ['session_id'] == session_id]
    # if len(past_interactions)> 0:
    #     #print("check")
    #     #print(type(past_interactions[-1]))
    #     last_msg = past_interactions[-1]
    #     past_context = f"User: {last_msg['input']}\nBot: {last_msg['output']}"
    # else:
    #     past_context = ""
    full_query = f"{all_questions} {query}"
    print(full_query)
    retrieved_output = retrieve.invoke({"data": {"query": full_query.strip(), "collection_name": collection_name}})

    system_message_content = (
         "You are an assistant for question-answering tasks. "
        "Use the following retrieved context and chat history to answer the question."
        "If there are any input and outputs in chat history, it means it is a follow up question. So answer accordingly, keeping chat history in mind."
        "If chat history is empty, then you have to answer is using the retrieved context only."
        "If you don't know the answer, say that you don't know."
        "Make the answer detailed and easily understandable."
        "Please list the relevant sources which are paths to a document at the end if there are any in the retrieved context. Do not give random github sources, just the sources in the following retrieved context which are useful for you to answer the query.\n\n"
        f"Chat History:\n{past_interactions}\n\n"
        f"New Query:\n{query}\n\n"
        f"Retrieved Context:\n{retrieved_output}"
    )
    print(system_message_content)

    response = llm.invoke(system_message_content)

    store_chat_history(session_id, query, response.content,collection_name)

    response_text = response.content

    return {"messages": [response_text]}


graph_builder = StateGraph(CustomState)  # Use CustomState instead of MessagesState

graph_builder.add_node(query_or_respond)
graph_builder.add_node(ToolNode([retrieve]))
graph_builder.set_entry_point("query_or_respond")

graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"}
)
graph_builder.add_edge("tools", END)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# def run_rag_pipeline(input_message, session_id="default_session"):
#     config = {"configurable": {"thread_id": session_id}}
#     formatted_message = {"role": "user", "content": input_message}
#     print("2nd")
#     print(session_id)
    
#     results = []
#     for step in graph.stream({"messages": [formatted_message], "session_id": session_id}, stream_mode="values", config=config):
#         results.append(step["messages"][-1]["content"])

#     return "\n".join(results)

def run_rag_pipeline(input_message, session_id="default_session",collection_name = "default_collection"):
    config = {"configurable": {"thread_id": session_id}}

    formatted_message = HumanMessage(content=input_message)
    print("2nd")
    print(session_id)
    results = []
    for step in graph.stream(
        {"messages": [formatted_message], "session_id": session_id,"collection_name":collection_name}, 
        stream_mode="values", 
        config=config
    ):
        last_message = step["messages"][-1]

        if isinstance(last_message, BaseMessage):
            results.append(last_message.content)
        else:
            results.append(str(last_message))

    return "\n".join(results)
