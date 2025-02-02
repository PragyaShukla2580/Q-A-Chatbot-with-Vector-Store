from fastapi import APIRouter, HTTPException
#from models.request import QueryRequest
from src.history_management import *
from src.rag_pipeline import run_rag_pipeline
from fastapi.responses import Response
from pydantic import BaseModel
from config import load_config
from utils.logging import logger
router = APIRouter()

class QueryRequest(BaseModel):
    session_id: str
    query: str  
    action: str
    collection_name : str


@router.post("/query")
def query_rag(request: QueryRequest):
    collection_name = request.collection_name
    try:
        config = load_config(collection_name)
        # vector_store_name = config["vector_store_name"]
        # chat_history_name = config["chat_history_name"]
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail=f"No FAISS vector store found for {collection_name}. Please run the store data API first.")
    if request.action == "stop":
        logger.info(f"Received request: {request.dict()} for collection: {collection_name}")
        return {"response": "Session ended."}

    session_id = request.session_id
    print("1st")
    print(session_id)
    if request.action == "start":
        logger.info(f"Clearing chat history for session {request.session_id}")
        clear_chat_history(session_id,collection_name)

    response_text = run_rag_pipeline(request.query, session_id=session_id,collection_name = collection_name)

    if "**Sources:**" in response_text:
        response_main, sources_section = response_text.split("**Sources:**")
        sources = sources_section.strip().split("\n")
    else:
        response_main, sources = response_text, []
    logger.info(f"Response for session {request.session_id}: {response_text}...")
    return Response(content=response_text, media_type="text/plain")