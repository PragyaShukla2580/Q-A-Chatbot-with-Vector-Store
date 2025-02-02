import os
import re
import faiss
from uuid import uuid4
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from utils.logging import logger
import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

router = APIRouter()


class FAISSRequest(BaseModel):
    folder_path: str
    doc_extensions: List[str]
    collection_name: str

# Initialize Embeddings
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")

# chat_memory_index = faiss.IndexFlatL2(len(embeddings.embed_query("random")))  # Adjust dimension as needed
# chat_memory_store = FAISS(
#     embedding_function=embeddings,
#     index=chat_memory_index,
#     docstore=InMemoryDocstore(),
#     index_to_docstore_id={},
# )
# chat_memory_store.save_local("history_sessions")

def chunk_markdown(text):
    """
    Splits Markdown content into chunks based on headings.
    Returns a list of (heading, chunk_text) tuples.
    """
    chunks = []
    current_chunk = []
    current_heading = "Introduction"

    lines = text.split("\n")
    for line in lines:
        if re.match(r"^#{1,6} ", line):  # Detect headings
            if current_chunk:  # Save the previous chunk
                chunks.append((current_heading, "\n".join(current_chunk)))
            current_heading = line.strip()  # Update heading
            current_chunk = [line]  # Start a new chunk
        else:
            current_chunk.append(line)

    if current_chunk:
        chunks.append((current_heading, "\n".join(current_chunk)))

    return chunks

def split_large_chunks(chunk_text, max_words=512):
    """
    Splits large chunks while ensuring code blocks remain intact.
    """
    lines = chunk_text.split("\n")
    new_chunks = []
    current_chunk = []
    inside_code_block = False
    word_count = 0

    for line in lines:
        if line.strip().startswith("```"):
            inside_code_block = not inside_code_block  # Toggle code block state

        current_chunk.append(line)
        word_count += len(line.split())

        # Split only if not inside a code block and exceeding max_words
        if word_count > max_words and not inside_code_block:
            new_chunks.append("\n".join(current_chunk))
            current_chunk = []
            word_count = 0

    if current_chunk:
        new_chunks.append("\n".join(current_chunk))

    return new_chunks

def merge_small_chunks(chunks, min_words=50):
    """
    Merges small chunks (< min_words) with the next chunk if possible.
    """
    merged_chunks = []
    buffer_chunk = None

    for heading, chunk_text in chunks:
        chunk_length = len(chunk_text.split())

        if chunk_length < min_words:
            if buffer_chunk:
                buffer_chunk[1] += "\n\n" + chunk_text  # Merge with previous buffer
            else:
                buffer_chunk = [heading, chunk_text]
        else:
            if buffer_chunk:
                merged_chunks.append(tuple(buffer_chunk))
                buffer_chunk = None
            merged_chunks.append((heading, chunk_text))

    if buffer_chunk:
        merged_chunks.append(tuple(buffer_chunk))

    return merged_chunks

def process_markdown(text, max_words=512, min_words=50):
    """
    Full processing pipeline:
    1. Chunk by headings
    2. Split large chunks while preserving code blocks
    3. Merge small chunks
    """
    # Step 1: Chunk by headings
    chunks = chunk_markdown(text)

    # Step 2: Split large chunks
    updated_chunks = []
    for heading, chunk_text in chunks:
        if len(chunk_text.split()) > max_words:
            split_chunks = split_large_chunks(chunk_text, max_words)
            for sub_chunk in split_chunks:
                updated_chunks.append((heading, sub_chunk))
        else:
            updated_chunks.append((heading, chunk_text))

    # Step 3: Merge small chunks
    final_chunks = merge_small_chunks(updated_chunks, min_words)

    return final_chunks



# # Function to Chunk Documents
# def chunk_document(text):

#     headings = re.split(r'(^#+ .*$)', text, flags=re.MULTILINE)
#     chunks = []
#     for i in range(1, len(headings), 2):
#         chunk_text = headings[i] + (headings[i + 1] if i + 1 < len(headings) else '')
#         chunks.append(chunk_text.strip())
#     return chunks

# Root Folder Path for Markdown Files
#chunk_store = []
def process_folder(folder_path, doc_extensions,collection_name):
    """Processes all files with given extensions, creates FAISS index, and saves configuration."""
    
    last_folder_name = collection_name
    vector_store_name = f"{last_folder_name}_vector_store"
    chat_history_name = f"{last_folder_name}_history"

    chunk_store = []
# Walk Through Folder and Extract Text Chunks
    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(file.endswith(ext) for ext in doc_extensions):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    chunks = process_markdown(content)
                    for chunk in chunks:
                        chunk_store.append(Document(page_content=chunk[1], metadata={"source": file_path}))

    # Generate UUIDs for Each Chunk
    uuids = [str(uuid4()) for _ in range(len(chunk_store))]

    # Initialize FAISS Index
    index = faiss.IndexFlatL2(len(embeddings.embed_query("random")))

    # Create Vector Store
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    # Add Documents to Vector Store
    vector_store.add_documents(documents=chunk_store, ids=uuids)

# Save FAISS Index
    vector_store.save_local(f"data/{vector_store_name}")

    chat_memory_index = faiss.IndexFlatL2(len(embeddings.embed_query("random")))  # Adjust dimension as needed
    chat_memory_store = FAISS(
        embedding_function=embeddings,
        index=chat_memory_index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    chat_memory_store.save_local(f"data/{chat_history_name}")
    config_data = {
        "vector_store_name": vector_store_name,
        "chat_history_name": chat_history_name
    }

    config_path = f"configs/{last_folder_name}_config.json"
    os.makedirs("configs", exist_ok=True)

    with open(config_path, "w") as config_file:
        json.dump(config_data, config_file, indent=4)

    logger.info(f"FAISS Index stored successfully in {vector_store_name}")
    return vector_store_name, chat_history_name, config_path


@router.post("/start_faiss")
def start_faiss_indexing(request: FAISSRequest):
    logger.info(f"Received request to process folder: {request.folder_path} with extensions: {request.doc_extensions}")

    try:
        vector_store_name, chat_history_name, config_path = process_folder(request.folder_path, request.doc_extensions,request.collection_name)
        return {
            "message": "FAISS Indexing Completed!",
            "vector_store_name": vector_store_name,
            "chat_history_name": chat_history_name,
            "config_file": config_path
        }
    except Exception as e:
        logger.error(f"Error processing folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))

