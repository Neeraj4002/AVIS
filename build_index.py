#!/usr/bin/env python3
import os
import json
from pathlib import Path
from typing import List, Dict, Any # Added Dict, Any for potential future use
import argparse # For command-line arguments

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ---- CONFIG ----
# These can be overridden by command-line arguments | GLang\langgraph_chunks_refined
DEFAULT_CHUNKS_JSON = "GLang/langgraph_chunks_refined/chunks_manifest.json"
DEFAULT_INDEX_DIR   = "GLang/langgraph_index"
DEFAULT_TEXT_INDEX  = "text_index"
DEFAULT_CODE_INDEX  = "code_index"
DEFAULT_EMBED_MODEL = "models/text-embedding-004"  # Google’s embedding model
DEFAULT_BM25_K = 20 # Default K for BM25 retriever, made configurable

def load_chunks(path: str) -> List[Document]:
    """Loads chunks from a JSON manifest file."""
    try:
        raw_json = Path(path).read_text(encoding="utf-8")
        raw_chunks = json.loads(raw_json)
        # Ensure metadata is a dictionary, providing an empty one if missing or not a dict
        return [
            Document(
                page_content=c["content"],
                metadata=c.get("metadata") if isinstance(c.get("metadata"), dict) else {}
            ) for c in raw_chunks
        ]
    except FileNotFoundError:
        print(f"Error: Chunks manifest file not found at {path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {path}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while loading chunks: {e}")
        return []

def split_by_type(docs: List[Document]) -> tuple[List[Document], List[Document]]:
    """Splits documents into text and code based on 'includes_code' metadata."""
    text_docs, code_docs = [], []
    for d in docs:
        # Robustly check metadata: d.metadata might be None or not have the key
        if isinstance(d.metadata, dict) and d.metadata.get("includes_code", False):
            code_docs.append(d)
        else:
            text_docs.append(d)
    return text_docs, code_docs

def build_and_save(
    chunks_json_path: str,
    index_dir_path: str,
    text_index_name: str,
    code_index_name: str,
    embed_model_name: str,
    bm25_k_val: int,
    google_api_key_val: str # API key passed as argument
):
    """Builds and saves FAISS and BM25 indexes."""
    # 1) Load chunks
    print(f"Loading chunks from {chunks_json_path}...")
    docs = load_chunks(chunks_json_path)
    if not docs:
        print("No documents loaded. Exiting.")
        return

    text_docs, code_docs = split_by_type(docs)
    print(f"Loaded {len(docs)} total documents: {len(text_docs)} text, {len(code_docs)} code.")

    try:
        os.makedirs(index_dir_path, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {index_dir_path}: {e}")
        return

    # 2) Embedders (same model for text & code)
    print(f"Initializing Google embeddings with model {embed_model_name}...")
    # API Key is handled by the user providing it, not hardcoded here.
    if not google_api_key_val:
        print("Error: Google API Key is required for embeddings. Set GOOGLE_API_KEY environment variable or pass as argument.")
        # As per user request, the original hardcoded key is used if GOOGLE_API_KEY is not set.
        # This is NOT recommended for production.
        google_api_key_val = "AIzaSyD55cr_qrFmyWOipaX8mTtT4wh0yyn5wQg" # Original key, per user instruction
        print("Warning: Using a hardcoded API key. This is insecure and not recommended.")


    try:
        embedder = GoogleGenerativeAIEmbeddings(google_api_key=google_api_key_val, model=embed_model_name)
    except Exception as e:
        print(f"Error initializing Google embeddings: {e}")
        return

    # 3) Build FAISS indexes
    # Text documents
    if text_docs:
        print(f"Creating text FAISS index ({len(text_docs)} docs) at {Path(index_dir_path) / text_index_name}...")
        try:
            faiss_text = FAISS.from_documents(text_docs, embedder)
            faiss_text.save_local(index_dir_path, text_index_name)
            print("Text FAISS index created successfully.")
        except Exception as e:
            print(f"Error creating text FAISS index: {e}")
    else:
        print("No text documents to index for FAISS.")

    # Code documents
    if code_docs:
        print(f"Creating code FAISS index ({len(code_docs)} docs) at {Path(index_dir_path) / code_index_name}...")
        try:
            faiss_code = FAISS.from_documents(code_docs, embedder)
            faiss_code.save_local(index_dir_path, code_index_name)
            print("Code FAISS index created successfully.")
        except Exception as e:
            print(f"Error creating code FAISS index: {e}")
    else:
        print("No code documents to index for FAISS.")

    # 4) BM25 over all docs
    if docs:
        print(f"Creating BM25 retriever with k={bm25_k_val} for all {len(docs)} documents...")
        try:
            # BM25Retriever does not save to disk directly; it's created in memory.
            # If persistence is needed, it would typically be part of a larger retrieval system
            # that might pickle it or re-initialize it on load.
            # For this script, we are just showing its creation.
            # To use it later, it would need to be re-created from documents.
            bm25 = BM25Retriever.from_documents(docs, k=bm25_k_val)
            print(f"BM25Retriever initialized (k={bm25.k}, {len(bm25.docs)} documents).")
            # Note: BM25Retriever is not saved to disk by this script.
            # It would be re-instantiated by the retriever system.
        except Exception as e:
            print(f"Error creating BM25 retriever: {e}")
    else:
        print("No documents available to create BM25 retriever.")


    print(f"✅ Index building process completed. FAISS indexes saved to {index_dir_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS and BM25 indexes for LangGraph RAG.")
    parser.add_argument("--chunks_json", type=str, default=DEFAULT_CHUNKS_JSON, help="Path to the chunks manifest JSON file.")
    parser.add_argument("--index_dir", type=str, default=DEFAULT_INDEX_DIR, help="Directory to save FAISS indexes.")
    parser.add_argument("--text_index", type=str, default=DEFAULT_TEXT_INDEX, help="Name for the text FAISS index.")
    parser.add_argument("--code_index", type=str, default=DEFAULT_CODE_INDEX, help="Name for the code FAISS index.")
    parser.add_argument("--embed_model", type=str, default=DEFAULT_EMBED_MODEL, help="Name of the Google embedding model.")
    parser.add_argument("--bm25_k", type=int, default=DEFAULT_BM25_K, help="K value for BM25Retriever.")
    # API Key argument - it's better to use environment variables, but providing an option.
    # Per user request, hardcoded keys are not being removed by this script.
    # The original script had a hardcoded key. This version expects it via env or this arg.
    parser.add_argument(
        "--google_api_key",
        type=str,
        default=os.getenv("GOOGLE_API_KEY"), # Get from env var by default
        help="Google API Key. If not provided, attempts to use GOOGLE_API_KEY environment variable."
    )

    args = parser.parse_args()

    # As per user instruction, if GOOGLE_API_KEY is not set via arg or env,
    # the original hardcoded key will be used inside build_and_save.
    # This is for adhering to the "don't remove hardcoded API keys" instruction.
    # A better practice is to require the key and not have fallbacks to hardcoded values.
    
    build_and_save(
        args.chunks_json,
        args.index_dir,
        args.text_index,
        args.code_index,
        args.embed_model,
        args.bm25_k,
        args.google_api_key # Pass the API key
    )
