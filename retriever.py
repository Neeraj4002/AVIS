import os
import json
from pathlib import Path
import re
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple # Added Tuple

from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---- CONFIG ----
# These could be made configurable (e.g., via environment variables or a config file) GLang\langgraph_chunks_refined
MANIFEST_JSON = "langgraph_chunks_refined/chunks_manifest.json"
INDEX_DIR     = "langgraph_index"
TEXT_INDEX    = "text_index" #GLang\langgraph_index
CODE_INDEX    = "code_index"
EMBED_MODEL   = "models/text-embedding-004"
DEFAULT_K     = 7
MMR_LAMBDA    = 0.5

# Ensemble retriever weights - made configurable as constants
# These weights determine the contribution of each retriever in the ensemble.
# Tuning these weights can significantly impact retrieval performance.
DEFAULT_FAISS_TEXT_WEIGHT = 0.4
DEFAULT_FAISS_CODE_WEIGHT = 0.4
DEFAULT_BM25_WEIGHT = 0.2

# LLM Models for reranking - consider consistency with the main QA LLM
# Using different models for reranking and generation can sometimes lead to
# discrepancies if their capabilities or biases differ significantly.
RERANK_LLM_FAST_MODEL = "gemini-2.0-flash-lite" # Or your preferred fast model
RERANK_LLM_FINAL_MODEL = "gemini-2.0-flash-lite"  # Or your preferred quality model


@lru_cache(maxsize=1) # Caches the loaded components to avoid reloading on each call
def load_retrieval_components(
    google_api_key_val: str, # API key passed as argument
    manifest_path: str = MANIFEST_JSON,
    index_dir_path: str = INDEX_DIR,
    text_idx_name: str = TEXT_INDEX,
    code_idx_name: str = CODE_INDEX,
    embed_model_name: str = EMBED_MODEL,
    bm25_k_val: int = 15, # Default K for BM25 during initialization for ensemble
    faiss_text_k: int = 15,
    faiss_code_k: int = 15,
    ensemble_weights: Optional[List[float]] = None
) -> Tuple[List[Document], GoogleGenerativeAIEmbeddings, Optional[FAISS], Optional[FAISS], BM25Retriever, EnsembleRetriever, ChatGoogleGenerativeAI, ChatGoogleGenerativeAI]:
    """
    Loads all necessary components for the retrieval system.
    Includes documents, embedder, FAISS stores, BM25 retriever, ensemble retriever, and reranking LLMs.
    """
    # API Key is handled by the user providing it, not hardcoded here.
    # Per user request, the original hardcoded key is used if GOOGLE_API_KEY is not set.
    # This is NOT recommended for production.
    if not google_api_key_val:
        google_api_key_val = "AIzaSyB-CXqCqmdcxv-WiaoNKa5mQpHw0n_A_aE" # Original key, per user instruction
        print("Warning: Using a hardcoded API key in load_retrieval_components. This is insecure.")

    # 1) Load manifest and create Langchain Documents
    print(f"Loading manifest from: {manifest_path}")
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            raw_manifest = json.load(f)
        docs = [Document(page_content=c["content"], metadata=c.get("metadata", {})) for c in raw_manifest]
        if not docs:
            print("👎Warning: No documents loaded from manifest. BM25 and FAISS might fail or be empty.")
    except FileNotFoundError:
        print(f"❌Error: Manifest file not found at {manifest_path}. Cannot load documents.")
        docs = []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {manifest_path}. Cannot load documents.")
        docs = []
    except Exception as e:
        print(f"An unexpected error occurred loading manifest: {e}")
        docs = []


    # 2) Initialize Google embeddings
    print(f"Initializing Google embeddings with model: {embed_model_name}")
    try:
        embedder = GoogleGenerativeAIEmbeddings(google_api_key=google_api_key_val, model=embed_model_name)
    except Exception as e:
        print(f"Fatal: Error initializing Google embeddings: {e}. Retrieval system cannot function.")
        raise  # Re-raise as this is critical

    # 3) Load FAISS indexes
    faiss_text_retriever, faiss_code_retriever = None, None
    # Text FAISS
    try:
        if os.path.exists(Path(index_dir_path) / f"{text_idx_name}.faiss"):
            print(f"Loading text FAISS index from: {Path(index_dir_path) / text_idx_name}")
            # User requested not to change this line regarding allow_dangerous_deserialization
            faiss_text_store = FAISS.load_local(index_dir_path, embedder, text_idx_name, allow_dangerous_deserialization=True)
            faiss_text_retriever = faiss_text_store.as_retriever(search_kwargs={"k": faiss_text_k})
            print("✅Text FAISS index loaded.")
        else:
            print(f"Warning: Text FAISS index not found at {Path(index_dir_path) / text_idx_name}. Text FAISS retriever will be disabled.")
    except Exception as e:
        print(f"Error loading text FAISS index: {e}. Text FAISS retriever will be disabled.")

    # Code FAISS
    try:
        if os.path.exists(Path(index_dir_path) / f"{code_idx_name}.faiss"):
            print(f"Loading code FAISS index from: {Path(index_dir_path) / code_idx_name}")
            # User requested not to change this line regarding allow_dangerous_deserialization
            faiss_code_store = FAISS.load_local(index_dir_path, embedder, code_idx_name, allow_dangerous_deserialization=True)
            faiss_code_retriever = faiss_code_store.as_retriever(search_kwargs={"k": faiss_code_k})
            print("✅Code FAISS index loaded.")
        else:
            print(f"❌Warning: Code FAISS index not found at {Path(index_dir_path) / code_idx_name}. Code FAISS retriever will be disabled.")
    except Exception as e:
        print(f"Error loading code FAISS index: {e}. Code FAISS retriever will be disabled.")

    # 4) Initialize BM25 retriever
    # BM25Retriever needs documents. If docs list is empty, it will error or be ineffective.
    if not docs:
        print("❌Warning: No documents available for BM25Retriever. It will be ineffective.")
        # Create a dummy BM25Retriever or handle appropriately
        # For now, let it proceed; it might error if from_documents is called with empty list
        # depending on Langchain version. A more robust approach would be to skip it.
        # However, Langchain's BM25Retriever.from_documents([]) might create an empty retriever.
    print(f"Initializing BM25 retriever with k={bm25_k_val}...")
    try:
        bm25_retriever = BM25Retriever.from_documents(docs if docs else [], k=bm25_k_val) # Pass empty list if no docs
        print("✅BM25 retriever initialized.")
    except Exception as e:
        print(f"❌Error initializing BM25 retriever: {e}")
        # Fallback to a dummy or raise error
        # For now, creating a non-functional one if docs were empty.
        # This is not ideal but avoids crashing if docs is empty.
        # A better solution is to not add it to the ensemble if it can't be created.
        class DummyBM25: # Simple dummy
            def invoke(self, query, config=None): return []
            def get_relevant_documents(self, query): return []
        bm25_retriever = DummyBM25()
    print("✅BM25 retriever initialized.")


    # 5) Initialize Hybrid (Ensemble) retriever
    # Filter out None retrievers before adding to ensemble
    active_retrievers = []
    if faiss_text_retriever:
        active_retrievers.append(faiss_text_retriever)
    if faiss_code_retriever:
        active_retrievers.append(faiss_code_retriever)
    if docs: # Only add BM25 if there were documents to build it from
         active_retrievers.append(bm25_retriever)


    if not active_retrievers:
        print("CRITICAL: No active retrievers could be initialized. Ensemble retriever will not work.")
        # Fallback or raise an error, as the system is non-functional
        # For now, we'll create a dummy ensemble retriever.
        class DummyEnsemble:
            def invoke(self, query, config=None): return []
            def get_relevant_documents(self, query): return []
        hybrid_retriever = DummyEnsemble()
    else:
        # Determine weights for the ensemble
        if ensemble_weights is None or len(ensemble_weights) != len(active_retrievers):
            print(f"❌Warning: Ensemble weights not provided or mismatched. Using default/adapted weights.")
            # Adapt default weights based on available retrievers
            default_weights_map = {
                "faiss_text": DEFAULT_FAISS_TEXT_WEIGHT,
                "faiss_code": DEFAULT_FAISS_CODE_WEIGHT,
                "bm25": DEFAULT_BM25_WEIGHT
            }
            current_weights = []
            if faiss_text_retriever in active_retrievers: current_weights.append(default_weights_map["faiss_text"])
            if faiss_code_retriever in active_retrievers: current_weights.append(default_weights_map["faiss_code"])
            if bm25_retriever in active_retrievers and docs : current_weights.append(default_weights_map["bm25"])

            # Normalize weights if they don't sum to 1 (optional, EnsembleRetriever handles it)
            if current_weights and sum(current_weights) > 0 :
                 final_weights = [w / sum(current_weights) for w in current_weights] if sum(current_weights) != 1 else current_weights
            else: # if somehow all weights are zero or no retrievers
                 final_weights = [1.0 / len(active_retrievers)] * len(active_retrievers) if active_retrievers else []

        else:
            final_weights = ensemble_weights
        
        print(f"Initializing Ensemble retriever with {len(active_retrievers)} components and weights: {final_weights}")
        try:
            hybrid_retriever = EnsembleRetriever(retrievers=active_retrievers, weights=final_weights)
            print("Ensemble retriever initialized.")
        except Exception as e:
            print(f"Error initializing Ensemble retriever: {e}")
            class DummyEnsemble:
                def invoke(self, query, config=None): return []
                def get_relevant_documents(self, query): return []
            hybrid_retriever = DummyEnsemble()


    # 6) Initialize Gemini LLMs for reranking
    print(f"Initializing LLMs for reranking: Fast ({RERANK_LLM_FAST_MODEL}), Final ({RERANK_LLM_FINAL_MODEL}).")
    try:
        llm_fast  = ChatGoogleGenerativeAI(google_api_key=google_api_key_val, model=RERANK_LLM_FAST_MODEL, temperature=0.5)
        llm_final = ChatGoogleGenerativeAI(google_api_key=google_api_key_val, model=RERANK_LLM_FINAL_MODEL, temperature=0.2)
        print("Reranking LLMs initialized.")
    except Exception as e:
        print(f"Error initializing reranking LLMs: {e}. Reranking might fail.")
        # Fallback to dummy LLMs if initialization fails
        class DummyLLM:
            def invoke(self, prompt): return type('obj', (object,), {'content': ""})() # Mock response
        llm_fast = llm_final = DummyLLM()


    return docs, embedder, faiss_text_store if faiss_text_retriever else None, faiss_code_store if faiss_code_retriever else None, bm25_retriever, hybrid_retriever, llm_fast, llm_final

class LangGraphRetrievalSystem:
    def __init__(self, k: int = DEFAULT_K, google_api_key: Optional[str] = None):
        # API key management: Prioritize passed key, then env var.
        # The hardcoded key in load_retrieval_components is a last resort per user instruction.
        self.api_key_to_use = google_api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key_to_use:
            print("Warning: No Google API Key provided to LangGraphRetrievalSystem or found in GOOGLE_API_KEY env var.")
            # Fallback to the original hardcoded key inside load_retrieval_components will occur
            # as per user's request not to remove them.

        try:
            (self.docs,
             self.embedder,
             self.faiss_text_store, # Store the FAISS objects themselves
             self.faiss_code_store,
             self.bm25_retriever,
             self.hybrid_retriever,
             self.llm_fast_reranker,
             self.llm_final_reranker) = load_retrieval_components(google_api_key_val=self.api_key_to_use)
        except Exception as e:
            print(f"CRITICAL ERROR during LangGraphRetrievalSystem initialization: {e}")
            # Handle critical failure: e.g., set a flag, use dummy components, or re-raise
            self.docs, self.embedder, self.faiss_text_store, self.faiss_code_store, \
            self.bm25_retriever, self.hybrid_retriever, self.llm_fast_reranker, self.llm_final_reranker = \
                [], None, None, None, None, None, None, None # Or dummy objects
            # Consider raising a custom exception to signal failure to the caller
            # raise RuntimeError(f"Failed to initialize LangGraphRetrievalSystem: {e}")

        self.k = k
        # self.google_api_key is no longer needed here as it's passed to components

    def _detect_intent(self, query: str) -> str:
        """
        Detects user intent from the query.
        Simple keyword-based. For more robust intent detection, consider:
        - Few-shot prompting with an LLM.
        - Training a small classifier.
        - Using more sophisticated NLP techniques.
        """
        q_lower = query.lower()
        # Prioritize more specific keywords
        if any(kw in q_lower for kw in ["how to implement", "example for", "show me code for", "generate code"]):
            return "code_generation_request" # More specific than just "code_example"
        if any(kw in q_lower for kw in ["code for", "example", "build", "create", "implement"]):
            return "code_example"
        if "api" in q_lower or "parameter" in q_lower or "method" in q_lower or "class" in q_lower or "function" in q_lower:
            return "api_reference"
        if "concept" in q_lower or "explain" in q_lower or "overview" in q_lower or "what is" in q_lower:
            return "concept_explanation" # More specific
        # Add more intents as needed, e.g., "comparison", "troubleshooting"
        return "general_query" # Default intent

    def _apply_mmr(self, docs: List[Document], query: str, k: int) -> List[Document]:
        """Applies Maximal Marginal Relevance (MMR) to diversify results."""
        if not docs or k <= 0:
            return []
        if self.embedder is None:
            print("Warning: Embedder not available for MMR. Returning original docs.")
            return docs[:k]
        if len(docs) <= k: # No need for MMR if fewer docs than k
             return docs

        try:
            query_embedding = self.embedder.embed_query(query)
            doc_embeddings = self.embedder.embed_documents([d.page_content for d in docs])
        except Exception as e:
            print(f"Error embedding for MMR: {e}. Returning original docs.")
            return docs[:k]

        selected_indices = []
        candidate_indices = list(range(len(docs)))

        # Add the most relevant document first
        similarities_to_query = cosine_similarity([query_embedding], doc_embeddings)[0]
        
        # Ensure candidate_indices is not empty before using np.argmax
        if not candidate_indices:
            return [] # Should not happen if docs is not empty

        # Find the document most similar to the query among available candidates
        # Need to map similarities_to_query to current candidate_indices
        current_candidate_similarities = similarities_to_query[candidate_indices]
        if not current_candidate_similarities.size: # No candidates left
            return [docs[i] for i in selected_indices]


        best_initial_candidate_idx_in_candidates = np.argmax(current_candidate_similarities)
        first_selection_actual_idx = candidate_indices.pop(best_initial_candidate_idx_in_candidates)
        selected_indices.append(first_selection_actual_idx)


        while len(selected_indices) < k and candidate_indices:
            mmr_scores = []
            # Embeddings of already selected documents
            selected_embeddings = [doc_embeddings[i] for i in selected_indices]

            for cand_idx in candidate_indices:
                sim_to_query = similarities_to_query[cand_idx]
                
                # Similarity to already selected documents
                sim_to_selected = 0
                if selected_embeddings: # Ensure selected_embeddings is not empty
                    # Embedding of the current candidate document
                    cand_embedding = doc_embeddings[cand_idx]
                    # Calculate max similarity between candidate and all selected documents
                    # Need to reshape cand_embedding to be 2D for cosine_similarity
                    max_sim_to_selected = np.max(cosine_similarity([cand_embedding], selected_embeddings))
                    sim_to_selected = max_sim_to_selected
                
                # MMR score calculation
                score = MMR_LAMBDA * sim_to_query - (1 - MMR_LAMBDA) * sim_to_selected
                mmr_scores.append((score, cand_idx))
            
            if not mmr_scores: # No more candidates to score
                break

            # Select the document with the highest MMR score
            mmr_scores.sort(key=lambda x: x[0], reverse=True)
            best_next_actual_idx = mmr_scores[0][1]
            
            selected_indices.append(best_next_actual_idx)
            candidate_indices.remove(best_next_actual_idx)

        return [docs[i] for i in selected_indices]

    def _rerank_with_llm(self, docs: List[Document], query: str, k: int) -> List[Document]:
        """Reranks documents using a two-pass LLM approach."""
        if not docs or k <= 0:
            return []
        if self.llm_fast_reranker is None or self.llm_final_reranker is None:
            print("Warning: Reranking LLMs not available. Returning original docs.")
            return docs[:k]

        # First pass with a faster LLM
        # Consider taking more than 2*k for the initial pool if performance allows
        candidate_docs_pass1 = docs[:min(len(docs), max(k * 2, 10))] # At least 10, or 2*k

        # Improved prompt for reranking, focusing on relevance and actionability for code
        prompt_pass1_parts = [
            f"Query: \"{query}\"\n\n"
            "Review the following document snippets. Identify the top documents that are most relevant and helpful for answering the query, especially if the query is about LangGraph code, concepts, or examples. Respond with a comma-separated list of zero-indexed numbers corresponding to the best documents, in order of relevance (e.g., '2, 0, 5'). Prioritize documents that directly address the query's core needs.\n"
        ]
        for i, d in enumerate(candidate_docs_pass1):
            # Include more context, or specific metadata if helpful (e.g., headers, topic)
            header_info = f" (Section: {d.metadata.get('headers', {}).get('h1', d.metadata.get('subsection', 'N/A'))})" if d.metadata else ""
            prompt_pass1_parts.append(f"[{i}] {d.page_content[:400]}...{header_info}\n") # Increased snippet size

        prompt_pass1 = "\n".join(prompt_pass1_parts)

        try:
            response_pass1 = self.llm_fast_reranker.invoke(prompt_pass1).content
            # Robustly parse indices, handling potential errors or empty responses
            extracted_indices_pass1 = [int(n) for n in re.findall(r'\d+', response_pass1) if int(n) < len(candidate_docs_pass1)]
            if not extracted_indices_pass1 and candidate_docs_pass1: # Fallback if LLM fails to provide indices
                 print("Warning: LLM reranker (pass 1) did not return valid indices. Using original order for first pass.")
                 reranked_docs_pass1 = candidate_docs_pass1[:k] # Or simply take top k
            else:
                 reranked_docs_pass1 = [candidate_docs_pass1[i] for i in extracted_indices_pass1][:k] # Take top K from this pass
        except Exception as e:
            print(f"Error during LLM reranking (pass 1): {e}. Returning MMR/original docs.")
            return docs[:k] # Fallback to pre-reranked list

        if not reranked_docs_pass1: # If first pass yielded nothing
            return docs[:k]

        # Second pass with a more powerful LLM for final ordering
        prompt_pass2_parts = [
            f"User Query: \"{query}\"\n\n"
            "You are refining the order of the following pre-selected documents. Your task is to determine the absolute best order of these documents to answer the user's query. Focus on accuracy, completeness, and direct relevance to LangGraph. Respond with a comma-separated list of zero-indexed numbers (e.g., '1, 0, 2') representing the optimal order.\n"
        ]
        for i, d in enumerate(reranked_docs_pass1):
            header_info = f" (Section: {d.metadata.get('headers', {}).get('h1', d.metadata.get('subsection', 'N/A'))})" if d.metadata else ""
            prompt_pass2_parts.append(f"[{i}] {d.page_content[:500]}...{header_info}\n") # Slightly more context for final pass

        prompt_pass2 = "\n".join(prompt_pass2_parts)

        try:
            response_pass2 = self.llm_final_reranker.invoke(prompt_pass2).content
            extracted_indices_pass2 = [int(n) for n in re.findall(r'\d+', response_pass2) if int(n) < len(reranked_docs_pass1)]
            if not extracted_indices_pass2 and reranked_docs_pass1:
                print("Warning: LLM reranker (pass 2) did not return valid indices. Using pass 1 order.")
                final_reranked_docs = reranked_docs_pass1
            else:
                final_reranked_docs = [reranked_docs_pass1[i] for i in extracted_indices_pass2]
        except Exception as e:
            print(f"Error during LLM reranking (pass 2): {e}. Returning pass 1 docs.")
            return reranked_docs_pass1

        return final_reranked_docs[:k] # Ensure we don't exceed k

    def retrieve(self, query: str, k: Optional[int] = None, rerank_llm: bool = True) -> List[Document]:
        """
        Retrieves relevant documents for a query.
        Steps:
        1. Detect intent.
        2. Apply filters based on intent (if any) using hybrid retriever.
        3. Apply MMR for diversity.
        4. Optionally rerank using LLMs.
        """
        effective_k = k or self.k
        if self.hybrid_retriever is None:
            print("Error: Hybrid retriever not available. Cannot retrieve.")
            return []

        intent = self._detect_intent(query)
        print(f"Detected intent: {intent} for query: '{query}'")

        # Metadata filters based on intent.
        # The effectiveness of these filters depends heavily on the quality of
        # metadata generated by the HybridChunker.
        filters = {}
        # Example: if intent is 'code_example', we might prefer chunks marked as 'is_full_code_example'
        # This requires careful schema design in your chunk metadata.
        # For Langchain's EnsembleRetriever, direct metadata filtering during invoke is not standard.
        # Filtering usually happens by configuring the base retrievers (e.g., FAISS search_kwargs with filter)
        # or by post-filtering the results from the ensemble.
        # Here, we'll assume post-filtering or that base retrievers handle it if configured.

        # For now, we are not applying pre-retrieval filters at the ensemble level directly.
        # If FAISS stores were initialized with filterable metadata, their `as_retriever`
        # calls could include `search_kwargs={'filter': ...}`.
        # Let's assume the ensemble retriever gets a broad set of documents.

        try:
            # The EnsembleRetriever itself doesn't take a 'config' with 'filters' in its standard invoke.
            # We'd typically filter *after* retrieval or configure base retrievers.
            # For this example, let's assume hybrid_retriever.invoke gets all docs.
            retrieved_docs = self.hybrid_retriever.invoke(query)
        except Exception as e:
            print(f"Error during hybrid retrieval: {e}")
            return []

        # Post-retrieval filtering based on intent (example)
        # This is a simple way if base retrievers aren't pre-filtering.
        if intent == "code_example" or intent == "code_generation_request":
            # Prefer documents that are marked as code examples or contain code
            retrieved_docs = [
                doc for doc in retrieved_docs
                if (doc.metadata and (doc.metadata.get("is_full_code_example") or doc.metadata.get("includes_code")))
            ] or retrieved_docs # Fallback to original if filter yields nothing
        elif intent == "api_reference":
            retrieved_docs = [
                doc for doc in retrieved_docs
                if (doc.metadata and doc.metadata.get("topic") == "api_reference")
            ] or retrieved_docs
        elif intent == "concept_explanation":
             retrieved_docs = [
                doc for doc in retrieved_docs
                if (doc.metadata and doc.metadata.get("topic") == "concept")
            ] or retrieved_docs
        
        # Deduplicate documents based on content (simple approach)
        unique_docs_content = set()
        unique_docs = []
        for doc in retrieved_docs:
            if doc.page_content not in unique_docs_content:
                unique_docs.append(doc)
                unique_docs_content.add(doc.page_content)
        retrieved_docs = unique_docs


        # Apply MMR if enough documents are retrieved
        if len(retrieved_docs) > effective_k :
            print(f"Applying MMR to {len(retrieved_docs)} documents for k={effective_k}...")
            try:
                mmr_docs = self._apply_mmr(retrieved_docs, query, effective_k * 2) # Get more for reranker
                print(f"MMR resulted in {len(mmr_docs)} documents.")
            except Exception as e:
                print(f"Error during MMR: {e}. Using pre-MMR docs.")
                mmr_docs = retrieved_docs[:effective_k * 2]
        else:
            mmr_docs = retrieved_docs

        if not mmr_docs:
            print("No documents after MMR/initial retrieval.")
            return []

        # Optionally rerank using LLM
        if rerank_llm:
            print(f"Reranking {len(mmr_docs)} documents with LLM for k={effective_k}...")
            try:
                final_docs = self._rerank_with_llm(mmr_docs, query, effective_k)
                print(f"LLM Reranking resulted in {len(final_docs)} documents.")
            except Exception as e:
                print(f"Error during LLM reranking: {e}. Using pre-reranked docs.")
                final_docs = mmr_docs[:effective_k]
        else:
            final_docs = mmr_docs[:effective_k] # Take top k from MMR/retrieval

        print(f"Final retrieved documents count: {len(final_docs)}")
        return final_docs

# Example Usage (for testing purposes)
# if __name__ == "__main__":
#     print("Testing LangGraphRetrievalSystem...")
#     # Ensure GOOGLE_API_KEY is set in your environment or pass it directly
#     # For this test, it will rely on the fallback mechanism if not set, which is not ideal.
#     retrieval_sys = LangGraphRetrievalSystem(k=5, google_api_key=os.getenv("GOOGLE_API_KEY"))

#     if retrieval_sys.hybrid_retriever is None or retrieval_sys.embedder is None:
#          print("Retrieval system could not be initialized properly. Exiting test.")
#     else:
#         test_query = "How to use StateGraph in LangGraph?"
#         print(f"\nRetrieving for query: '{test_query}'")
        
#         # Test without LLM reranking
#         results_no_rerank = retrieval_sys.retrieve(test_query, k=3, rerank_llm=False)
#         print(f"\nResults (k=3, no LLM rerank, {len(results_no_rerank)} docs):")
#         for i, doc in enumerate(results_no_rerank):
#             print(f"  [{i}] Score: (N/A for this output) Content: {doc.page_content[:150]}...")
#             print(f"      Metadata: {doc.metadata}")

#         # Test with LLM reranking
#         results_with_rerank = retrieval_sys.retrieve(test_query, k=3, rerank_llm=True)
#         print(f"\nResults (k=3, with LLM rerank, {len(results_with_rerank)} docs):")
#         for i, doc in enumerate(results_with_rerank):
#             print(f"  [{i}] Score: (N/A for this output) Content: {doc.page_content[:150]}...")
#             print(f"      Metadata: {doc.metadata}")

#         test_query_code = "Example of a LangGraph chatbot with memory"
#         print(f"\nRetrieving for query: '{test_query_code}'")
#         results_code = retrieval_sys.retrieve(test_query_code, k=2, rerank_llm=True)
#         print(f"\nResults (k=2, code query, with LLM rerank, {len(results_code)} docs):")
#         for i, doc in enumerate(results_code):
#             print(f"  [{i}] Score: (N/A for this output) Content: {doc.page_content[:150]}...")
#             print(f"      Metadata: {doc.metadata}")
