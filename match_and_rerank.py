# semantic_rerank_chroma.py

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
import os
import yaml
import argparse
import sys
import time
import logging
import json
from typing import List, Dict, Any

# Load configuration from config.yml
# try:
#     with open("config.yml", "r") as f:
#         config = yaml.safe_load(f)
# except Exception as e:
#     print(f"Error loading config.yml: {e}")
#     sys.exit(1)
# Direct assignment without reading from config
CHROMA_PERSIST_DIR = "chroma_persist"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K = 30
RERANK_TOP = 10
DEFAULT_ALLOWED_EMAILS = []
COLLECTION_NAME = "candidates_collection"


# Retry decorator
def retry(func):
    def wrapper(*args, **kwargs):
        max_attempts = 3
        initial_delay = 1.0  # seconds
        backoff_factor = 2.0
        for attempt in range(1, max_attempts + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_attempts:
                    print(f"Operation failed after {max_attempts} attempts: {e}")
                    raise
                sleep_time = initial_delay * (backoff_factor ** (attempt - 1))
                print(f"Attempt {attempt} failed: {e}. Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
    return wrapper

# Initialize embeddings and vectorstore globally
embedding_fn = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
chroma_db = retry(lambda: Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_function=embedding_fn,
    collection_name=COLLECTION_NAME
))()

# Initialize the cross-encoder for reranking
cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

# 5) DEFINE A RERANK FUNCTION
# ----------------------------

def rerank_documents(query: str, docs: list):
    """
    Given a query and a list of langchain.Document objects,
    run the cross-encoder to get a more precise relevance score,
    then return the docs sorted by that score (highest first).
    """
    # Return early if no documents to rerank
    if not docs:
        return []
    # Prepare pairs: [[query, doc1_text], [query, doc2_text], ...]
    pairs = [[query, doc.page_content] for doc in docs]

    # cross_encoder.predict returns a list of floats (one score per pair)
    scores = cross_encoder.predict(pairs)

    # Pair each doc with its score, sort descending
    doc_score_pairs = list(zip(docs, scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

    # Return only the sorted documents (dropping scores)
    reranked = [doc for doc, _score in doc_score_pairs]
    return reranked

# ─── READY FUNCTION FOR IMPORT ────────────────────────────────────────────────
def semantic_search_and_rerank(candidates: List[Dict[str, Any]], query: str,
                               top_k: int = TOP_K, rerank_top: int = RERANK_TOP) -> List[Dict[str, Any]]:
    """
    Perform email-filtered semantic search and cross-encoder reranking.
    """
    # Extract valid emails
    print(f"Extracting emails from {len(candidates)} candidates...")
    print("these are the candidates", candidates)
    emails = [c['email'] for c in candidates if c.get('email') and c.get('email') != 'NA']
    print(f"Found {len(emails)} valid emails in candidates.")
    if not emails:
        print("No valid emails found in candidates. Returning empty list.")
        return []
    
    # Adjust k to avoid errors
    k = min(top_k, len(emails))
    filter_arg = {"email": {"$in": emails}}
    # Semantic search
    print(f"Performing semantic search with k={k} and filter={filter_arg}")
    retriever = chroma_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k, "filter": filter_arg}
    )
    print(f"Retriever initialized with {k} top results and email filter.")
    docs_found = retry(retriever.invoke)(query)
    print(f"Retrieved {len(docs_found)} documents from semantic search.")
    if not docs_found:
        print("No documents found after semantic search. Returning empty list.")
        return []
    # Rerank retrieved docs
    reranked_docs = retry(lambda q, ds: rerank_documents(q, ds))(query, docs_found)
    print(f"Reranked {len(reranked_docs)} documents based on query: '{query}'")
    selected_emails = [doc.metadata.get('email') for doc in reranked_docs[:rerank_top]]
    print(f"Selected top {rerank_top} emails after reranking: {selected_emails}")
    # Map back to candidate dicts
    email_map = {c['email']: c for c in candidates}
    print(f"Mapping emails back to candidates, found {len(email_map)} candidates.")
    return [email_map[email] for email in selected_emails if email in email_map]

# 6) EXAMPLE USAGE
# ----------------

if __name__ == "__main__":
    # Use CLI-provided query
    # query_text = args.query
    # print(f"Retrieving top {TOP_K} documents for query: '{query_text}'")
    # try:
    #     candidate_docs = retry(retriever.invoke)(query_text)
    # except Exception as e:
    #     print(f"Error retrieving documents: {e}")
    #     sys.exit(1)

    # # If no docs remain after filtering, exit
    # if not candidate_docs:
    #     print("No documents found after applying email filter.")
    #     sys.exit(0)

    # # Stage 2: Rerank those documents
    # print(f"Retrieved {len(candidate_docs)} candidate documents, now reranking top {len(candidate_docs)}...")
    # try:
    #     final_docs = retry(rerank_documents)(query_text, candidate_docs)
    # except Exception as e:
    #     print(f"Error during reranking: {e}")
    #     sys.exit(1)

    # # Output the top reranked JSON results
    # result_jsons = [doc.metadata for doc in final_docs[:RERANK_TOP]]
    # print(json.dumps(result_jsons, indent=2))

    #import D:\Resume_Screening\backend\utils\data.json 
    with open('D:\\Resume_Screening\\backend\\utils\\.json', 'r') as f:
        candidates = json.load(f)  # Expecting a list of dictionaries
    
    
    
    #call the semantic_search_and_rerank function
    query_text = "Python developer with experience in machine learning"
    candidates = semantic_search_and_rerank(candidates, query_text, top_k=TOP_K, rerank_top=RERANK_TOP)
    
    #save the results to a file
    with open('D:\\Resume_Screening\\backend\\utils\\reranked_candidates.json', 'w') as f:
        json.dump(candidates, f, indent=4)
    
    # print(type(candidates))
    # print(candidates)