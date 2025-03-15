import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import pinecone
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
from os import environ as env
from index import embed_text
from llm import generate_response

load_dotenv()


def retrieve_documents_and_rag(query, index, embedder, generator, top_k=1):
    """
    Takes a user query, retrieves the most significant blockchain record from the index,
    and uses a RAG pipeline to produce an answer.
    """
    print(f"\nProcessing user query: '{query}'")
    query_embedding = embed_text(query, embedder)

    print("Querying Pinecone index...")
    query_response = index.query(
        vector=query_embedding, top_k=top_k, include_metadata=True
    )
    print("Raw query response from Pinecone:", query_response)

    if not query_response.get("matches"):
        print("No matching documents found.")
        return "No relevant blockchain data found.", ""

    # Get the single most relevant match.
    match = query_response["matches"][0]
    metadata = match.get("metadata", {})
    document = (
        f"Merkle Root: {metadata.get('merkle_root', '')}. "
        f"Transaction: {metadata.get('transaction_hash', '')}. "
        f"Proof: {metadata.get('proof', '')}. "
        f"Status: {metadata.get('status', '')}."
        f"Block Number: {metadata.get('block_number', '')}."
        f"Timestamp: {metadata.get('timestamp', '')}."
    )
    print("Most significant document retrieved:", document)

    prompt = (
        "You are an expert blockchain verifier. Given the following blockchain data, "
        "please answer the user's query in detail.\n\n"
        "Blockchain Data:\n" + document + "\n\n"
        "User Query: " + query + "\n\n"
        "Answer:"
    )
    messages = [{"role": "system", "content": prompt}]
    print("\nFinal prompt sent to language model:")
    print(prompt)

    response_text = generate_response(
        generator, messages, max_new_tokens=256, temperature=0.7, top_p=0.9
    )
    print("Final generated response obtained.\n")
    return response_text, document
