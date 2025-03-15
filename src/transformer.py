import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import pinecone
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
from os import environ as env
from index import (
    embed_text,
    create_pinecone_index,
    populate_index_with_blockchain_data_bulk,
)
from llm import create_text_generator
from rag_pipeline import retrieve_documents_and_rag

load_dotenv()


def main():
    # Model settings.
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    cache_dir = "model_cache"

    # Pinecone settings.
    pinecone_api_key = env.get("PINECONE_API_KEY")
    region = env.get("PINECONE_REGION")  # e.g., "us-west-2"
    index_name = "llama-index"

    # Create the text-generation pipeline.
    generator = create_text_generator(model_id, cache_dir=cache_dir)

    # Create the embedding model.
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Create or load the Pinecone index.
    index = create_pinecone_index(
        index_name, dimension=384, pinecone_api_key=pinecone_api_key, region=region
    )

    # --- Populate the Index with Blockchain Data ---
    transaction_hashes = [
        "0xABC123DEF456",
        "0xDEF789ABC123",
        "0x123456789ABC",
        "0x987654321FED",
    ]
    merkle_roots = [
        "0xABC123DEF456",
        "0xDEF789ABC123",
        "0x123456789ABC",
        "0x987654321FED",
    ]
    populate_index_with_blockchain_data_bulk(
        index, embedder, transaction_hashes, merkle_roots
    )

    # --- RAG: Retrieve and Answer User Query ---
    user_query = input("Enter your query about blockchain data availability: ")
    answer, context = retrieve_documents_and_rag(
        user_query, index, embedder, generator, top_k=1
    )

    print("\n--- Retrieved Context ---")
    print(context)
    print("\n--- Generated Answer ---")
    print(answer)


if __name__ == "__main__":
    main()
