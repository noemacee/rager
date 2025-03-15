import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import pinecone
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
from os import environ as env

load_dotenv()


def create_pinecone_index(index_name, dimension, pinecone_api_key, region):
    """
    Initializes Pinecone using the new API and creates an index if it does not exist.
    """
    print("Initializing Pinecone and creating/loading index...")
    pc = Pinecone(api_key=pinecone_api_key)
    existing_indexes = pc.list_indexes().names()
    print(f"Existing indexes: {existing_indexes}")
    if index_name not in existing_indexes:
        print(f"Index '{index_name}' not found. Creating index...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=region,
            ),
        )
        print("Index created.")
    else:
        print(f"Index '{index_name}' already exists.")
    index = pc.Index(index_name)
    print("Pinecone index loaded.\n")
    return index


def embed_text(text, embedder):
    """
    Converts a piece of text into an embedding vector.
    """
    if not isinstance(text, str):
        text = str(text)
    vector = embedder.encode(text).tolist()
    print(f"Embedded text: '{text[:50]}...' to vector of length {len(vector)}")
    return vector


def simulate_blockchain_data_availability(transaction_hash, merkle_root):
    """
    Simulate retrieval of blockchain data availability proof.
    """
    simulated_proof = f"Simulated_Cairo_Proof_for_{transaction_hash}"
    data = {
        "merkle_root": merkle_root,
        "transaction_hash": transaction_hash,
        "proof": simulated_proof,
        "status": "verified",
        "block_number": 12345,
        "timestamp": 1630000000,
    }
    print(f"Simulated blockchain data for {transaction_hash}: {data}")
    return data


def fill_index_with_blockchain_data(index, embedder, data_items):
    """
    Fills the Pinecone index with blockchain data availability records.
    """
    vectors = []
    for item in data_items:
        text_for_embedding = (
            f"Merkle Root: {item['merkle_root']}. "
            f"Transaction: {item['transaction_hash']}. "
            f"Proof: {item['proof']}. "
            f"Status: {item['status']}."
            f"Block Number: {item['block_number']}."
            f"Timestamp: {item['timestamp']}."
        )
        vector = embed_text(text_for_embedding, embedder)
        vectors.append(
            {
                "id": item["merkle_root"],
                "values": vector,
                "metadata": item,
            }
        )
        print(f"Prepared vector for transaction {item['merkle_root']}")
    upsert_response = index.upsert(vectors=vectors)
    print("Pinecone Upsert Response:", upsert_response, "\n")


def populate_index_with_blockchain_data_bulk(
    index, embedder, transaction_hashes, merkle_roots
):
    """
    Populates the Pinecone index with blockchain data availability records
    for a list of transaction hashes.
    """
    print("Populating index with blockchain data...")
    data_items = []
    for tx_hash in transaction_hashes:
        for merkle_root in merkle_roots:
            blockchain_data = simulate_blockchain_data_availability(
                tx_hash, merkle_root
            )
            data_items.append(blockchain_data)
    fill_index_with_blockchain_data(index, embedder, data_items)
    print("Index population complete.\n")
