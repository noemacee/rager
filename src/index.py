########################################
# index.py
########################################
import os
from os import environ as env
from pinecone import Pinecone, ServerlessSpec

# Make sure to have `langchain-huggingface` installed
from langchain_pinecone import PineconeVectorStore

from langchain_huggingface import HuggingFaceEmbeddings


import os
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone as LCPinecone
from dotenv import load_dotenv

load_dotenv()


def create_pinecone_index(index_name, dimension=384):
    print("Initializing Pinecone with the new recommended usage...")

    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pinecone_region = os.environ.get("PINECONE_REGION")  # e.g. "us-west-2"

    # Create the new Pinecone client
    pc = Pinecone(api_key=pinecone_api_key)

    # Create index if needed
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
                region=pinecone_region,
            ),
        )
        print("Index created.")
    else:
        print(f"Index '{index_name}' already exists.")

    # Retrieve the Pinecone index
    pinecone_index = pc.Index(index_name)
    print("Pinecone index loaded.\n")

    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = PineconeVectorStore(index=pinecone_index, embedding=embedding)

    return vectorstore


def embed_text(text):
    """
    Embeds text using LangChain's HuggingFace embeddings.
    """
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embedder.embed_query(text)


def fill_index_with_blockchain_data(index, data_items):
    """
    Fills the Pinecone index with blockchain data availability records.
    """
    texts = []
    metadatas = []

    for item in data_items:
        text_for_embedding = (
            f"Merkle Root: {item['merkle_root']}. "
            f"Transaction: {item['transaction_hash']}. "
            f"Proof: {item['proof']}. "
            f"Status: {item['status']}."
            f"Block Number: {item['block_number']}. "
            f"Timestamp: {item['timestamp']}."
        )

        texts.append(text_for_embedding)
        metadatas.append(
            {
                "merkle_root": item["merkle_root"],
                "transaction_hash": item["transaction_hash"],
                "proof": item["proof"],
                "status": item["status"],
                "block_number": item["block_number"],
                "timestamp": item["timestamp"],
            }
        )

        print(f"Prepared vector for transaction {item['transaction_hash']}")

    # 🔥 Use `add_texts()` instead of `upsert()`
    index.add_texts(texts=texts, metadatas=metadatas)

    print("\n✅ Pinecone Index Updated Successfully!\n")
