import os
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone as LCPinecone
from langchain_pinecone import PineconeVectorStore

load_dotenv()


def create_pinecone_index(index_name="blockchain-index", dimension=384):
    """
    Creates or retrieves an existing Pinecone index,
    attaches a LangChain-compatible vector store, and returns it.
    """
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pinecone_region = os.environ.get("PINECONE_REGION")  # e.g. "us-west1-gcp"
    print("Initializing Pinecone with recommended usage...")

    pc = Pinecone(api_key=pinecone_api_key)
    existing_indexes = pc.list_indexes().names()
    print(f"Existing indexes: {existing_indexes}")

    if index_name not in existing_indexes:
        print(f"Index '{index_name}' not found. Creating index...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=pinecone_region),
        )
        print("Index created.")
    else:
        print(f"Index '{index_name}' already exists.")

    # Connect to the Pinecone index
    pinecone_index = pc.Index(index_name)
    print(f"Pinecone index '{index_name}' loaded.\n")

    # Create embeddings & vector store
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = PineconeVectorStore(index=pinecone_index, embedding=embedding)
    return vectorstore


def fill_index_with_blockchain_data(vectorstore, data_items):
    """
    Fills the Pinecone vector store with blockchain data availability records.
    """
    texts = []
    metadatas = []

    for item in data_items:
        text_for_embedding = (
            f"Merkle Root: {item['merkle_root']}. "
            f"Transaction: {item['transaction_hash']}. "
            f"Proof: {item['proof']}. "
            f"Status: {item['status']}. "
            f"Block Number: {item['block_number']}. "
            f"Timestamp: {item['timestamp']}."
        )

        metadata = {
            "merkle_root": item["merkle_root"],
            "transaction_hash": item["transaction_hash"],
            "proof": item["proof"],
            "status": item["status"],
            "block_number": item["block_number"],
            "timestamp": item["timestamp"],
        }

        texts.append(text_for_embedding)
        metadatas.append(metadata)

        print(f"Prepared vector for transaction {item['transaction_hash']}")

    vectorstore.add_texts(texts=texts, metadatas=metadatas)
    print("\n✅ Pinecone Index Updated Successfully!\n")


if __name__ == "__main__":
    # Example usage: fill 'blockchain-index' with some sample data
    vectorstore = create_pinecone_index(index_name="blockchain-index", dimension=384)

    data_items = [
        {
            "merkle_root": "0xabc123",
            "transaction_hash": "0x111",
            "proof": "[0xaa, 0xbb, 0xcc]",
            "status": "final",
            "block_number": "12345",
            "timestamp": "2025-04-15T12:00:00Z",
        },
        {
            "merkle_root": "0xdef456",
            "transaction_hash": "0x222",
            "proof": "[0xdd, 0xee, 0xff]",
            "status": "pending",
            "block_number": "23456",
            "timestamp": "2025-04-16T09:30:00Z",
        },
        {
            "merkle_root": "0x333333",
            "transaction_hash": "0x333",
            "proof": "[0x33, 0x44, 0x55]",
            "status": "final",
            "block_number": "34567",
            "timestamp": "2025-04-17T15:45:00Z",
        },
    ]

    fill_index_with_blockchain_data(vectorstore, data_items)
    print("Done populating the Pinecone index with dummy data!\n")
