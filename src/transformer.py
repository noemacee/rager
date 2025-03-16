import torch
from dotenv import load_dotenv
import os
from os import environ as env
from llm import create_langchain_llm
from index import create_pinecone_index, fill_index_with_blockchain_data
from rag_pipeline import retrieve_documents_and_rag

load_dotenv()


def main():
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    index_name = "llama-index"

    print("\nLoading LangChain LLM...")
    llm = create_langchain_llm(model_id)

    print("\nInitializing Pinecone index...")
    vectorstore = create_pinecone_index(index_name)

    # --- 🔥 Populate the index BEFORE querying ---
    print("\nPopulating Pinecone index with blockchain data...")

    blockchain_data = [
        {
            "merkle_root": "0xABC123DEF456",
            "transaction_hash": "0xABC123DEF456",
            "proof": "Simulated_Cairo_Proof_for_0xABC123DEF456",
            "status": "verified",
            "block_number": 12345,
            "timestamp": 1630000000,
        },
        {
            "merkle_root": "0xDEF789ABC123",
            "transaction_hash": "0xDEF789ABC123",
            "proof": "Simulated_Cairo_Proof_for_0xDEF789ABC123",
            "status": "pending",
            "block_number": 12346,
            "timestamp": 1630001000,
        },
    ]

    # fill_index_with_blockchain_data(vectorstore, blockchain_data)
    # print("\nBlockchain data added to Pinecone!")

    # --- 🔥 Now we can safely query ---
    user_query = input("\nEnter your blockchain query: ")
    answer, context = retrieve_documents_and_rag(user_query, vectorstore, llm, top_k=1)

    print("\n--- Retrieved Context ---")
    print(context)
    print("\n--- Generated Answer ---")
    print(answer)


if __name__ == "__main__":
    main()
