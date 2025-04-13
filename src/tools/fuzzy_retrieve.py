import os
import json
from langchain.tools import tool
from utils.index import create_pinecone_index
from utils.rag import build_fuzzy_context, retrieve_top_docs


@tool
def fuzzy_retrieval(query: str) -> str:
    """
    Fuzzy semantic retrieval using your pinecone-based RAG approach.
    Returns structured JSON with context.
    """
    vectorstore = create_pinecone_index("blockchain-index")
    docs = retrieve_top_docs(query, vectorstore, top_k=3)
    if not docs:
        return json.dumps({"error": f"No relevant data found for '{query}'"})

    top = docs[0].metadata
    result = {
        "match_type": "fuzzy",
        "transaction": top.get("transaction_hash", ""),
        "merkle_root": top.get("merkle_root", ""),
        "proof": top.get("proof", []),
        "status": top.get("status", ""),
        "block_number": top.get("block_number", ""),
        "timestamp": top.get("timestamp", ""),
    }
    return json.dumps(result)
