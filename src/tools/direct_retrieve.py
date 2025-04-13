import json
from langchain.tools import tool
from utils.index import create_pinecone_index


@tool
def direct_retrieval(query: str) -> str:
    """
    Exact keyword-based retrieval, matching metadata if it equals the query.
    Returns structured JSON with context.
    """
    cleaned_query = (
        query.strip("'\"")
        .replace("transaction ", "")
        .replace("merkle root ", "")
        .strip()
    )
    vectorstore = create_pinecone_index("blockchain-index")
    candidate_docs = vectorstore.similarity_search(cleaned_query, k=10)

    for doc in candidate_docs:
        md = doc.metadata
        if (
            md.get("transaction_hash", "") == cleaned_query
            or md.get("merkle_root", "") == cleaned_query
        ):
            return json.dumps(
                {
                    "match_type": "exact",
                    "transaction": md.get("transaction_hash", ""),
                    "merkle_root": md.get("merkle_root", ""),
                    "proof": md.get("proof", []),
                    "status": md.get("status", ""),
                    "block_number": md.get("block_number", ""),
                    "timestamp": md.get("timestamp", ""),
                }
            )

    return json.dumps(
        {"error": f"No exact metadata match found for: '{cleaned_query}'"}
    )
