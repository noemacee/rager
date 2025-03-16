from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document
from llm import generate_response_llm

from dotenv import load_dotenv

load_dotenv()


def retrieve_documents_and_rag(query, vectorstore, llm, top_k=1):
    """
    Uses LangChain's retriever to fetch blockchain data and generate an answer.
    """
    print(f"\nProcessing user query: '{query}'")

    retriever = VectorStoreRetriever(
        vectorstore=vectorstore, search_type="similarity", search_kwargs={"k": top_k}
    )
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant blockchain data found.", ""

    # Process the best match
    best_match = docs[0]
    metadata = best_match.metadata

    document = (
        f"Merkle Root: {metadata.get('merkle_root', '')}. "
        f"Transaction: {metadata.get('transaction_hash', '')}. "
        f"Proof: {metadata.get('proof', '')}. "
        f"Status: {metadata.get('status', '')}. "
        f"Block Number: {metadata.get('block_number', '')}. "
        f"Timestamp: {metadata.get('timestamp', '')}."
    )

    system_prompt = (
        "You are a blockchain expert. Respond ONLY to the user's query. "
        "DO NOT explain concepts unless asked. "
        "DO NOT repeat blockchain data. "
        "If the user asks for a Merkle root, return ONLY the root."
    )

    response_text = generate_response_llm(llm, system_prompt, query, document)

    return response_text, document
