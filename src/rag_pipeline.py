from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document
from llm import generate_response_llm, generate_rag_prompt

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
        f"The Merkle Root is{metadata.get('merkle_root', '')}. "
        f"The Transaction id is {metadata.get('transaction_hash', '')}. "
        f"The Proof associated to the transaction in the root is {metadata.get('proof', '')}. "
        f"The Status of the transaction of the proof {metadata.get('status', '')}. "
        f"The Block Number associated to the merkle root {metadata.get('block_number', '')}. "
        f"The Timestamp associated to the merkle root {metadata.get('timestamp', '')}."
    )

    prompt = generate_rag_prompt(query, document)

    response_text = generate_response_llm(llm, prompt)

    return response_text, document
