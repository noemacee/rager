from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document


def build_fuzzy_context(docs: list[Document]) -> str:
    """
    Summarizes the top doc's metadata as a context string.
    """
    if not docs:
        return "No docs found."
    md = docs[0].metadata
    context_str = (
        f"Merkle Root: {md.get('merkle_root','')}\n"
        f"Transaction: {md.get('transaction_hash','')}\n"
        f"Proof: {md.get('proof','')}\n"
        f"Status: {md.get('status','')}\n"
        f"Block Number: {md.get('block_number','')}\n"
        f"Timestamp: {md.get('timestamp','')}\n"
    )
    return context_str


def retrieve_top_docs(query: str, vectorstore, top_k: int = 3) -> list[Document]:
    """
    Retrieve top_k docs from the vectorstore.
    """
    retriever = VectorStoreRetriever(
        vectorstore=vectorstore, search_type="similarity", search_kwargs={"k": top_k}
    )
    docs = retriever.invoke(query)
    return docs
