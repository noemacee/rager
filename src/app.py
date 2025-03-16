from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
from llm import create_langchain_llm
from index import create_pinecone_index, fill_index_with_blockchain_data
from rag_pipeline import retrieve_documents_and_rag

# Load environment variables
load_dotenv()

# Flask app setup
app = Flask(__name__)

# 🔥 Initialize LangChain LLM and Pinecone vector store
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
INDEX_NAME = "llama-index"

try:
    print("\n🔄 Initializing Pinecone VectorStore...")
    vectorstore = create_pinecone_index(INDEX_NAME)
    print("✅ Pinecone VectorStore initialized!\n")
except Exception as e:
    print(f"❌ Error initializing Pinecone: {e}")
    vectorstore = None

try:
    print("\n🔄 Loading LangChain LLM...")
    llm = create_langchain_llm(MODEL_ID)
    print("✅ LangChain LLM loaded!\n")
except Exception as e:
    print(f"❌ Error loading LLM: {e}")
    llm = None

# Simple in-memory chat history
chat_history = []


@app.route("/")
def home():
    return render_template("chat.html", chat_history=chat_history)


@app.route("/get_response", methods=["POST"])
def get_response():
    if not vectorstore or not llm:
        return jsonify({"error": "LLM or Vector Store is not initialized."}), 500

    data = request.get_json()
    user_message = data.get("message", "")

    # 🔥 Retrieve relevant blockchain data & generate response
    answer, _ = retrieve_documents_and_rag(user_message, vectorstore, llm, top_k=1)

    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": answer})

    return jsonify({"response": answer})


@app.route("/clear_chat", methods=["POST"])
def clear_chat():
    global chat_history
    chat_history = []
    return jsonify({"message": "Chat history cleared!"})


@app.route("/populate_index", methods=["POST"])
def populate_index():
    if not vectorstore:
        return jsonify({"error": "VectorStore is not initialized."}), 500

    # 🔥 Sample blockchain data (modify as needed)
    blockchain_data = [
        {
            "merkle_root": "0xABC123DEF456",
            "transaction_hash": "0xABC123DEF456",
            "proof": "Sample proof data",
            "status": "verified",
            "block_number": 12345,
            "timestamp": 1630000000,
        },
        {
            "merkle_root": "0xDEF789ABC123",
            "transaction_hash": "0xDEF789ABC123",
            "proof": "Another proof",
            "status": "pending",
            "block_number": 12346,
            "timestamp": 1630000500,
        },
    ]

    fill_index_with_blockchain_data(vectorstore, blockchain_data)
    return jsonify({"message": "Blockchain data populated successfully!"})


if __name__ == "__main__":
    app.run(debug=True)
