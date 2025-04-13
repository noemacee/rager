import os
import getpass
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from utils.index import create_pinecone_index, fill_index_with_blockchain_data
from agent import run_agent

load_dotenv()

app = Flask(__name__)


INDEX_NAME = "blockchain-index"
try:
    print("\n🔄 Initializing Pinecone VectorStore...")
    vectorstore = create_pinecone_index(INDEX_NAME)
    print("✅ Pinecone VectorStore initialized!\n")
except Exception as e:
    print(f"❌ Error initializing Pinecone: {e}")
    vectorstore = None

chat_history = []


@app.route("/")
def home():
    return render_template("chat.html", chat_history=chat_history)


@app.route("/get_response", methods=["POST"])
def get_response():
    """Receive user message, call the agent, return the response."""
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message.strip():
        return jsonify({"response": "No query provided."})

    try:
        answer = run_agent(user_message)
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": answer})
        return jsonify({"response": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/clear_chat", methods=["POST"])
def clear_chat():
    """Clear in-memory chat history."""
    global chat_history
    chat_history = []
    return jsonify({"message": "Chat history cleared!"})


@app.route("/populate_index", methods=["POST"])
def populate_index():
    """Fill Pinecone with some sample data."""
    if not vectorstore:
        return jsonify({"error": "VectorStore is not initialized."}), 500

    blockchain_data = [
        {
            "merkle_root": "0xABC123DEF456",
            "transaction_hash": "0xABC123DEF456",
            "proof": "[Sample proof data A]",
            "status": "verified",
            "block_number": 12345,
            "timestamp": 1630000000,
        },
        {
            "merkle_root": "0xDEF789ABC123",
            "transaction_hash": "0xDEF789ABC123",
            "proof": "[Sample proof data B]",
            "status": "pending",
            "block_number": 12346,
            "timestamp": 1630000500,
        },
    ]

    fill_index_with_blockchain_data(vectorstore, blockchain_data)
    return jsonify({"message": "Blockchain data populated successfully!"})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
