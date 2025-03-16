from flask import Flask, render_template, request, redirect, url_for
from llm import (
    create_text_generator,
    generate_response,
)  # Adjust imports as needed
from dotenv import load_dotenv
import os
from flask import jsonify, request

load_dotenv()

app = Flask(__name__)

# Initialize your LLM pipeline once (e.g., using your previously defined function)
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
CACHE_DIR = "model_cache"
generator = create_text_generator(MODEL_ID, cache_dir=CACHE_DIR)

# Simple in-memory chat history (for demonstration only)
chat_history = []


@app.route("/")
def home():
    global chat_history
    return render_template("chat.html", chat_history=chat_history)


@app.route("/get_response", methods=["POST"])
def get_response():
    data = request.get_json()
    user_message = data.get("message")

    # Build your prompt or use the conversation
    prompt_messages = [
        {"role": "system", "content": "You are a helpful chatbot."},
        {"role": "user", "content": user_message},
    ]
    # Generate response with your LLM
    response = generate_response(generator, prompt_messages, max_new_tokens=256)

    # Optionally store in chat_history
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": response})

    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=True)
