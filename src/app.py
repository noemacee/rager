from flask import Flask, render_template, request, redirect, url_for
from llm import (
    create_text_generator,
    generate_response,
)  # Adjust imports as needed
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

# Initialize your LLM pipeline once (e.g., using your previously defined function)
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
CACHE_DIR = "model_cache"
generator = create_text_generator(MODEL_ID, cache_dir=CACHE_DIR)

# Simple in-memory chat history (for demonstration only)
chat_history = []


@app.route("/", methods=["GET", "POST"])
def chat():
    global chat_history
    if request.method == "POST":
        user_message = request.form.get("message")
        if user_message:
            # Append user message to chat history
            chat_history.append({"role": "user", "content": user_message})

            # Build a simple prompt from the conversation
            # For a more advanced version, you can pass the entire chat history
            prompt_messages = [
                {"role": "system", "content": "You are a helpful chatbot."},
                {"role": "user", "content": user_message},
            ]
            response = generate_response(generator, prompt_messages, max_new_tokens=256)

            # Append the generated answer
            chat_history.append({"role": "assistant", "content": response})

        return redirect(url_for("chat"))

    return render_template("chat.html", chat_history=chat_history)


if __name__ == "__main__":
    app.run(debug=True)
