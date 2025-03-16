import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import pinecone
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
from os import environ as env

load_dotenv()


def create_text_generator(model_id, cache_dir="model_cache"):
    """
    Loads the model and tokenizer using a cache directory and
    returns a text-generation pipeline.
    """
    print("Loading text-generation model and tokenizer...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": device} if device.type == "mps" else "auto",
        cache_dir=cache_dir,
    )
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    print("Text-generation model loaded.\n")
    return generator


def generate_response(
    generator, messages, max_new_tokens=256, temperature=0.7, top_p=0.9
):
    """
    Generates a text response given conversation messages.
    Prints the prompt, raw output, and extracts the answer.
    Handles both string and list outputs.
    """
    print("Generating response with prompt:")
    for m in messages:
        print(f"  {m['role']}: {m['content']}")

    outputs = generator(
        messages, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p
    )
    print("\nRaw generator output:")
    print(outputs)

    # Attempt to extract the generated text.
    if isinstance(outputs, list) and outputs:
        raw_text = outputs[0].get("generated_text")
        # Case 1: raw_text is a string.
        if isinstance(raw_text, str):
            if "Answer:" in raw_text:
                answer = raw_text.split("Answer:", 1)[1].strip()
            else:
                answer = raw_text.strip()
            print("\nExtracted Answer:")
            print(answer)
            return answer
        # Case 2: raw_text is a list (e.g., list of messages).
        elif isinstance(raw_text, list):
            for msg in raw_text:
                if msg.get("role") == "assistant":
                    answer = msg.get("content", "").strip()
                    print("\nExtracted Answer:")
                    print(answer)
                    return answer
            # Fallback: join all messages' content.
            combined = " ".join(msg.get("content", "") for msg in raw_text)
            print("\nExtracted Answer by concatenation:")
            print(combined)
            return combined
    else:
        print("No output generated.")
        return ""
