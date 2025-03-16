from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline
import torch
from dotenv import load_dotenv

load_dotenv()


def create_langchain_llm(
    model_id="meta-llama/Llama-3.2-3B-Instruct", cache_dir="model_cache"
):
    """
    Loads a Hugging Face model into a LangChain-compatible LLM.
    """
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": device} if device.type == "mps" else "auto",
        cache_dir=cache_dir,
    )

    pipe = hf_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device.type != "mps" else None,
        max_new_tokens=256,  # Moved here
        temperature=0.7,  # Moved here
        top_p=0.9,  # Moved here
    )

    llm = HuggingFacePipeline(pipeline=pipe)  # Pass params inside pipeline, not here

    print("LangChain LLM loaded.\n")
    return llm


from langchain_core.prompts import ChatPromptTemplate


from langchain_core.prompts import ChatPromptTemplate


def generate_response_llm(llm, system_prompt, user_query, document_text):
    """
    Generates a structured response using LangChain-compatible LLM.
    """
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),  # Ensure system instructions are followed
            ("user", f"Query: {user_query}"),  # Only include the user’s question
            (
                "assistant",
                "Respond concisely. Do not repeat details. Only answer the query.",
            ),
            ("human", f"Relevant Blockchain Data:\n{document_text}"),
        ]
    )

    formatted_prompt = prompt_template.format()
    print("🔍 Debugging Prompt Sent to LLM:\n", formatted_prompt)  # Debugging step

    response = llm.invoke(formatted_prompt)  # Ensures structured LLM response
    return response.strip()
