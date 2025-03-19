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
    tokenizer.pad_token = tokenizer.eos_token

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
        max_new_tokens=50,  # Moved here
        temperature=0.0,  # Moved here
        top_p=1.0,  # Moved here
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,  # Explicitly define padding behavior
    )

    llm = HuggingFacePipeline(pipeline=pipe)  # Pass params inside pipeline, not here

    print("LangChain LLM loaded.\n")
    return llm


from langchain_core.prompts import ChatPromptTemplate


from langchain_core.prompts import ChatPromptTemplate


def generate_response_llm(llm, prompt):
    """
    Generates a structured response using LangChain-compatible LLM.
    """

    response = llm.invoke(prompt, do_sample=False)  # Ensures structured LLM response

    print("Raw Response:", response)

    end_context = response.find("<<<END CONTEXT>>>")
    start_response = response.find("<<<RESPONSE>>>", end_context)
    end_response = response.find("<<<END RESPONSE>>>", start_response)

    if start_response != -1 and end_response != -1:
        answer = response[start_response + len("<<<RESPONSE>>>") : end_response].strip()
    else:
        answer = "ERROR: No valid response found."

    print("Final Answer:", answer)

    return answer


def generate_rag_prompt(query: str, document: str) -> str:
    """
    Generates a structured prompt for the LLaMA-3.2-3B-Instruct model
    using the retrieved Merkle tree data and the user query.

    :param query: The user's query.
    :param metadata: Dictionary containing the retrieved blockchain data.
    :return: Formatted prompt string.
    """

    prompt = f"""
        You are a blockchain expert. Your task is to answer ONLY the query provided by the user using only the context given. 
        Do not include any extra text or restate the context or query.

        Instructions:
        1. Use ONLY the information provided in the context.
        2. If you are not sure or the context does not have enough information, respond with "I don't know."
        3. Output your final answer enclosed between the delimiters <<<RESPONSE>>> and <<<END RESPONSE>>>.
        4. STOP IMMEDIATELY AFTER <<<END RESPONSE>>>. DO NOT GENERATE ANYTHING ELSE.

        Context:
        <<<CONTEXT>>>
        {document}
        <<<END CONTEXT>>>

        Query:
        <<<QUERY>>>
        {query}
        <<<END QUERY>>>
    """
    return prompt.strip()
