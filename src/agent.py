import os
import getpass
from dotenv import load_dotenv

from langchain.agents import initialize_agent, AgentType, Tool
from langchain_openai import ChatOpenAI

from tools.direct_retrieve import direct_retrieval
from tools.fuzzy_retrieve import fuzzy_retrieval
from tools.merkle_verify import verify_merkle_tree

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))


def create_chat_llm():
    """
    Creates a ChatOpenAI model with 'gpt-4o-mini' from OpenAI.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=openai_key,
        temperature=0.0,
    )
    print("Loaded GPT-4o-mini via ChatOpenAI.\n")
    return llm


PREFIX = """You are an AI assistant that uses tools to answer user questions.
Your goal is to always provide a 'Final Answer' after gathering the necessary data.
Prefer calling 'direct_retrieval' if you are looking for a specific transaction hash or Merkle root.
If 'direct_retrieval' returns no results, you may try 'fuzzy_retrieval' to find approximate matches.
After retrieving data with any tool, always use 'verify_merkle_tree' to validate Merkle proofs.

Use the following format exactly:
Question: {input}
Thought: your step-by-step reasoning
Action: [direct_retrieval|fuzzy_retrieval|verify_merkle_tree]
Action Input: the input to the tool
Observation: the result of the tool
... (repeat Thought/Action/Observation if needed) ...
Thought: now you know the final answer
Final Answer: <final answer here>
"""
SUFFIX = """Begin!"""


def run_agent(query: str):
    llm = create_chat_llm()

    tools = [
        Tool(
            name="fuzzy_retrieval",
            func=fuzzy_retrieval,
            description="Fuzzy semantic retrieval over documents.",
        ),
        Tool(
            name="direct_retrieval",
            func=direct_retrieval,
            description="Exact keyword-based retrieval over documents.",
        ),
        Tool(
            name="verify_merkle_tree",
            func=verify_merkle_tree,
            description="Verify Merkle tree correctness using a smart contract.",
        ),
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True,
        agent_kwargs={
            "prefix": PREFIX,
            "suffix": SUFFIX,
        },
    )

    result = agent.run(query)
    return result


if __name__ == "__main__":
    print("Running agent...\n")
    response = run_agent("Verify transaction 0x111")
    print("\nRESULT:\n", response)
