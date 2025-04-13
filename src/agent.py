import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_community.tools.python.tool import PythonREPLTool
from langchain_community.tools.requests.tool import RequestsGetTool
from langchain.tools import tool
from your_smart_contract_tools import verify_merkle_proof

load_dotenv()


# 1. Define your custom verification tool (wrap Cairo smart contract call)
@tool
def verify_merkle_tool(leaf: str, root: str, proof: list[str]) -> str:
    """
    Verify if a Merkle proof is valid for a given leaf and root using a Cairo smart contract.
    Input: leaf (str), root (str), proof (list of hex strings)
    Output: "valid" or "invalid"
    """
    try:
        is_valid = verify_merkle_proof(leaf, root, proof)
        return "valid" if is_valid else "invalid"
    except Exception as e:
        return f"Error: {str(e)}"


# 2. Prepare LangChain-compatible LLM (you already have it)
from transformer import create_langchain_llm  # Assuming your code is in transformer.py

llm = create_langchain_llm()

# 3. Define tools for the agent
tools = [
    Tool(
        name="VerifyMerkleProof",
        func=verify_merkle_tool,
        description="Use to verify a Merkle proof given a leaf, root, and proof list.",
    ),
    PythonREPLTool(),  # optional
    RequestsGetTool(),  # optional for web scraping
]

# 4. Create the agent
agent = initialize_agent(
    tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

# 5. Example usage
response = agent.run(
    "Verify whether leaf 0xabc... belongs to root 0x123... using proof [0xaaa, 0xbbb, 0xccc]"
)
print("Agent Response:", response)
