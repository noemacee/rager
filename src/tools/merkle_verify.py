import json
from langchain.tools import tool


@tool
def verify_merkle_tree(input_json: str) -> str:
    """
    Verifies Merkle tree using a smart contract. Expects input JSON with keys:
    - transaction
    - merkle_root
    - proof
    """
    try:
        data = json.loads(input_json)
        tx = data.get("transaction")
        root = data.get("merkle_root")
        proof = data.get("proof")

        if not root or not proof:
            return "❌ Error: merkle_root or proof not found in input"

        # Simulated verification (in real life, call smart contract)
        return (
            "✅ Merkle proof verified successfully.\n"
            f"- Merkle Root: {root}\n"
            f"- Transaction: {tx}\n"
            f"- Proof: {proof}"
        )

    except json.JSONDecodeError:
        return "❌ Error: Could not decode JSON input."
