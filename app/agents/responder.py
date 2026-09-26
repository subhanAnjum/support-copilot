"""Generates the customer reply from the retrieved KB chunks."""
from app.config import get_chat_model

SYSTEM_PROMPT = """You are a support agent for FitTrack, a fitness tracking app.
Answer the customer's ticket using ONLY the knowledge base context provided below.

Rules:
- If the context answers the question, give a clear, concise, friendly answer.
- If the context does NOT contain enough information to answer confidently,
  say so explicitly and say the ticket will be escalated to a specialist —
  do NOT invent policy details that aren't in the context.
- Keep the tone warm but concise. No more than 4 sentences.
- Do not mention "the context" or "the knowledge base" to the customer directly.
"""


def respond_node(state: dict) -> dict:
    context = "\n\n---\n\n".join(state.get("retrieved_chunks", []))
    llm = get_chat_model()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"KNOWLEDGE BASE CONTEXT:\n{context}\n\nCUSTOMER TICKET:\n{state['ticket_text']}",
        },
    ]
    response = llm.invoke(messages)

    state["response"] = response.content
    state["sources_cited"] = list(set(state.get("retrieved_sources", [])))
    return state


def escalate_node(state: dict) -> dict:
    """Hands the ticket to a human instead of the LLM. Currently unused."""
    state["response"] = (
        "Thanks for reaching out — this has been flagged as urgent and escalated "
        "directly to our support team, who will contact you shortly. For account "
        "security issues, we also recommend changing your password as a precaution."
    )
    state["sources_cited"] = []
    state["escalated"] = True
    return state
