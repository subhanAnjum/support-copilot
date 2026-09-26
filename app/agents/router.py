"""Classifies the ticket and picks the next node."""
from app.classifier.infer import classify_ticket


def route_node(state: dict) -> dict:
    result = classify_ticket(state["ticket_text"])
    state["category"] = result["category"]
    state["classifier_confidence"] = result["confidence"]
    state["classifier_method"] = result["method"]
    return state


def route_decision(state: dict) -> str:
    # TODO: return "escalate" for urgent tickets once urgency detection is back
    return "retrieve"
