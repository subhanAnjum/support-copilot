"""
Ticket pipeline: route -> retrieve -> respond.

The escalate branch is disabled until urgency detection is back.
"""
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END

from app.agents.router import route_node, route_decision
from app.agents.retriever import retrieve_node
from app.agents.responder import respond_node


class TicketState(TypedDict, total=False):
    ticket_text: str
    category: str
    classifier_confidence: Optional[float]
    classifier_method: str
    retrieved_chunks: List[str]
    retrieved_sources: List[str]
    response: str
    sources_cited: List[str]
    escalated: bool


def build_graph():
    graph = StateGraph(TicketState)

    graph.add_node("route", route_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("respond", respond_node)

    graph.set_entry_point("route")

    graph.add_conditional_edges(
        "route",
        route_decision,
        {"retrieve": "retrieve"},
    )
    graph.add_edge("retrieve", "respond")
    graph.add_edge("respond", END)

    return graph.compile()


support_graph = build_graph()


def run_ticket(ticket_text: str) -> dict:
    initial_state: TicketState = {"ticket_text": ticket_text}
    return support_graph.invoke(initial_state)


if __name__ == "__main__":
    import json
    result = run_ticket("I was charged twice for my premium subscription this month")
    print(json.dumps(result, indent=2))
