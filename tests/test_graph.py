import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.router import route_decision


def test_route_decision_billing_goes_to_retrieve():
    state = {"category": "billing"}
    assert route_decision(state) == "retrieve"


def test_route_decision_technical_goes_to_retrieve():
    state = {"category": "technical"}
    assert route_decision(state) == "retrieve"


def test_graph_compiles():
    from app.graph import build_graph
    graph = build_graph()
    assert graph is not None


def test_keyword_fallback_classifier():
    from app.classifier.infer import _keyword_classify
    assert _keyword_classify("my account was hacked, help immediately") == "account"
    assert _keyword_classify("I was charged twice for my subscription") == "billing"
    assert _keyword_classify("the app keeps crashing") == "technical"
    assert _keyword_classify("I need to reset my password") == "account"
