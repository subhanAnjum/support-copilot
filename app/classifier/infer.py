"""
Ticket classification with the LoRA adapter. Falls back to keyword matching
if no adapter has been trained yet.
"""
import os
import torch
from functools import lru_cache

from app.config import CLASSIFIER_BASE_MODEL, CLASSIFIER_ADAPTER_DIR, TICKET_CATEGORIES

_KEYWORD_FALLBACK = {
    "billing": ["charged", "refund", "subscription", "billed", "payment", "cancel"],
    "technical": ["crash", "sync", "bug", "not working", "loading", "freeze", "notification"],
    "account": ["password", "email", "delete my account", "merge", "login", "hacked", "locked out"],
}


def _keyword_classify(text: str) -> str:
    lower = text.lower()
    for category, keywords in _KEYWORD_FALLBACK.items():
        if any(kw in lower for kw in keywords):
            return category
    return "technical"


@lru_cache(maxsize=1)
def _load_model():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_ADAPTER_DIR)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        CLASSIFIER_BASE_MODEL, num_labels=len(TICKET_CATEGORIES)
    )
    model = PeftModel.from_pretrained(base_model, CLASSIFIER_ADAPTER_DIR)
    model.eval()
    return tokenizer, model


def classify_ticket(text: str) -> dict:
    """Returns {"category": str, "confidence": float, "method": "lora"|"keyword_fallback"}"""
    if not os.path.exists(CLASSIFIER_ADAPTER_DIR):
        return {"category": _keyword_classify(text), "confidence": None, "method": "keyword_fallback"}

    tokenizer, model = _load_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    pred_id = int(torch.argmax(probs))
    return {
        "category": TICKET_CATEGORIES[pred_id],
        "confidence": round(float(probs[pred_id]), 3),
        "method": "lora",
    }


if __name__ == "__main__":
    samples = [
        "I was charged twice this month for premium",
        "My account was hacked, please help immediately",
        "Workouts aren't syncing between my devices",
    ]
    for s in samples:
        print(s, "->", classify_ticket(s))
