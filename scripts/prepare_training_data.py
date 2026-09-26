"""
Builds data/tickets/public_tickets.jsonl from public HF datasets:

  - mteb/banking77 (CC-BY-4.0)
  - Tobi-Bueck/customer-support-tickets (CC-BY-NC-4.0, English rows only)
  - bitext/Bitext-customer-support-llm-chatbot-training-dataset (CDLA-Sharing-1.0)

Tobi-Bueck emails are cut down to their first sentence, otherwise the model
learns text length instead of topic. Run scripts/check_training_data.py
afterwards.

Usage: python scripts/prepare_training_data.py [--per-class 1500]
"""
import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset

from app.config import PUBLIC_TICKETS_PATH, TICKET_CATEGORIES, TICKETS_PATH

TOBI_TECHNICAL_QUEUES = {
    "Technical Support",
    "Product Support",
    "IT Support",
    "Service Outages and Maintenance",
}
TOBI_BILLING_QUEUES = {"Billing and Payments"}
# "critical" priority only exists in German rows, so urgent = high-priority
# English incidents with security / data loss tags
TOBI_URGENT_TAGS = {
    "security", "data breach", "breach", "cybersecurity", "malware", "virus",
    "data leak", "dataleak", "data exposure", "unauthorized",
    "unauthorized access", "cyberattack", "data loss", "ransomware",
    "phishing", "hacked", "fraud",
}

# Login tickets are filed under technical queues but are "account" for us.
# Too mixed to relabel, so they're skipped.
TOBI_LOGIN_TAGS = {
    "login", "password", "password reset", "authentication", "sign-in",
    "credentials", "two-factor authentication", "multifactorauthentication",
    "2fa", "account access", "access control",
}

# Only intents that make sense for a fitness subscription app.
BANKING77_INTENT_MAP = {
    # billing
    "transaction_charged_twice": "billing",
    "request_refund": "billing",
    "Refund_not_showing_up": "billing",
    "extra_charge_on_statement": "billing",
    "card_payment_fee_charged": "billing",
    # account
    "passcode_forgotten": "account",
    "edit_personal_details": "account",
    "terminate_account": "account",
    "unable_to_verify_identity": "account",
    # technical
    "balance_not_updated_after_bank_transfer": "technical",
    "balance_not_updated_after_cheque_or_cash_deposit": "technical",
    "virtual_card_not_working": "technical",
    "card_not_working": "technical",
    # urgent
    "compromised_card": "urgent",
    "card_payment_not_recognised": "urgent",
    "cash_withdrawal_not_recognised": "urgent",
    "lost_or_stolen_phone": "urgent",
}

BITEXT_INTENT_MAP = {
    # billing
    "check_payment_methods": "billing",
    "payment_issue": "billing",
    "get_invoice": "billing",
    "check_invoice": "billing",
    "get_refund": "billing",
    "check_refund_policy": "billing",
    "track_refund": "billing",
    "check_cancellation_fee": "billing",
    # account
    "recover_password": "account",
    "edit_account": "account",
    "delete_account": "account",
    "create_account": "account",
    "switch_account": "account",
    "registration_problems": "account",
}


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


_GREETING = re.compile(r"^(dear|hello|hi|hey|greetings|to whom)[^,\n]*[,\n]", re.I)
_FILLER = re.compile(r"^(i hope|i trust|thank you|thanks)", re.I)
MAX_WORDS = 20


def _shorten_email(subject, body) -> str:
    """Returns the first real sentence of the email (subject as fallback)."""
    body = _GREETING.sub("", _clean(body)).strip()
    sentences = [x for x in re.split(r"(?<=[.!?])\s+", body) if x and not _FILLER.match(x)]
    text = sentences[0] if sentences else _clean(subject)
    return " ".join(text.split()[:MAX_WORDS])


def load_tobi():
    ds = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
    rows = []
    for r in ds:
        if r["language"] != "en" or not r["body"]:
            continue
        tags = {(r[f"tag_{i}"] or "").strip().lower() for i in range(1, 9)}
        security_related = bool(tags & TOBI_URGENT_TAGS)
        if r["priority"] == "high" and r["type"] == "Incident" and security_related:
            category = "urgent"
        elif security_related or tags & TOBI_LOGIN_TAGS:
            continue  # ambiguous
        elif r["queue"] in TOBI_BILLING_QUEUES:
            category = "billing"
        elif r["queue"] in TOBI_TECHNICAL_QUEUES and r["type"] in ("Incident", "Problem"):
            # Request/Change tickets are mostly how-to questions
            category = "technical"
        else:
            continue
        text = _shorten_email(r["subject"], r["body"])
        if len(text.split()) < 4:
            continue
        rows.append({"text": text, "category": category, "source": "tobi-bueck"})
    return rows


def load_banking77():
    rows = []
    for split in ("train", "test"):
        for r in load_dataset("mteb/banking77", split=split):
            category = BANKING77_INTENT_MAP.get(r["label_text"])
            if category is not None:
                rows.append({"text": _clean(r["text"]), "category": category, "source": "banking77"})
    return rows


def load_bitext():
    ds = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train"
    )
    rows = []
    for r in ds:
        category = BITEXT_INTENT_MAP.get(r["intent"])
        # skip unfilled templates like "{{Order Number}}"
        if category is None or "{{" in r["instruction"]:
            continue
        rows.append(
            {"text": _clean(r["instruction"]), "category": category, "source": "bitext"}
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=1500,
                        help="max examples per category (keeps classes balanced)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    rows = load_banking77() + load_tobi() + load_bitext()
    # drop categories that are currently disabled (e.g. urgent)
    rows = [r for r in rows if r["category"] in TICKET_CATEGORIES]

    # dedupe and exclude anything in the test set
    with open(TICKETS_PATH) as f:
        held_out = {_clean(json.loads(line)["text"]).lower() for line in f}
    seen, unique = set(), []
    for row in rows:
        key = row["text"].lower()
        if key in seen or key in held_out:
            continue
        seen.add(key)
        unique.append(row)

    # balance classes, round-robin across sources within each class
    by_class = {}
    for row in unique:
        by_class.setdefault(row["category"], {}).setdefault(row["source"], []).append(row)

    final = []
    for category, by_source in by_class.items():
        for pool in by_source.values():
            rng.shuffle(pool)
        picked, pools = [], list(by_source.values())
        while len(picked) < args.per_class and any(pools):
            for pool in pools:
                if pool and len(picked) < args.per_class:
                    picked.append(pool.pop())
        final.extend(picked)
    rng.shuffle(final)

    PUBLIC_TICKETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PUBLIC_TICKETS_PATH, "w") as f:
        for row in final:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(final)} examples to {PUBLIC_TICKETS_PATH}")
    for (category, source), n in sorted(Counter((r["category"], r["source"]) for r in final).items()):
        print(f"  {category:<10} {source:<12} {n}")


if __name__ == "__main__":
    main()
