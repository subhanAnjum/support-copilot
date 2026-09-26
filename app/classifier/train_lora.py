"""
LoRA fine-tuning of DistilBERT for ticket categories.

Trains on data/tickets/public_tickets.jsonl if present and uses
sample_tickets.jsonl as the test set; otherwise trains on the sample tickets.

Usage: python app/classifier/train_lora.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType

from app.config import (
    CLASSIFIER_BASE_MODEL,
    CLASSIFIER_ADAPTER_DIR,
    PUBLIC_TICKETS_PATH,
    TICKET_CATEGORIES,
    TICKETS_PATH,
)

LABEL2ID = {label: i for i, label in enumerate(TICKET_CATEGORIES)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


def _read_jsonl(path):
    texts, labels = [], []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            texts.append(row["text"])
            labels.append(LABEL2ID[row["category"]])
    return texts, labels


def load_dataset():
    """Returns (train, val, test). test is None without public data."""
    if Path(PUBLIC_TICKETS_PATH).exists():
        texts, labels = _read_jsonl(PUBLIC_TICKETS_PATH)
        test_texts, test_labels = _read_jsonl(TICKETS_PATH)
        test = Dataset.from_dict({"text": test_texts, "label": test_labels})
    else:
        texts, labels = _read_jsonl(TICKETS_PATH)
        test = None

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    return (
        Dataset.from_dict({"text": train_texts, "label": train_labels}),
        Dataset.from_dict({"text": val_texts, "label": val_labels}),
        test,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": (preds == labels).mean(),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def main():
    print(f"Loading base model: {CLASSIFIER_BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_BASE_MODEL)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        CLASSIFIER_BASE_MODEL,
        num_labels=len(TICKET_CATEGORIES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"],
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    train_ds, val_ds, test_ds = load_dataset()
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds) if test_ds else 0}")
    num_epochs = 3 if len(train_ds) > 500 else 8

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    if test_ds is not None:
        test_ds = test_ds.map(tokenize, batched=True)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="./classifier_checkpoints",
        learning_rate=2e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=num_epochs,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(f"\nFinal validation accuracy: {metrics['eval_accuracy']:.2%}")

    if test_ds is not None:
        output = trainer.predict(test_ds)
        preds = np.argmax(output.predictions, axis=-1)
        print("\nTest set (sample_tickets.jsonl):")
        print(classification_report(
            output.label_ids, preds,
            labels=list(ID2LABEL), target_names=TICKET_CATEGORIES, zero_division=0,
        ))

    Path(CLASSIFIER_ADAPTER_DIR).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(CLASSIFIER_ADAPTER_DIR)
    tokenizer.save_pretrained(CLASSIFIER_ADAPTER_DIR)
    print(f"LoRA adapter saved to {CLASSIFIER_ADAPTER_DIR}")
    if test_ds is None:
        print("Trained on sample tickets only - run scripts/prepare_training_data.py for more data.")


if __name__ == "__main__":
    main()
