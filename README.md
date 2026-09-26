# Support Copilot

Support ticket assistant for FitTrack, a fitness app. Incoming tickets are
classified by a LoRA-tuned DistilBERT model, matched against the help-centre
KB in Chroma, and answered by an LLM with the source articles cited.
A LangGraph graph runs the steps, FastAPI serves the API and a small web UI,
and Ragas is used for evaluation.

## How it works

```
ticket ─▶ route ─▶ retrieve ─▶ respond ─▶ reply + sources
          │         │           │
          │         │           └ LLM, answers only from retrieved chunks
          │         └ top-4 chunks from Chroma
          └ LoRA classifier: billing / technical / account
            (keyword fallback if no adapter is trained)
```

The graph also has an escalation node that hands a ticket to a human instead
of the LLM. It's disabled for now: urgency used to be a fourth classifier
label, but it overlaps with the others ("billed £500, fix it right now" is
billing and urgent), so it needs to be detected separately. Tickets in
`data/tickets/sample_tickets.jsonl` that should escalate are marked
`"urgent": true`.

## Stack

- LangGraph for the pipeline
- Chroma + OpenAI embeddings for retrieval
- DistilBERT + PEFT/LoRA for classification
- Ragas for evaluation
- FastAPI, Docker

## Project layout

```
app/
  graph.py                 pipeline definition
  agents/                  router, retriever, responder nodes
  rag/                     KB ingestion and Chroma helpers
  classifier/              LoRA training, inference, trained adapter
  eval/                    Ragas eval + dataset
api/main.py                FastAPI app (POST /ticket, serves web UI)
web/index.html             chat UI
data/kb/                   help-centre articles
data/tickets/              sample tickets (classifier test set)
scripts/                   seed vector store, build/check training data
docker/                    Dockerfile, compose
render.yaml, railway.json  deploy configs
```

## Setup

Requires Python 3.12 (torch 2.5.1 has no wheels for 3.13+).

```bash
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set OPENAI_API_KEY

python scripts/seed_vectorstore.py
uvicorn api.main:app --reload
```

Open http://localhost:8000, or:

```bash
curl -X POST localhost:8000/ticket -H "Content-Type: application/json" \
  -d '{"text": "I was charged twice for my premium subscription this month"}'
```

With Docker:

```bash
docker compose -f docker/docker-compose.yml up --build
```

## Classifier

A trained adapter is included in `app/classifier/lora_adapter/`. To retrain:

```bash
python scripts/prepare_training_data.py   # builds data/tickets/public_tickets.jsonl
python scripts/check_training_data.py     # balance, length leakage, TF-IDF baseline
python app/classifier/train_lora.py
```

Training data is 4,500 tickets (1,500 per category) from Banking77
(CC-BY-4.0), Tobi-Bueck/customer-support-tickets (CC-BY-NC-4.0) and Bitext
(CDLA-Sharing-1.0). The 28 tickets in `sample_tickets.jsonl` are the test
set. Current adapter: 97.7% on validation, 86% on the test set.

## Evaluation

```bash
python app/eval/run_eval.py
```

Scores faithfulness, answer relevancy and context precision on
`app/eval/eval_dataset.jsonl`, writes `eval_report.csv`, and exits non-zero if
any metric is below its threshold.

## Tests

```bash
pytest tests/
```

## Deploy

One Docker image serves both the API and the UI. On startup it seeds the
vector store and loads the classifier, so no volume or seed step is needed.

- **Railway:** New Project → Deploy from GitHub repo (uses `railway.json`).
  Set `OPENAI_API_KEY`, then generate a domain under Settings → Networking.
- **Render:** New → Blueprint → select the repo (uses `render.yaml`). Needs the
  Standard plan; Starter (512 MB) runs out of memory.

`/ticket` is rate limited per IP (`RATE_LIMIT_PER_MINUTE`, default 10) and
capped at 2,000 characters. It's still worth setting a spend limit on the
OpenAI key.

## TODO

- Urgency detection + re-enable escalation
- Voice input in the web UI
- Hybrid search (BM25 + dense)
- LangSmith tracing
