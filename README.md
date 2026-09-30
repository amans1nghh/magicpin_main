# Vera — Merchant AI Assistant for magicpin

A stateful HTTP bot that composes high-quality WhatsApp messages for Indian merchants using a 4-context LLM composition framework. Built for the magicpin AI Challenge.

## Approach

Vera uses a **trigger-kind prompt dispatch** architecture: each trigger type (research digest, recall due, perf dip, etc.) gets a specialized prompt template that maximizes the 5-dimension scoring rubric (Specificity, Category Fit, Merchant Fit, Trigger Relevance, Engagement Compulsion).

Key design choices:

- **Trigger-specific prompts** — 20+ prompt variants tuned per trigger kind, each injecting the right compulsion levers (curiosity for research digests, loss aversion for perf dips, effort externalization for renewals).
- **Anti-pattern validator** — Post-composition checks structurally prevent score-killers: no generic discounts when service-at-price exists, no multiple CTAs, no URLs, no taboo vocabulary, no repetition.
- **Conversation intelligence** — Pattern-based auto-reply detection (escalate → wait → end), intent commitment detection (switch to action mode immediately), hostile/opt-out handling (graceful exit + suppress).
- **Adaptive context** — The bot always reads the latest version from its context store, so mid-test context injections (new digest items, updated performance) are used in the next composition automatically.

### Tradeoffs

- **Sequential composition** — Triggers are composed one at a time per tick to stay within timeout budgets. This limits throughput to ~5-7 messages per tick but ensures quality.
- **In-memory state** — All context and conversation state lives in memory. Simple and fast, but doesn't survive restarts. Fine for the test window.
- **Single LLM provider** — Currently wired to OpenAI or Anthropic. No fallback chain across providers.

### What would have helped most

- Real merchant conversation logs to fine-tune auto-reply detection patterns for Indian WhatsApp Business accounts.
- Category-specific offer catalogs with actual pricing data per city/locality.
- A/B test data on which compulsion levers drive the highest reply rates per category.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. Dependencies: FastAPI, uvicorn, pydantic, httpx, hypothesis, pytest.

### 2. Configure LLM

Create a `.env` file in the project root:

```env
LLM_PROVIDER=openai
LLM_API_KEY=sk-your-key-here
LLM_MODEL=gpt-4o
```

Supported providers: `openai`, `anthropic`.

### 3. Expand the dataset

Generate the full 50-merchant / 200-customer / 100-trigger dataset from seeds:

```bash
python main.py --expand-dataset
```

This is deterministic (fixed seed) — every run produces identical output in `dataset/expanded/`.

### 4. Start the bot server

```bash
python main.py
```

The server starts on `http://0.0.0.0:8080`. Custom host/port:

```bash
python main.py --host 127.0.0.1 --port 9000
```

### 5. Run the judge simulator

In a separate terminal (with the bot running):

```bash
python judge_simulator.py
```

This runs all behavioral tests (warmup, auto-reply detection, intent transition, hostile handling) and prints pass/fail results.

To run the full LLM-scored evaluation, edit `judge_simulator.py` line 56:

```python
TEST_SCENARIO = "full_evaluation"
```

Then run it again. This pushes all contexts, sends tick batches, and scores each composed message across the 5 dimensions.

### 6. Generate submission.jsonl

```bash
python main.py --generate-submission
```

Loads the expanded dataset, composes messages for the 30 canonical test pairs, and writes `submission.jsonl`.

### 7. Run tests

```bash
python -m pytest tests/ -v
```

248 tests covering all modules: context store, trigger evaluator, conversation manager, composer, validators, LLM client, bot endpoints, and submission generator.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/healthz` | Liveness probe — returns uptime and context counts |
| `GET` | `/v1/metadata` | Team identity and approach description |
| `POST` | `/v1/context` | Receive context push (category/merchant/customer/trigger) |
| `POST` | `/v1/tick` | Periodic wake-up — bot composes proactive messages |
| `POST` | `/v1/reply` | Receive merchant/customer reply — bot responds |

---

## Project Structure

```
├── main.py                    # Entry point: server, submission gen, dataset expansion
├── src/
│   ├── bot.py                 # FastAPI app with all 5 endpoints
│   ├── config.py              # LLM config, team metadata, timeout budgets
│   ├── context_store.py       # In-memory versioned context persistence
│   ├── trigger_evaluator.py   # Trigger filtering, suppression, urgency ranking
│   ├── conversation_manager.py # Multi-turn state, auto-reply/intent/hostile detection
│   ├── composer.py            # LLM composition engine with trigger-kind dispatch
│   ├── llm_client.py          # Async LLM client (OpenAI/Anthropic) with timeout
│   ├── validators.py          # Post-composition anti-pattern checks
│   ├── submission.py          # submission.jsonl generator
│   ├── models.py              # Pydantic request/response models
│   └── prompts/               # Trigger-kind-specific prompt templates
├── tests/                     # 248 tests (unit + integration)
├── dataset/
│   ├── categories/            # 5 category contexts (dentists, salons, etc.)
│   ├── merchants_seed.json    # 10 seed merchants
│   ├── customers_seed.json    # 15 seed customers
│   ├── triggers_seed.json     # 25 seed triggers
│   ├── generate_dataset.py    # Expands seeds to 50/200/100
│   └── expanded/              # Generated full dataset
├── judge_simulator.py         # Local judge harness for testing
├── Dockerfile                 # Container build (expands dataset at build time)
├── render.yaml                # Render.com deployment config
├── requirements.txt           # Python dependencies
└── .env                       # LLM API key (not committed)
```

---

## Deployment

### Docker

```bash
docker build -t vera-bot .
docker run -p 8080:8080 -e LLM_API_KEY=sk-your-key vera-bot
```

The Dockerfile expands the dataset at build time so the container starts fast.

### Render.com

Push to a Git repo and connect it to Render. The `render.yaml` configures a free-tier web service. Set `LLM_API_KEY` in the Render dashboard.

### Any cloud / ngrok

The bot just needs a public URL that exposes `http(s)://<host>/v1/*`. Deploy anywhere — AWS, GCP, Railway, Fly, or tunnel with ngrok:

```bash
python main.py &
ngrok http 8080
```

Submit the ngrok URL to the challenge portal.
