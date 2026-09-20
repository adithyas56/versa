# Versa — Demo Script

Target: ~90–120 seconds. The one causal chain a judge must see in under a
minute:

> Student asks → Moss instantly finds relevant memories about *this* student
> → Gemini uses them → Versa answers differently for different students.

## Setup (once)

```bash
cp .env.example .env          # add GEMINI_API_KEY; add MOSS_* for real Moss numbers
docker compose up -d          # Postgres
uv run probe migrate          # or let pytest/first run build the schema
uv run python scripts/seed_demo_learners.py     # Learner A (example-first), Learner B (theory-first)
uv run python scripts/build_moss_index.py       # project facts → Moss index
uv run probe serve            # http://127.0.0.1:8000
```

Set `VERSA_DEMO_MODE=true` to keep the retrieval telemetry panel visible.

## Opening (10s)

> "Most AI tutors can answer your question. The problem is they don't
> remember *how you* learn — and bolting memory on usually makes every
> reply slower."

## Stage 1 — Versa learns a preference (25s)

As **demo-learner-a**, ask:

> Explain recursion.

Versa detects the style isn't pinned down and offers one sharp question with
clickable options:

```
[ Real-world example ] [ Step-by-step technical ] [ Exam-focused ]
```

Click **Real-world example**. The choice is persisted as a durable
`learner_facts` row and projected into Moss.

> "Instead of guessing, Versa asked once — and remembered the answer."

## Stage 2 — Memory is retrieved later (25s)

Later, as the same learner, ask:

> Explain dynamic programming.

Point at the **MOSS MEMORY RETRIEVAL** panel under the answer:

```
MOSS MEMORY RETRIEVAL
engine  local index   |   retrieved 3/5   |   latency  X.XX ms
• start with a concrete real-world example before any formal definition
• studied recursion but struggled with the base case
```

Versa opens with a concrete analogy — not a textbook definition — *because*
those memories were retrieved and injected.

> "That retrieval ran through Moss, locally — so personalization didn't cost
> a slow round trip on the turn."

## Stage 3 — Two students, same question (30s)

Same question to **demo-learner-a** and **demo-learner-b**:

> How should you explain a new concept to me?

- **Learner A** → Moss retrieves the example-first memory → analogy-first answer.
- **Learner B** → Moss retrieves the definition-first memory → formal-definition-first answer.

Same model, same question, **different retrieved memory → different answer.**

## Stage 4 — The latency claim (15s)

Show the measured benchmark (see [BENCHMARKS.md](BENCHMARKS.md)):

```bash
uv run python scripts/benchmark_retrieval.py --engine moss --docs 10000 --queries 200
uv run python scripts/benchmark_retrieval.py --engine pgvector --docs 10000 --queries 200
```

Read the P50/P95 for each. State the claim precisely: *Moss reduces the
retrieval portion of the personalization path* — not that it speeds up
Gemini.

## Closing (10s)

> "Versa gives AI a memory of how you learn. Moss makes that memory fast
> enough to use inside the conversation itself."

---

**Honesty notes for the recording:** the numbers on screen must be the ones
the tools actually printed. The in-process stub engine is for offline
development; record the on-screen latency and the benchmark with real Moss
credentials configured so the panel and BENCHMARKS.md show real Moss timings,
not the stub's.
