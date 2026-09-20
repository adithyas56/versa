# Versa — Product Requirements Document

**Versa — Instant Memory for Personalized AI**
*An AI tutor that stops guessing and remembers how you learn — without slowing down.*

---

## Problem

AI tutors are good at generating explanations but weak at maintaining a
durable, evidence-backed understanding of the individual learner. A
student ends up re-explaining, every session, how they like to learn:
examples before definitions or the reverse, what they recently studied,
where they got stuck, what a recurring shorthand means.

Adding memory to fix this introduces a second problem: retrieval. Once a
learner has real history, every meaningful turn needs the *relevant* slice
of that history — and naïve retrieval (a remote embedding call plus a
remote vector search on every turn) adds latency to an interaction that is
supposed to feel immediate.

## Users

Students using an AI tutor repeatedly across multiple sessions. For the
hackathon these are synthetic demo learners; no real minors' data is used.

## Solution

Versa does two things:

1. **Stops guessing.** When a message is genuinely ambiguous it detects the
   ambiguity, presents 2–4 concrete interpretations as one-tap choices,
   records the resolution, and reuses it later — never silently assuming.

2. **Remembers, fast.** Every resolution and evidence-backed observation
   becomes a compact semantic *memory document*. **Moss** loads those
   memories into a locally-held index and retrieves the relevant few on
   every answer turn, with no per-query network round trip. **Gemini** uses
   the retrieved memory to personalize the explanation. **PostgreSQL**
   remains the durable system of record.

The central claim is narrow and testable: *Moss reduces the retrieval
portion of the learner-personalization path, so personalization can live on
the live interaction path without adding a noticeable retrieval delay.*

## Core feature set

1. **Ambiguity detection** — the live `minimal_branch` disambiguation flow.
2. **One-tap interpretation selection** — concrete clickable readings, the
   choice persisted.
3. **Evidence-backed learner memory** — `learner_facts`, append-only,
   written only from real resolutions (a click or a direct answer).
4. **Moss-powered memory retrieval** — one authoritative retrieval adapter
   (`memory_retrieval.retrieve_learner_memory`) on the live path, with
   centralized per-learner filtering.
5. **Personalized generation** — retrieved memory injected into the
   final-answer prompt; correctness prioritized over personalization.
6. **Retrieval telemetry** — measured retrieval latency, engine, result
   count, and the injected memories, shown live in the UI.
7. **Benchmark / control path** — a reproducible Moss-vs-pgvector benchmark.

## What we deliberately did not build

Per the build scope: no multi-agent platform, no revived concept graph or
planner (both retired on measured evidence), no custom embedding model, no
voice, no real-student-data ingestion, no heavy auth. The project is
technically deep but product-simple.

## Data ownership

| System | Owns |
|---|---|
| **PostgreSQL** | Durable source of truth: learners, sessions, turns, disambiguation events, options, resolutions, `learner_facts`, claims, capabilities, audit trail. |
| **Moss** | The retrieval index: compact semantic *projections* of durable memory, loaded locally, queried on the hot path. Never the source of truth. |
| **Gemini** | Generated responses. No durable learner truth — a generated answer is never promoted to a fact except through Versa's existing evidence rules. |

## Success metrics

- **Retrieval latency** — measured `time_taken_ms` per query; P50/P95/P99.
- **Responsiveness** — retrieval stays a small, bounded fraction of the turn.
- **Personalization correctness** — same question + different learner memory
  → visibly different, still-correct answers (the two-learner demo).
- **Retrieval relevance** — Recall@k on a labeled set (`evaluate_retrieval.py`).
- **Cross-learner leakage = 0** — a hard safety metric, enforced centrally
  and tested (`test_cross_learner_isolation.py`).

## Non-goals

A medical/psychological assessment tool, a diagnosis engine, a full student
information system, a teacher replacement, a general AGI memory framework, a
production system for real minors' data, a multi-agent platform, or a
voice-only application.
