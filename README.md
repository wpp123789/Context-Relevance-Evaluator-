# Pacific · Context Relevance Evaluator

A tool that evaluates the quality of retrieved context for financial document queries — built for Pacific's context management use case.

## What It Does

Given a query and one or more financial documents, the tool:
1. **Chunks** documents into overlapping segments
2. **Retrieves** the top-5 most relevant chunks using free LLMs via OpenRouter
3. **Evaluates** each chunk: relevance score (1–10), answer contribution, noise level
4. **Summarizes** overall context quality and actionability

This directly addresses Pacific's core problem: knowing whether your retrieved context is actually *good enough* to answer a query reliably.

## Setup

### 1. Install dependencies

```bash
pip install flask openai
```

### 2. Run

```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

## Usage

1. Type a financial query (e.g. *"What drove Q3 revenue growth?"*)
2. Paste document content — earnings reports, research notes, filings, internal memos
3. Click **Evaluate ↗**
4. Review per-chunk scores, noise analysis, and the overall context quality verdict

## Tech Stack

- **Backend**: Flask + OpenRouter free model router
- **Frontend**: Vanilla HTML/CSS/JS — no build step required
- **Model**: `openrouter/free` — auto-selects best available free LLM

## Why This Matters for Pacific

Pacific's ECMS needs to guarantee that context delivered to AI tools is *reliable and relevant* — not just retrieved. This evaluator makes context quality measurable and auditable, which is especially critical for financial institutions where context errors carry real risk.
