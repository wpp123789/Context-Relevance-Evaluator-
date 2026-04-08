from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI
import json
import re
import os

app = Flask(__name__, static_folder=".")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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"
)
MODEL = "openrouter/free"

def call_model(prompt, max_tokens=1000):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()

def chunk_text(text, chunk_size=400, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def retrieve_chunks(query, chunks, top_k=5):
    if not chunks:
        return []
    chunk_list = "\n\n".join([f"[Chunk {i+1}]:\n{c}" for i, c in enumerate(chunks)])
    prompt = f"""You are a retrieval system. Given a query and document chunks, return the indices of the top {top_k} most relevant chunks.

Query: {query}

Chunks:
{chunk_list}

Return ONLY a JSON array of chunk indices (1-based), e.g. [1, 3, 5]. No explanation."""

    text = call_model(prompt)
    match = re.search(r'\[[\d,\s]+\]', text)
    if match:
        indices = json.loads(match.group())
        return [chunks[i-1] for i in indices if 1 <= i <= len(chunks)]
    return chunks[:top_k]

def evaluate_context(query, retrieved_chunks):
    evaluations = []
    for i, chunk in enumerate(retrieved_chunks):
        prompt = f"""Evaluate this document chunk for answering the given query. Be concise and precise.

Query: {query}

Chunk:
{chunk}

Return ONLY a JSON object with these exact fields:
{{
  "relevance_score": <integer 1-10>,
  "relevance_reason": "<one sentence>",
  "answer_contribution": "<what specific information this chunk provides, or 'None' if irrelevant>",
  "noise_level": "<Low|Medium|High>",
  "noise_reason": "<one sentence about what makes it noisy or clean>"
}}"""

        text = call_model(prompt)
        text = re.sub(r'^```json\s*', '', text)
        text = re.sub(r'^```\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        text = text.strip()

        try:
            eval_data = json.loads(text)
            eval_data["chunk_index"] = i + 1
            eval_data["chunk_preview"] = chunk[:200] + "..." if len(chunk) > 200 else chunk
            evaluations.append(eval_data)
        except json.JSONDecodeError:
            evaluations.append({
                "chunk_index": i + 1,
                "chunk_preview": chunk[:200] + "...",
                "relevance_score": 0,
                "relevance_reason": "Evaluation failed",
                "answer_contribution": "None",
                "noise_level": "High",
                "noise_reason": "Could not parse evaluation"
            })
    return evaluations

def generate_summary(query, evaluations):
    scores = [e.get("relevance_score", 0) for e in evaluations]
    avg_score = sum(scores) / len(scores) if scores else 0
    high_noise = sum(1 for e in evaluations if e.get("noise_level") == "High")
    useful_chunks = [e for e in evaluations if e.get("relevance_score", 0) >= 7]

    prompt = f"""Summarize the context quality for this financial query in 2-3 sentences. Be direct and actionable.

Query: {query}
Average relevance score: {avg_score:.1f}/10
Useful chunks (score >=7): {len(useful_chunks)}/{len(evaluations)}
High-noise chunks: {high_noise}/{len(evaluations)}

Give a brief verdict: is this context good enough to answer the query? What's missing or what's the main issue?
Return ONLY the summary text, no labels or prefixes."""

    return call_model(prompt)

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/evaluate", methods=["POST"])
def evaluate():
    data = request.json
    query = data.get("query", "").strip()
    documents = data.get("documents", [])

    if not query:
        return jsonify({"error": "Query is required"}), 400
    if not documents:
        return jsonify({"error": "At least one document is required"}), 400

    all_chunks = []
    for doc in documents:
        text = doc.get("content", "")
        name = doc.get("name", "Document")
        chunks = chunk_text(text)
        for chunk in chunks:
            all_chunks.append({"text": chunk, "source": name})

    if not all_chunks:
        return jsonify({"error": "No content to process"}), 400

    chunk_texts = [c["text"] for c in all_chunks]
    retrieved_texts = retrieve_chunks(query, chunk_texts, top_k=5)

    retrieved = []
    for rt in retrieved_texts:
        for c in all_chunks:
            if c["text"] == rt:
                retrieved.append(c)
                break

    evaluations = evaluate_context(query, [r["text"] for r in retrieved])

    for i, eval_item in enumerate(evaluations):
        if i < len(retrieved):
            eval_item["source"] = retrieved[i]["source"]

    summary = generate_summary(query, evaluations)

    scores = [e.get("relevance_score", 0) for e in evaluations]
    avg_score = sum(scores) / len(scores) if scores else 0

    return jsonify({
        "query": query,
        "total_chunks_processed": len(all_chunks),
        "chunks_retrieved": len(retrieved),
        "average_relevance_score": round(avg_score, 1),
        "summary": summary,
        "evaluations": evaluations
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
