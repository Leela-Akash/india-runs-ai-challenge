# India Runs – Data & AI Challenge
## Senior AI Engineer Candidate Ranker
**Author:** Leela Akash Maridi | KL University | 2300030412

---

## Setup & Run

```bash
# Install dependencies (all stdlib except nothing — zero external deps)
python ranker.py
```

Place `candidates.jsonl.gz` in the same directory. Output: `submission.csv`

## Architecture

```
candidates.jsonl.gz
        ↓
  Feature Extraction
  ┌─────────────────────────────────────────┐
  │  Skill Match       (weight: 0.50)       │
  │  Experience Fit    (weight: 0.20)       │
  │  Career Relevance  (weight: 0.10)       │
  │  Redrob Signals    (weight: 0.10)       │
  │  Education         (weight: 0.05)       │
  │  Recency/Activity  (weight: 0.05)       │
  └─────────────────────────────────────────┘
        ↓
  Anti-Cheat Penalty Layer
  (keyword stuffing, fake profiles, wrappers-only)
        ↓
  Sort → Top 100 → submission.csv
```

## Skill Matching Strategy

- **Tier 1** (full weight): Production embeddings, FAISS, Milvus, Pinecone, RAG,
  Ranking, NDCG, PyTorch, Sentence Transformers, LoRA/QLoRA, etc.
- **Tier 2** (30% weight): HuggingFace, Docker, AWS, Pandas, etc.
- **Penalty**: Marketing, Sales, HR, Civil/Mechanical (non-AI)

## Anti-Cheat Logic

1. **Keyword stuffing**: >20 tier-1 skills in skills list → 30% penalty
2. **No production evidence**: AI keywords + zero deployment signals → 20% penalty
3. **Wrapper-only**: LangChain/OpenAI API without real ML → 25% penalty
4. **Title-experience mismatch**: Claims 10yrs, only 1 job → 15% penalty
5. **Irrelevant background**: Current title in HR/Sales/Marketing → 40% penalty
6. **Ghost profiles**: Unverified + inactive + not open to work → 30% penalty

## Evaluation Alignment

Optimized for **NDCG@10 (50% weight)** — top 10 candidates are highest-confidence
matches combining:
- Production AI/ML experience verified through career descriptions
- Core skill coverage (embeddings + retrieval + ranking)
- Strong behavioral signals (verified, active, responsive)
- Experience in sweet spot (5-9 years)

## Constraints Met

- ✅ CPU only
- ✅ ~2-3 min runtime on 100k candidates
- ✅ <1 GB RAM usage
- ✅ Zero external dependencies (stdlib only)
- ✅ No internet / LLM APIs at runtime
