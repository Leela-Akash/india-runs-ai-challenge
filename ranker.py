"""
India Runs - Data & AI Challenge
Candidate Ranker for Senior AI Engineer Role (Redrob AI)
Author: Leela Akash Maridi | KL University
"""

import gzip
import json
import csv
import re
import math
import time
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
INPUT_FILE = "candidates.jsonl.gz"   # change if uncompressed: "candidates.jsonl"
OUTPUT_FILE = "submission.csv"
TOP_N = 100

# Scoring weights (as specified in problem statement)
W_SKILL      = 0.50
W_EXPERIENCE = 0.20
W_CAREER     = 0.10
W_SIGNALS    = 0.10
W_EDUCATION  = 0.05
W_RECENCY    = 0.05

# ─────────────────────────────────────────────
# SKILL MATCHING
# ─────────────────────────────────────────────

# Tier 1 - Must Have (full points)
TIER1_SKILLS = {
    "python", "embeddings", "embedding", "vector search", "vector database",
    "faiss", "milvus", "pinecone", "weaviate", "elasticsearch", "opensearch",
    "qdrant", "retrieval", "rag", "retrieval augmented generation",
    "ranking", "learning to rank", "ltr", "ndcg", "map", "mrr",
    "a/b testing", "ab testing", "semantic search", "hybrid search",
    "sentence transformers", "bge", "e5", "transformers", "nlp",
    "llm", "large language model", "fine-tuning", "finetuning",
    "lora", "qlora", "peft", "xgboost", "production ml",
    "ml evaluation", "pytorch", "tensorflow", "information retrieval",
    "recommendation system", "recommender", "search ranking",
}

# Tier 2 - Nice to Have (half points)
TIER2_SKILLS = {
    "huggingface", "langchain", "openai", "bert", "gpt",
    "spark", "kafka", "redis", "mongodb", "postgresql",
    "docker", "kubernetes", "aws", "gcp", "azure",
    "mlflow", "airflow", "dbt", "scala", "java",
    "distributed systems", "microservices", "rest api",
    "machine learning", "deep learning", "neural network",
    "scikit-learn", "sklearn", "pandas", "numpy",
    "open source", "github", "ci/cd",
}

# Penalty skills (negative signals)
PENALTY_SKILLS = {
    "marketing", "sales", "accounting", "hr", "human resources",
    "civil engineering", "mechanical engineering", "electrical engineering",
    "seo", "social media", "content writing", "graphic design",
}

# Pure wrapper red flags (penalize if ONLY these, no production)
WRAPPER_ONLY = {"langchain", "openai api", "chatgpt", "prompt engineering"}

def normalize_skill(s):
    return s.lower().strip()

def score_skills(candidate):
    """Score candidate skills. Returns 0-1."""
    skills_list = candidate.get("skills", [])
    
    skill_names = set()
    skill_proficiency_boost = 0.0
    skill_durations = {}
    
    for sk in skills_list:
        name = normalize_skill(sk.get("name", ""))
        skill_names.add(name)
        prof = sk.get("proficiency", "").lower()
        endorsements = sk.get("endorsements", 0) or 0
        duration = sk.get("duration_months", 0) or 0
        
        if prof in ("expert", "advanced"):
            skill_proficiency_boost += 0.02
        if endorsements > 10:
            skill_proficiency_boost += 0.01
        skill_durations[name] = duration
    
    # Also extract from headline, summary, career descriptions
    text_blob = _extract_text_blob(candidate).lower()
    
    tier1_hits = 0
    tier2_hits = 0
    penalty_hits = 0
    
    for sk in TIER1_SKILLS:
        if sk in skill_names or sk in text_blob:
            tier1_hits += 1
            # Bonus for longer duration
            dur = skill_durations.get(sk, 0)
            if dur > 24:
                tier1_hits += 0.1
    
    for sk in TIER2_SKILLS:
        if sk in skill_names or sk in text_blob:
            tier2_hits += 1
    
    for sk in PENALTY_SKILLS:
        if sk in skill_names:
            penalty_hits += 1
    
    # Max reasonable tier1 hits ~15, tier2 ~10
    tier1_score = min(tier1_hits / 12.0, 1.0)
    tier2_score = min(tier2_hits / 8.0, 1.0) * 0.3  # tier2 worth 30% of skill score
    
    raw = tier1_score * 0.7 + tier2_score + min(skill_proficiency_boost, 0.1)
    raw -= penalty_hits * 0.05
    
    # Anti-cheat: detect keyword stuffing
    if len(skill_names) > 60:
        raw -= 0.15  # suspiciously many skills
    
    return max(0.0, min(1.0, raw))

def _extract_text_blob(candidate):
    """Extract all searchable text from a candidate."""
    parts = []
    profile = candidate.get("profile", {})
    parts.append(profile.get("headline", ""))
    parts.append(profile.get("summary", ""))
    parts.append(profile.get("current_title", ""))
    
    for job in candidate.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
        parts.append(job.get("company", ""))
    
    for sk in candidate.get("skills", []):
        parts.append(sk.get("name", ""))
    
    for cert in candidate.get("certifications", []):
        parts.append(cert.get("name", "") if isinstance(cert, dict) else str(cert))
    
    return " ".join(parts).lower()

# ─────────────────────────────────────────────
# EXPERIENCE SCORING
# ─────────────────────────────────────────────

def score_experience(candidate):
    """Score experience fit. Returns 0-1."""
    profile = candidate.get("profile", {})
    yoe = profile.get("years_of_experience", 0) or 0
    
    # Sweet spot: 5-9 years
    if 5 <= yoe <= 9:
        exp_score = 1.0
    elif 4 <= yoe < 5:
        exp_score = 0.85
    elif 9 < yoe <= 12:
        exp_score = 0.80  # senior but maybe over-experienced / management drift
    elif 3 <= yoe < 4:
        exp_score = 0.65
    elif yoe > 12:
        exp_score = 0.60  # risk of being architect-only
    elif 2 <= yoe < 3:
        exp_score = 0.40
    else:
        exp_score = 0.20
    
    # Check for management drift (no longer codes)
    title = profile.get("current_title", "").lower()
    if any(t in title for t in ["vp", "director", "cto", "chief", "head of", "principal architect"]):
        exp_score -= 0.20
    
    # Penalize frequent job hopping
    career = candidate.get("career_history", [])
    if len(career) > 2:
        short_stints = sum(1 for job in career 
                          if _parse_duration_months(job.get("duration", "")) < 12)
        if short_stints >= 3:
            exp_score -= 0.15
        elif short_stints >= 2:
            exp_score -= 0.07
    
    return max(0.0, min(1.0, exp_score))

def _parse_duration_months(duration_str):
    """Parse duration string to months. e.g. '2 years 3 months' -> 27"""
    if not duration_str:
        return 0
    duration_str = str(duration_str).lower()
    years = re.findall(r'(\d+\.?\d*)\s*year', duration_str)
    months = re.findall(r'(\d+\.?\d*)\s*month', duration_str)
    total = sum(float(y) * 12 for y in years) + sum(float(m) for m in months)
    return total if total > 0 else 0

# ─────────────────────────────────────────────
# CAREER RELEVANCE SCORING
# ─────────────────────────────────────────────

AI_ML_TITLES = {
    "machine learning", "ml engineer", "ai engineer", "data scientist",
    "nlp engineer", "research engineer", "applied scientist",
    "search engineer", "ranking engineer", "recommendation",
    "computer vision", "deep learning", "ai researcher",
    "backend engineer", "software engineer",  # lower weight but relevant
}

IRRELEVANT_TITLES = {
    "marketing", "sales", "operations", "hr manager", "accountant",
    "mechanical", "civil", "electrical", "finance", "business analyst",
    "customer success", "support engineer",
}

AI_COMPANIES = {
    "google", "deepmind", "openai", "anthropic", "meta ai", "microsoft",
    "amazon", "apple", "nvidia", "hugging face", "cohere", "mistral",
    "redrob", "linkedin", "twitter", "uber", "airbnb", "netflix",
    "flipkart", "swiggy", "zomato", "meesho", "razorpay", "freshworks",
    "zoho", "infosys", "tcs", "wipro", "hcl",
}

def score_career_relevance(candidate):
    """Score how relevant career history is. Returns 0-1."""
    career = candidate.get("career_history", [])
    if not career:
        return 0.1
    
    relevance_score = 0.0
    total_weight = 0.0
    
    for i, job in enumerate(career):
        title = job.get("title", "").lower()
        company = job.get("company", "").lower()
        description = job.get("description", "").lower()
        industry = job.get("industry", "").lower()
        duration = _parse_duration_months(job.get("duration", ""))
        
        # Recency weight — more recent jobs matter more
        recency_weight = 1.0 / (i + 1)
        
        job_score = 0.0
        
        # Title relevance
        for t in AI_ML_TITLES:
            if t in title:
                job_score = max(job_score, 0.9 if "ml" in title or "ai" in title or "nlp" in title else 0.6)
        
        for t in IRRELEVANT_TITLES:
            if t in title:
                job_score = min(job_score, 0.1)
        
        # Description signals
        prod_signals = ["deployed", "production", "shipped", "built", "launched",
                       "serving", "inference", "pipeline", "api", "system"]
        research_only = ["paper", "arxiv", "published", "research intern", "phd research"]
        
        prod_count = sum(1 for s in prod_signals if s in description)
        research_count = sum(1 for s in research_only if s in description)
        
        if prod_count > 2:
            job_score = min(1.0, job_score + 0.15)
        if research_count > 1 and prod_count == 0:
            job_score = max(0.0, job_score - 0.20)
        
        # Company signals
        for ac in AI_COMPANIES:
            if ac in company:
                job_score = min(1.0, job_score + 0.05)
        
        # Industry
        if "technology" in industry or "software" in industry or "ai" in industry:
            job_score = min(1.0, job_score + 0.05)
        
        relevance_score += job_score * recency_weight * (duration / 12.0 + 0.5)
        total_weight += recency_weight * (duration / 12.0 + 0.5)
    
    if total_weight == 0:
        return 0.0
    
    return max(0.0, min(1.0, relevance_score / total_weight))

# ─────────────────────────────────────────────
# REDROB SIGNALS SCORING
# ─────────────────────────────────────────────

def score_redrob_signals(candidate):
    """Score behavioral signals from the platform. Returns 0-1."""
    signals = candidate.get("redrob_signals", {})
    if not signals:
        return 0.3  # neutral if no data
    
    score = 0.5  # start at neutral
    
    # Positive signals
    if signals.get("open_to_work") == True:
        score += 0.10
    if signals.get("verified_profile") == True:
        score += 0.05
    if signals.get("willing_to_relocate") == True:
        score += 0.05
    
    # GitHub activity (0-100 scale assumed)
    github = signals.get("github_activity_score", 0) or 0
    score += (github / 100.0) * 0.15
    
    # Recruiter response rate
    response_rate = signals.get("recruiter_response_rate", 0) or 0
    score += (response_rate / 100.0) * 0.08
    
    # Interview completion rate
    interview_rate = signals.get("interview_completion_rate", 0) or 0
    score += (interview_rate / 100.0) * 0.05
    
    # Profile completeness
    completeness = signals.get("profile_completeness", 0) or 0
    score += (completeness / 100.0) * 0.05
    
    # Recruiter saves and views (normalize to 0-1)
    saves = min((signals.get("recruiter_saves", 0) or 0) / 20.0, 1.0)
    views = min((signals.get("recruiter_views", 0) or 0) / 100.0, 1.0)
    score += saves * 0.04 + views * 0.02
    
    # Skill assessment scores
    assessment = signals.get("skill_assessment_score", 0) or 0
    score += (assessment / 100.0) * 0.05
    
    # Negative signals
    notice_period = signals.get("notice_period_days", 0) or 0
    if notice_period > 90:
        score -= 0.08
    elif notice_period > 60:
        score -= 0.04
    
    if signals.get("recently_active") == False:
        score -= 0.08
    
    if signals.get("open_to_work") == False:
        score -= 0.05
    
    if signals.get("verified_profile") == False:
        score -= 0.03
    
    # Failed interviews
    failed = signals.get("failed_interviews", 0) or 0
    score -= failed * 0.03
    
    # Low response rate
    if response_rate < 30:
        score -= 0.05
    
    return max(0.0, min(1.0, score))

# ─────────────────────────────────────────────
# EDUCATION SCORING
# ─────────────────────────────────────────────

TIER1_INSTITUTIONS = {
    "iit", "iim", "bits pilani", "nit", "iisc", "iiit",
    "mit", "stanford", "carnegie mellon", "cmu", "berkeley",
    "oxford", "cambridge", "eth zurich",
}

RELEVANT_FIELDS = {
    "computer science", "artificial intelligence", "machine learning",
    "data science", "information technology", "software engineering",
    "statistics", "mathematics", "computational", "electronics",
}

def score_education(candidate):
    """Score education. Returns 0-1."""
    education = candidate.get("education", [])
    if not education:
        return 0.3
    
    best_score = 0.3
    
    for edu in education:
        inst = edu.get("institution", "").lower()
        degree = edu.get("degree", "").lower()
        field = edu.get("field", "").lower()
        tier = edu.get("tier", "").lower()
        grade = edu.get("grade", "") or ""
        
        edu_score = 0.3
        
        # Institution tier
        if tier in ("tier1", "tier 1", "t1", "premium"):
            edu_score += 0.30
        elif tier in ("tier2", "tier 2", "t2"):
            edu_score += 0.15
        
        for t1 in TIER1_INSTITUTIONS:
            if t1 in inst:
                edu_score = max(edu_score, 0.70)
        
        # Degree level
        if "phd" in degree or "doctorate" in degree:
            edu_score += 0.15
        elif "master" in degree or "m.tech" in degree or "m.s" in degree:
            edu_score += 0.10
        elif "bachelor" in degree or "b.tech" in degree or "b.e" in degree:
            edu_score += 0.05
        
        # Field relevance
        for f in RELEVANT_FIELDS:
            if f in field:
                edu_score += 0.10
                break
        
        best_score = max(best_score, min(1.0, edu_score))
    
    return best_score

# ─────────────────────────────────────────────
# RECENCY / ACTIVITY SCORING
# ─────────────────────────────────────────────

def score_recency(candidate):
    """Score recency and activity signals. Returns 0-1."""
    signals = candidate.get("redrob_signals", {})
    profile = candidate.get("profile", {})
    
    score = 0.5
    
    if signals.get("recently_active") == True:
        score += 0.30
    elif signals.get("recently_active") == False:
        score -= 0.20
    
    github = signals.get("github_activity_score", 0) or 0
    score += (github / 100.0) * 0.20
    
    # Open to work = actively looking = likely responsive
    if signals.get("open_to_work") == True:
        score += 0.10
    
    return max(0.0, min(1.0, score))

# ─────────────────────────────────────────────
# ANTI-CHEAT / HONEYPOT DETECTION
# ─────────────────────────────────────────────

def detect_red_flags(candidate):
    """Returns penalty multiplier (0.0-1.0). Lower = more penalized."""
    penalty = 1.0
    text = _extract_text_blob(candidate)
    skills_list = [normalize_skill(s.get("name", "")) for s in candidate.get("skills", [])]
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])
    
    # 1. Keyword stuffing detection
    tier1_in_skills = sum(1 for s in skills_list if s in TIER1_SKILLS)
    if tier1_in_skills > 20:
        penalty *= 0.70  # suspiciously comprehensive
    
    # 2. No production evidence despite many AI keywords
    has_ai_keywords = any(s in text for s in ["embedding", "faiss", "vector", "rag", "llm"])
    has_production = any(s in text for s in ["deployed", "production", "shipped", "serving", "api"])
    if has_ai_keywords and not has_production:
        penalty *= 0.80
    
    # 3. Wrapper-only profile (just LangChain/OpenAI, no real ML)
    has_real_ml = any(s in text for s in ["pytorch", "tensorflow", "faiss", "training", "fine-tun", "embedding model"])
    has_only_wrappers = any(s in text for s in ["langchain", "openai api"]) and not has_real_ml
    if has_only_wrappers:
        penalty *= 0.75
    
    # 4. Title-experience mismatch (claims 10 yrs but only 2 jobs)
    yoe = profile.get("years_of_experience", 0) or 0
    if yoe > 8 and len(career) <= 1:
        penalty *= 0.85
    
    # 5. Irrelevant background with AI sprinkled in
    current_title = profile.get("current_title", "").lower()
    for t in IRRELEVANT_TITLES:
        if t in current_title:
            penalty *= 0.60
            break
    
    # 6. Unverified + inactive + not open to work = likely ghost profile
    if (signals.get("verified_profile") == False and 
        signals.get("recently_active") == False and
        signals.get("open_to_work") == False):
        penalty *= 0.70
    
    return penalty

# ─────────────────────────────────────────────
# GENERATE REASONING
# ─────────────────────────────────────────────

def generate_reasoning(candidate, scores):
    """Generate 1-2 sentence reasoning for the candidate."""
    profile = candidate.get("profile", {})
    yoe = profile.get("years_of_experience", 0) or 0
    title = profile.get("current_title", "")
    company = profile.get("current_company", "")
    signals = candidate.get("redrob_signals", {})
    
    skill_score = scores["skill"]
    exp_score = scores["experience"]
    
    # Find top matching skills
    text = _extract_text_blob(candidate)
    matched = [s for s in ["embeddings", "faiss", "rag", "vector search", "pytorch",
                             "ranking", "semantic search", "fine-tuning", "lora",
                             "elasticsearch", "pinecone", "milvus", "ndcg"]
               if s in text][:4]
    
    skills_str = ", ".join(matched) if matched else "relevant AI skills"
    
    active_str = "recently active" if signals.get("recently_active") else ""
    open_str = "open to work" if signals.get("open_to_work") else ""
    status = ", ".join(filter(None, [active_str, open_str]))
    status_str = f"; {status}" if status else ""
    
    reason = (
        f"{yoe:.1f}yr {title} at {company} with strong {skills_str} background"
        f"{status_str}."
    )
    
    if skill_score > 0.75:
        reason += " Demonstrates production AI/ML expertise aligned with Senior AI Engineer requirements."
    elif skill_score > 0.50:
        reason += " Shows solid ML foundation with relevant retrieval and ranking experience."
    else:
        reason += " Partial skill match; selected for strong behavioral signals and experience fit."
    
    # Trim to reasonable length
    return reason[:300]

# ─────────────────────────────────────────────
# MAIN SCORING FUNCTION
# ─────────────────────────────────────────────

def score_candidate(candidate):
    """Compute final weighted score. Returns (score, component_scores)."""
    s_skill  = score_skills(candidate)
    s_exp    = score_experience(candidate)
    s_career = score_career_relevance(candidate)
    s_signals = score_redrob_signals(candidate)
    s_edu    = score_education(candidate)
    s_recency = score_recency(candidate)
    
    raw_score = (
        W_SKILL      * s_skill   +
        W_EXPERIENCE * s_exp     +
        W_CAREER     * s_career  +
        W_SIGNALS    * s_signals +
        W_EDUCATION  * s_edu     +
        W_RECENCY    * s_recency
    )
    
    # Apply anti-cheat penalty
    penalty = detect_red_flags(candidate)
    final_score = raw_score * penalty
    
    return final_score, {
        "skill": s_skill,
        "experience": s_exp,
        "career": s_career,
        "signals": s_signals,
        "education": s_edu,
        "recency": s_recency,
        "penalty": penalty,
    }

# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    start = time.time()
    print(f"[INFO] Loading candidates from {INPUT_FILE}...")
    
    candidates = []
    
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        # Try uncompressed
        alt = Path("candidates.jsonl")
        if alt.exists():
            input_path = alt
        else:
            print(f"[ERROR] File not found: {INPUT_FILE}")
            return
    
    opener = gzip.open if str(input_path).endswith(".gz") else open
    
    count = 0
    with opener(input_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            count += 1
            if count % 10000 == 0:
                elapsed = time.time() - start
                print(f"[INFO] Loaded {count:,} candidates... ({elapsed:.1f}s)")
    
    print(f"[INFO] Total candidates loaded: {len(candidates):,} in {time.time()-start:.1f}s")
    
    # Score all candidates
    print("[INFO] Scoring candidates...")
    scored = []
    for i, cand in enumerate(candidates):
        score, components = score_candidate(cand)
        scored.append((cand["candidate_id"], score, components, cand))
        if (i+1) % 10000 == 0:
            print(f"[INFO] Scored {i+1:,}/{len(candidates):,}... ({time.time()-start:.1f}s)")
    
    # Sort by score descending
    scored.sort(key=lambda x: (-x[1], x[0]))  # tie-break: ascending candidate_id
    
    # Take top 100
    top100 = scored[:TOP_N]
    
    print(f"[INFO] Top 100 selected. Score range: {top100[-1][1]:.4f} - {top100[0][1]:.4f}")
    
    # Write output CSV
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        
        for rank, (cid, score, components, cand) in enumerate(top100, 1):
            reasoning = generate_reasoning(cand, components)
            writer.writerow([cid, rank, f"{score:.10f}", reasoning])
    
    elapsed = time.time() - start
    print(f"[SUCCESS] Output written to {OUTPUT_FILE}")
    print(f"[INFO] Total runtime: {elapsed:.1f}s")
    
    # Print top 10 summary
    print("\n── TOP 10 CANDIDATES ──")
    for rank, (cid, score, comp, cand) in enumerate(top100[:10], 1):
        profile = cand.get("profile", {})
        print(f"#{rank:2d} {cid} | {score:.4f} | "
              f"{profile.get('current_title','?')[:30]} @ {profile.get('current_company','?')[:20]} | "
              f"YoE:{profile.get('years_of_experience','?')} | "
              f"Skill:{comp['skill']:.2f} Exp:{comp['experience']:.2f} "
              f"Career:{comp['career']:.2f} Sig:{comp['signals']:.2f}")

if __name__ == "__main__":
    main()
