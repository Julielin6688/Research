# Context Window Management & Memory Retrieval Strategies
# 上下文窗口管理与记忆检索策略

> Research Track 4: The Read Path | April 2026
> For: AI Companion Product Team (Chinese-language, emotion-focused)
> Covers: 2024–2026 papers, production systems, and best practices

---

## Table of Contents

1. [Overview & Problem Statement](#1-overview--problem-statement)
2. [Context Window Management Strategies](#2-context-window-management-strategies)
3. [Memory Retrieval Approaches](#3-memory-retrieval-approaches)
4. [Recency vs. Relevance Tradeoff](#4-recency-vs-relevance-tradeoff)
5. [Compression & Summarization Strategies](#5-compression--summarization-strategies)
6. [Token Budget Allocation](#6-token-budget-allocation)
7. [Proactive vs. On-Demand Retrieval](#7-proactive-vs-on-demand-retrieval)
8. [Evaluation Metrics & Benchmarks](#8-evaluation-metrics--benchmarks)
9. [Recommended Architecture for AI Companion](#9-recommended-architecture-for-ai-companion)
10. [References](#10-references)

---

## 1. Overview & Problem Statement

### 1.1 The Read Path Problem

The prior three research tracks (signal extraction, lifecycle management, entity resolution) address the **write path**: how memories are created, updated, and deduplicated. This document addresses the **read path**: given a user message arriving at inference time, how does the system decide what memories to load into the context window, in what form, and in what order?

This is arguably the more critical engineering problem: a perfect memory store is useless if the retrieval mechanism consistently surfaces stale, irrelevant, or poorly-ordered content.

### 1.2 The Core Tension

Three competing forces constrain every retrieval decision:

| Force | Description | Risk if over-weighted |
|-------|-------------|----------------------|
| **Relevance** | Surface memories topically related to the current query | Miss recent updates; rely on outdated facts |
| **Recency** | Prefer memories from recent sessions | Miss important but older biographical facts |
| **Token budget** | Fit everything in the context window | Truncate critical memories to save tokens |

### 1.3 Why Long Context Windows Don't Solve This

By 2025, production LLMs offer 128k–2M token context windows (Gemini 2.5 Pro at 2M, Claude Sonnet 4 at 1M). A naive approach would be: "just dump all memories into context." This fails for three reasons:

1. **Context rot**: Research shows LLM performance degrades as input length increases when relevant information is spread throughout a long context (Hsieh et al., 2024).
2. **Lost in the Middle**: Liu et al. (2024, published in TACL) demonstrated a U-shaped attention pattern — LLMs attend strongly to content at the beginning and end of context, with a 30%+ accuracy drop for content placed in the middle of 20-document contexts. This is caused by positional encoding decay (RoPE).
3. **Cost and latency**: Processing 1M tokens per request is economically prohibitive for a consumer companion app. Mem0's production data (2026) shows full-context retrieval achieved 72.9% accuracy but with 17-second P95 latency, versus selective memory at 66.9% accuracy with 1.44-second P95 latency.

### 1.4 Scope for This Document

This document covers the read path for a Chinese-language emotion AI companion with the following characteristics:
- Session-based conversations (users return across days/weeks/months)
- Accumulated memory store of hundreds to thousands of facts per user
- Target context budget: 8k–32k tokens per request
- Latency target: < 2 seconds for memory retrieval and assembly
- Primary concerns: personalization accuracy, emotional continuity, avoiding embarrassing factual errors

用中文总结: 本章阐述了"读取路径"问题——在推理时如何决定将什么记忆加载到上下文窗口中。三个核心张力是：相关性（和查询相关的记忆）、时近性（最近的记忆）和令牌预算（能放入上下文的内容总量）。扩大上下文窗口并不能解决问题，因为"lost in the middle"现象表明LLM对长上下文中间位置的内容注意力显著下降，同时长上下文带来的延迟和成本也是生产环境中的实际障碍。

---

## 2. Context Window Management Strategies

### 2.1 The MemGPT/Letta Virtual Context Model

The most influential architecture for long-term memory management is **MemGPT** (Packer et al., 2023), now productionized as **Letta** (open-sourced September 2024, $10M seed). It draws an explicit analogy to OS virtual memory: the context window is RAM, external storage is disk, and the agent controls paging between them.

**Context structure in Letta:**

```
┌─────────────────────────────────────────────────────┐
│                    CONTEXT WINDOW                    │
│                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │
│  │ System Prompt│  │Tool Schemas │  │  Metadata  │  │
│  │ (behavioral │  │ (available  │  │ (agent     │  │
│  │  instructions)│  │  actions)   │  │  stats)    │  │
│  └─────────────┘  └─────────────┘  └────────────┘  │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │              MEMORY BLOCKS                   │   │
│  │  [human]   [persona]   [relationship]        │   │
│  │  (what we know about  (agent character)      │   │
│  │   this user)                                 │   │
│  └──────────────────────────────────────────────┘   │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │           MESSAGE BUFFER (FIFO)              │   │
│  │  recent N turns of conversation              │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘

          ↕  (paging via tool calls)

┌─────────────────────────────────────────────────────┐
│                  EXTERNAL STORAGE                    │
│  - Archival memory (vector-indexed fact store)      │
│  - Conversation history (compressed summaries)      │
│  - Entity store (structured records)                │
└─────────────────────────────────────────────────────┘
```

**Memory blocks** in Letta are the key primitive. Each block has:

```python
class MemoryBlock:
    label: str          # e.g. "human", "persona", "relationship"
    value: str          # string content (may store JSON or markdown)
    size_limit: int     # max characters; hard constraint
    read_only: bool     # whether agent can modify
    block_id: str       # UUID for DB persistence
    description: str    # optional guidance for the agent
```

The agent edits blocks via system tools: `memory_replace`, `memory_append`, `memory_rethink`. When context overflow is imminent, the agent can call `archival_memory_search` to retrieve external records or `conversation_search` to find historical turns.

### 2.2 Architectural Patterns

Three architectural patterns exist along a complexity spectrum (from survey: Memory for Autonomous LLM Agents, arXiv 2603.07670, 2026):

| Pattern | Description | Use Case | Complexity |
|---------|-------------|----------|------------|
| **A: Monolithic Context** | All memory stuffed into prompt | Prototypes, simple bots | Low |
| **B: Context + Retrieval Store** | Working memory in context; long-term indexed externally | Production companion apps | Medium |
| **C: Tiered Memory with Learned Control** | Multiple tiers (context, structured DB, vector store, archive) managed by learned policies | Research systems, MemGPT/Letta | High |

**Recommendation for companion AI**: Start with Pattern B; upgrade to Pattern C only when empirical latency/accuracy data justifies the engineering cost.

### 2.3 Context Selection Strategies

Given a fixed token budget, five strategies exist for selecting what enters the context window:

**Strategy 1: Recency-first (sliding window)**
Keep the last N turns verbatim; discard everything older. Simple but loses long-term biographical facts.

```python
def build_context_sliding_window(history, budget_tokens, tokenizer):
    context = []
    tokens_used = 0
    for turn in reversed(history):  # most recent first
        turn_tokens = tokenizer.count(turn)
        if tokens_used + turn_tokens > budget_tokens:
            break
        context.insert(0, turn)
        tokens_used += turn_tokens
    return context
```

**Strategy 2: Importance-scored selection**
Score each memory by recency + importance + relevance; select top-k to fill budget.

```python
def score_memory(memory, query_embedding, now):
    recency = 0.995 ** hours_since(memory.created_at, now)
    importance = memory.importance_score  # 1-10, LLM-assigned at write time
    relevance = cosine_similarity(memory.embedding, query_embedding)
    return 0.35 * recency + 0.25 * importance + 0.40 * relevance
```

**Strategy 3: Hierarchical summarization**
Verbatim for recent N turns; compressed summaries for older sessions; key facts always pinned.

**Strategy 4: Role-based filtering**
Different context slots for different content types: pinned biographical facts, recent history, retrieved episodic memories, session summary.

**Strategy 5: LLM-driven self-selection (MemGPT)**
The agent itself decides what to load from external storage by issuing tool calls.

### 2.4 Comparison Table

| Strategy | Recency | Long-term Facts | Token Efficiency | Latency | Complexity |
|----------|---------|----------------|-----------------|---------|------------|
| Sliding window | Excellent | Poor | Medium | Minimal | Very low |
| Importance scoring | Good | Good | Good | Low | Low |
| Hierarchical summary | Good | Good | Excellent | Medium | Medium |
| Role-based slots | Good | Excellent | Excellent | Medium | Medium |
| LLM self-selection | Good | Excellent | Variable | High | High |

用中文总结: 本章介绍了上下文窗口管理的主要策略，以Letta/MemGPT的"虚拟内存"架构为代表。上下文由记忆块（Memory Blocks）、消息缓冲区和系统提示组成，超出容量的内容被分页到外部存储。对于AI伴侣产品，推荐采用"模式B"（上下文+检索存储），结合重要性评分策略来决定哪些记忆进入上下文窗口。

---

## 3. Memory Retrieval Approaches

### 3.1 Vector (Dense) Retrieval

**How it works**: Each memory is embedded into a high-dimensional vector. At query time, the query is embedded and the top-k nearest neighbors are retrieved via ANN (approximate nearest neighbor) search.

**Strengths**: Captures semantic similarity; finds conceptually related content even with different phrasing.

**Weaknesses**: Poor at exact keyword matching; struggles with names, dates, and rare terms; computationally expensive to index.

**Common models**: OpenAI text-embedding-ada-002 (1536-dim), text-embedding-3-small, Qwen3-Embedding-4B (strong multilingual performance), BGE-M3.

```python
# Dense retrieval example
def dense_retrieve(query: str, memory_store, k: int = 20):
    query_emb = embed_model.encode(query)
    candidates = memory_store.ann_search(
        query_emb,
        k=k,
        metric="cosine"
    )
    return candidates  # [(memory, score), ...]
```

**Chinese-specific note**: For Chinese companion AI, use a model with strong Chinese support. BGE-M3 (BAAI, 2024) and Qwen3-Embedding achieve state-of-the-art Chinese retrieval performance. OpenAI's ada-002 handles Chinese adequately but underperforms on colloquial or emotionally-nuanced text.

### 3.2 Sparse (Keyword/BM25) Retrieval

**How it works**: TF-IDF-style scoring based on term overlap. BM25 is the standard formula:

```
BM25(q,d) = Σᵢ IDF(qᵢ) · (f(qᵢ,d) · (k₁+1)) / (f(qᵢ,d) + k₁ · (1 - b + b · |d|/avgdl))
```

where `f(qᵢ,d)` is term frequency, `k₁=1.5`, `b=0.75`, `avgdl` is average document length.

**Strengths**: Fast; exact keyword matching; no embedding cost; handles rare terms, proper nouns, product names.

**Weaknesses**: No semantic understanding; fails on synonyms and paraphrases; poor recall for conceptual queries.

**Production numbers**: Top-1 BM25 recall: ~22.1%; dense retrievers (DPR): ~48.7%; hybrid pipelines: up to 53.4% (dev.to/qvfagundes, 2024).

### 3.3 Hybrid Search (BM25 + Dense)

**The standard production pattern** as of 2025: retrieve with both sparse and dense, then fuse results.

**Reciprocal Rank Fusion (RRF)** is the most widely used fusion algorithm:

```python
def reciprocal_rank_fusion(rankings: list[list], k: int = 60):
    """
    rankings: list of ranked result lists (one per retriever)
    k: smoothing constant (typically 60)
    returns: merged ranking sorted by RRF score
    """
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

**Alternative: Linear combination (α-fusion)**:
```
hybrid_score(d) = α · dense_score(d) + (1-α) · bm25_score(d)
```
where α is tuned per domain (typically 0.7 for semantic-heavy tasks, 0.3 for keyword-heavy).

**Zep's production results** (arXiv 2501.13956, January 2025):
- Hybrid search: BM25 + semantic embeddings + graph traversal
- P95 retrieval latency: 300ms
- LongMemEval accuracy improvement: up to 18.5% over baselines
- 90% latency reduction vs. full-context approaches

### 3.4 Graph-Based Retrieval

**The emerging frontier**: Knowledge graphs capture relationships that flat vector stores cannot represent.

**Zep/Graphiti architecture** (Rasmussen et al., arXiv 2501.13956):
- Temporal knowledge graph where facts have validity windows (start/end timestamps)
- When information changes, old facts are invalidated — not deleted
- Query both "what is true now" and "what was true at time T"
- Hybrid retrieval: BM25 keywords + semantic embeddings + graph traversal edges
- Performance: 94.8% on DMR benchmark vs. MemGPT's 93.4%

**HyperMem architecture** (arXiv 2604.08256, April 2026 — published days before this research):

Three-level hypergraph hierarchy with RRF-then-rerank retrieval:

```
HYPERMEM RETRIEVAL PIPELINE
────────────────────────────
Query
  │
  ▼
[1] Topic Retrieval
    RRF(BM25 topics, semantic topics) → reranker → top-kᵀ topics
  │
  ▼
[2] Episode Retrieval
    Expand via hyperedges → RRF(BM25, semantic) → reranker → top-kᴱ episodes
  │
  ▼
[3] Fact Retrieval
    Expand via hyperedges → RRF(BM25, semantic) → reranker → top-kᶠ facts
  │
  ▼
Context Assembly: facts + episode summaries
```

**Node structure:**
```python
class TopicNode:
    title: str
    summary: str
    episode_ids: list[str]  # hyperedge connecting episodes

class EpisodeNode:
    dialogue: str           # raw turns
    title: str
    narrative_summary: str  # LLM-generated
    fact_ids: list[str]     # hyperedge connecting facts
    timestamp: datetime

class FactNode:
    content: str            # atomic assertion: "User's mother is named Li Hua"
    query_patterns: list[str]  # anticipated retrieval queries
    keywords: list[str]
    importance: float       # 0-1
```

**LoCoMo benchmark results (HyperMem, 2026)**:
- Overall accuracy: 92.73% (GPT-4o-mini as judge)
- Single-hop: 96.08%
- Multi-hop: 93.62%
- Temporal reasoning: 89.72%

### 3.5 Reranking

**The standard pipeline**: retrieve top-50 with hybrid search → rerank with cross-encoder → pass top-5 to LLM.

**Cross-encoder reranking**: Unlike bi-encoders (used in dense retrieval), a cross-encoder takes (query, document) as joint input and outputs a single relevance score. Much more accurate but O(n) latency — only feasible for small candidate sets (20-100).

```python
def rerank(query: str, candidates: list[Memory], reranker_model, top_k: int = 5):
    scored = []
    for mem in candidates:
        score = reranker_model.score(query, mem.content)
        scored.append((mem, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [mem for mem, _ in scored[:top_k]]
```

**Common reranker models**:
- Cohere Rerank API
- BGE-Reranker-v2-m3 (strong Chinese support)
- ms-marco-MiniLM-L-6-v2 (fast English-focused)

**Performance impact**: LangChain/LlamaIndex studies show BM25 + dense + reranking reduces hallucination by 22–37% vs. pure vector search.

### 3.6 Retrieval Strategy Comparison

| Method | Recall | Precision | Latency | Handles Synonyms | Exact Match | Cost |
|--------|--------|-----------|---------|-----------------|-------------|------|
| BM25 only | ~22% | High | Very low | No | Yes | Very low |
| Dense only | ~49% | Medium | Medium | Yes | No | Medium |
| Hybrid (BM25+dense) | ~53% | High | Medium | Yes | Yes | Medium |
| Hybrid + reranking | ~53% | Very high | Higher | Yes | Yes | Higher |
| Graph traversal | Variable | Very high | Medium-High | Partial | Yes | High |
| Hybrid + Graph + Rerank | Best | Highest | High | Yes | Yes | Highest |

用中文总结: 本章比较了四种主要检索策略：纯向量搜索（Dense Retrieval）、稀疏关键词检索（BM25）、混合检索（BM25+Dense，通过RRF融合），以及基于图的检索（如Zep的时态知识图谱和HyperMem的超图结构）。生产环境的标准模式是：BM25+Dense混合检索召回top-50候选，再通过交叉编码器重排序取top-5送入LLM。对中文伴侣AI，推荐使用支持中文的嵌入模型（如BGE-M3或Qwen3-Embedding）和重排序模型（如BGE-Reranker-v2-m3）。

---

## 4. Recency vs. Relevance Tradeoff

### 4.1 The Problem

Pure semantic retrieval is "atemporal" — it ranks evidence purely by meaning similarity without any consideration of when the information was recorded. This creates concrete failure modes in companion AI:

- User says "my boyfriend and I broke up" (recent) → system still retrieves old memories describing the relationship positively
- User changes jobs → old employer information still ranks high semantically
- User's mood/situation changes → outdated emotional context retrieved

### 4.2 The Scoring Formula

The most influential scoring formula for combining recency and relevance comes from the generative agents literature (Park et al., 2023, popularized in companion AI contexts):

```python
def score_memory_retrieval(memory, query_embedding, current_time, weights=(0.35, 0.25, 0.40)):
    """
    Three-component scoring: recency + importance + relevance

    Args:
        memory: Memory object with created_at, importance_score, embedding
        query_embedding: vector of the current user query
        current_time: datetime
        weights: (w_recency, w_importance, w_relevance)
    """
    w_r, w_i, w_s = weights

    # 1. Recency: exponential decay, half-life ~60 hours (0.995^60 ≈ 0.74)
    hours_elapsed = (current_time - memory.created_at).total_seconds() / 3600
    recency_score = 0.995 ** hours_elapsed  # normalized to [0, 1]

    # 2. Importance: LLM-assigned at write time, scale 1-10, normalized
    importance_score = memory.importance / 10.0

    # 3. Relevance: cosine similarity between memory and query
    relevance_score = cosine_similarity(memory.embedding, query_embedding)
    # Normalize from [-1, 1] to [0, 1]
    relevance_score = (relevance_score + 1) / 2

    return w_r * recency_score + w_i * importance_score + w_s * relevance_score
```

**Default weights**: Recency 0.35, Importance 0.25, Relevance 0.40. These weights should be tuned per use case — companion AI benefits from higher recency weight (0.40–0.45) to catch life changes quickly.

### 4.3 Temporal Decay with Half-Life Control

A more flexible formula using a configurable half-life (from arXiv 2509.19376, 2025):

```python
def fused_score(query_emb, doc_emb, doc_age_days, alpha=0.7, half_life_days=14):
    """
    alpha: weight on semantic similarity (0.7 default)
    half_life_days: how many days until temporal weight is halved (14 default)
    """
    semantic = cosine_similarity(query_emb, doc_emb)
    temporal = 0.5 ** (doc_age_days / half_life_days)
    return alpha * semantic + (1 - alpha) * temporal
```

**Key finding**: This simple formula achieved perfect accuracy (1.00) on freshness tasks (Latest@10 benchmark). Pure semantic retrieval scored 0.00. The temporal component is essential, not marginal. The formula is robust for α in [0.4, 0.7]; performance degrades when α > 0.9 (over-weighting semantics).

**Half-life tuning guidance for companion AI:**

| Memory Type | Recommended Half-Life | Rationale |
|-------------|----------------------|-----------|
| Mood / emotional state | 2–7 days | Changes frequently |
| Current events in user's life | 7–14 days | Moderately volatile |
| Job, relationships | 30–90 days | Stable but can change |
| Biographical facts (birthdate, hometown) | ∞ (no decay) | Permanent |
| Preferences (food, music) | 60–180 days | Slow drift |

### 4.4 LongMemEval Temporal Boost

The LongMemEval paper (arXiv 2410.10813, ICLR 2025) found that **temporal filtering** boosts temporal recall by up to 11.4% at Recall@5:

```python
def temporal_filter(query: str, memories: list, date_range: tuple):
    """
    If query contains temporal language ("last week", "recently", "when"),
    pre-filter memories to the inferred date range before scoring.
    """
    if has_temporal_language(query):
        inferred_start, inferred_end = parse_temporal_reference(query)
        memories = [m for m in memories
                    if inferred_start <= m.created_at <= inferred_end]
    return memories
```

**Also useful**: fact-augmented key expansion (adding synonyms/related facts to the query embedding) increased recall by 4% and QA accuracy by 5% in the same study.

### 4.5 Memory Pinning: Bypassing Decay for Critical Facts

Not all memories should decay. Implement a "pinned" flag for memories that must always appear:

```python
class Memory:
    content: str
    embedding: list[float]
    created_at: datetime
    importance: float
    pinned: bool = False    # if True, always include in context; no decay
    memory_type: str        # "fact", "episode", "preference", "mood"

    def effective_score(self, query_emb, now):
        if self.pinned:
            return 1.0  # always top-ranked
        return score_memory_retrieval(self, query_emb, now)
```

Pin candidates: user's name, core relationship status, major life facts (chronic illness, disability, significant recent loss), user-explicitly-requested pins.

用中文总结: 本章分析了时近性与相关性之间的权衡问题。核心结论是：纯语义检索（无时间维度）会导致AI引用过时信息，引发尴尬错误。推荐采用三分量评分公式：时近性（指数衰减）× 0.35 + 重要性（LLM评分）× 0.25 + 相关性（余弦相似度）× 0.40。不同类型的记忆应配置不同的"半衰期"——情绪状态半衰期约2-7天，生平事实（如出生地）则永不衰减。对于时间敏感查询（"最近…""上周…"），应先进行时间过滤再执行语义检索。

---

## 5. Compression & Summarization Strategies

### 5.1 The Core Problem

Conversation histories grow unboundedly. A user who has chatted 500 times with a companion app has a history that cannot fit in any context window verbatim. Three complementary strategies address this:

1. **Rolling summaries**: Compress conversation segments into shorter summaries
2. **Hierarchical compression**: Maintain summaries at multiple granularities (turn → session → month)
3. **Fact extraction**: Distill key facts from dialogue into structured records (the write path — covered in signal extraction research)

### 5.2 Recursive Summarization

**Algorithm** (from "Recursively Summarizing Enables Long-Term Dialogue Memory," arXiv 2308.15022, published in Neurocomputing 2025):

```python
# Probabilistic model: P(response | context, sessions) = P(response | context, M_N)
# where M_i = LLM(S_i, M_{i-1}, prompt_memory)
# Each memory update follows a Markov process

def recursive_memory_update(sessions: list[Session], llm, prompt_memory):
    """
    Input: dialogue sessions S_1...S_N
    Output: memory M_N usable at inference time
    """
    M_prev = None  # no prior memory

    for session in sessions:
        # Update memory using current session + previous memory
        M_curr = llm.generate(
            prompt=prompt_memory,
            context=session.turns,
            prior_memory=M_prev
        )
        M_prev = M_curr

    return M_curr  # final memory (max 20 natural language sentences)

def generate_response(current_context, M_N, llm, prompt_response):
    """At inference time, combine current context with accumulated memory."""
    return llm.generate(
        prompt=prompt_response,
        context=current_context,
        memory=M_N
    )
```

**Performance**: 48.2% win rate vs. 11.9% loss rate against MemoryBank in GPT-4 pairwise evaluation. Consistency score: 1.45 vs. 1.40 baseline.

**Trigger mechanism**: Summarization runs when a session ends, not continuously. This avoids mid-session interruption and gives the LLM a coherent unit to summarize.

### 5.3 Factory.ai's Rolling Summary Approach

Factory.ai's production system maintains a lightweight, persistent conversation state as a rolling summary:

```python
class RollingSummary:
    """
    Maintains a compressed representation of conversation history.
    New content is appended; old content is compressed on overflow.
    """
    pinned_summary: str      # anchored facts, never compressed
    rolling_text: str        # recent summary, grows over time
    max_tokens: int = 2000   # budget for summary section

    def update(self, new_turns: list[Turn], llm):
        candidate = self.rolling_text + "\n" + format_turns(new_turns)

        if token_count(candidate) > self.max_tokens:
            # Compress the oldest portion of rolling_text + new_turns
            span_to_compress = get_oldest_span(candidate, self.max_tokens * 0.3)
            compressed = llm.summarize(span_to_compress)
            self.rolling_text = compressed + "\n" + get_recent_span(candidate)
        else:
            self.rolling_text = candidate
```

**Key design choices**:
- Pinned summaries (core biographical facts) are never compressed away
- Only the "newly dropped span" is summarized and merged — avoids re-summarizing already-compressed content
- Triggers based on token count threshold, not time

### 5.4 Hierarchical Memory Architecture

The most sophisticated approach, used in systems like MemGPT/Letta and proposed in cognitive memory frameworks:

```
MEMORY HIERARCHY
═════════════════

Level 0: Working Memory (Current Session)
  - Raw turns: verbatim, last 10-20 turns
  - Freshness: seconds to hours
  - Token budget: 2k-4k

Level 1: Session Summary
  - Generated at session end via LLM
  - Captures: key topics, facts disclosed, emotional arc
  - Freshness: 1 day to weeks
  - Token budget: 200-500 tokens per session

Level 2: Weekly/Monthly Digest
  - Aggregate multiple session summaries
  - Identifies patterns, major life events
  - Freshness: weeks to months
  - Token budget: 500-1k tokens covering a month

Level 3: Core User Profile
  - Stable biographical facts, persistent preferences
  - Always pinned in context (never compressed away)
  - Updated rarely, when major changes detected
  - Token budget: 300-600 tokens
```

### 5.5 EDU-Based Memory (State of the Art, 2025)

The EMem system (arXiv 2511.17208, November 2025) introduces the most principled representation: **Elementary Discourse Units (EDUs)** — short event-style statements extracted from dialogue.

```python
class EDU:
    """Atomic memory unit grounded in conversational event"""
    content: str            # "User's sister got engaged in October 2025"
    participants: list[str]  # ["user", "sister_mei"]
    action: str             # "got_engaged"
    temporal: str           # "October 2025"
    session_id: str
    turn_range: tuple[int, int]
    embedding: list[float]

# Performance: reduces context from 101K to 1.0K-3.6K tokens
# while maintaining 77.9% QA accuracy (vs. 80%+ for full-context)
# +32.7 points improvement on temporal reasoning
# +35.3 points on multi-session questions
```

### 5.6 Summary Strategy Decision Table

| Scenario | Recommended Strategy |
|----------|---------------------|
| < 50 sessions total | Store verbatim, use sliding window + retrieval |
| 50-500 sessions | Recursive session summaries + fact extraction |
| 500+ sessions | Full hierarchy: EDUs + session summaries + monthly digests + core profile |
| Real-time latency critical (< 500ms) | Pre-compute summaries offline; serve from cache |
| High factual accuracy required | EDU extraction + graph storage (preserves provenance) |

用中文总结: 本章讨论了如何压缩和总结对话历史，以管理无限增长的记忆。三种核心策略是：（1）递归摘要——每次会话结束后更新记忆摘要，通过马尔可夫链式更新保持连贯性；（2）滚动摘要——维护固定大小的摘要缓冲区，超出时压缩最旧的部分；（3）分层记忆架构——从原始对话（Level 0）到会话摘要（Level 1）、月度摘要（Level 2）到核心用户档案（Level 3）。最前沿的方法（EMem，2025）使用"基本话语单元"（EDU）——将对话压缩为事件式原子陈述，能将上下文从101K压缩到1-3.6K令牌，同时保持高QA准确性。

---

## 6. Token Budget Allocation

### 6.1 The Budget Problem

Given a total context budget B tokens, the system must allocate tokens across:
1. System prompt (behavioral instructions)
2. Long-term memory snippets (retrieved facts)
3. Recent conversation history (verbatim turns)
4. Compressed summaries of older history
5. Current user turn
6. Reserved space for model output

### 6.2 Recommended Allocation Table

For a 16k token budget (typical for a companion app using GPT-4o or Claude Haiku):

| Component | Tokens | % of Budget | Notes |
|-----------|--------|-------------|-------|
| System prompt | 800–1,200 | 5–8% | Include persona, behavioral rules, output format |
| Core user profile (pinned) | 400–600 | 3–4% | Name, key facts, relationship status |
| Retrieved long-term memories | 1,200–2,400 | 8–15% | Top 5-10 retrieved facts, 150-250 tokens each |
| Session summaries (last 3-5 sessions) | 600–1,200 | 4–8% | Compressed; 150-300 tokens per session |
| Recent turns (verbatim) | 4,000–8,000 | 25–50% | Last 10-30 turns depending on turn length |
| Current user turn | 200–500 | 1–3% | Current message |
| Output buffer | 1,500–3,000 | 10–20% | Reserve for model response |
| Safety margin | 500–1,000 | 3–6% | Avoid overflow |

**Total**: ~9,200–17,900 tokens (fits a 16k–20k window)

### 6.3 Budget for Different Window Sizes

| Window Size | System + Profile | Retrieved Memories | Recent History | Output Buffer |
|-------------|-----------------|-------------------|----------------|---------------|
| 8k (cost-optimized) | 1,000 (12.5%) | 1,000 (12.5%) | 4,000 (50%) | 1,500 (18.75%) |
| 16k (standard) | 1,500 (9.4%) | 2,000 (12.5%) | 9,000 (56%) | 2,000 (12.5%) |
| 32k (high quality) | 2,000 (6.25%) | 4,000 (12.5%) | 22,000 (68.75%) | 3,000 (9.4%) |
| 128k (premium) | 2,000 (1.6%) | 6,000 (4.7%) | 110,000 (85.9%) | 4,000 (3.1%) |

**Key principle**: Recent verbatim history should dominate for emotional continuity. Retrieved long-term memories are the secondary budget item. System prompt should be as concise as possible.

### 6.4 Dynamic Budget Allocation

Static allocation is suboptimal. Production systems should adjust based on query type:

```python
class ContextBudgetAllocator:
    def __init__(self, total_budget: int):
        self.total = total_budget

    def allocate(self, query: str, query_type: str) -> dict:
        """
        Dynamically adjust token allocation based on what the query needs.
        """
        base = {
            "system_prompt": 1200,
            "user_profile": 500,
            "output_buffer": 2000,
        }
        remaining = self.total - sum(base.values())

        if query_type == "factual":
            # Factual queries: more retrieved memories, less recent history
            return {**base,
                    "retrieved_memories": int(remaining * 0.60),
                    "recent_history": int(remaining * 0.30),
                    "session_summaries": int(remaining * 0.10)}

        elif query_type == "emotional_support":
            # Emotional queries: prioritize recent history for context
            return {**base,
                    "retrieved_memories": int(remaining * 0.20),
                    "recent_history": int(remaining * 0.65),
                    "session_summaries": int(remaining * 0.15)}

        elif query_type == "biographical":
            # Questions about user's background: maximize memory retrieval
            return {**base,
                    "retrieved_memories": int(remaining * 0.50),
                    "recent_history": int(remaining * 0.35),
                    "session_summaries": int(remaining * 0.15)}

        else:  # general conversation
            return {**base,
                    "retrieved_memories": int(remaining * 0.25),
                    "recent_history": int(remaining * 0.60),
                    "session_summaries": int(remaining * 0.15)}
```

### 6.5 The "Lost in the Middle" Placement Rule

Given the U-shaped attention pattern (Liu et al., 2024), placement within the context window matters as much as what you include:

```
OPTIMAL CONTEXT ORDERING (for companion AI)
════════════════════════════════════════════

[TOP of context — highest attention]
1. System prompt + agent persona
2. Core user profile (pinned biographical facts)
3. Current user turn (most important)

[MIDDLE — lower attention; avoid critical content here]
4. Retrieved long-term memories (less critical, can be partially lost)
5. Session summaries

[BOTTOM — high attention due to recency bias]
6. Last 5-10 verbatim conversation turns
7. [END] Final instruction: "Respond to the user's message above."
```

**Mitigation strategies for lost-in-the-middle**:
- Keep retrieved memories concise (150 tokens max per snippet)
- Use structured formatting (bullet points) to help the model scan
- Include the most critical retrieved fact at the top AND near the bottom
- Consider **Lost in the Middle** reordering: place the highest-scored memory first, lowest-scored second, second-highest third, etc. (puts high-value content near boundaries)

用中文总结: 本章提供了令牌预算分配的具体建议。在16k令牌预算下，推荐分配：系统提示1,500令牌（9.4%）、用户档案500令牌、检索到的长期记忆2,000令牌（12.5%）、近期对话历史9,000令牌（56%）、输出缓冲区2,000令牌。预算分配应根据查询类型动态调整——情感支持类查询应加大近期历史权重，而事实查询应加大检索记忆权重。此外，由于"lost in the middle"现象，应将最关键的内容放在上下文的开头或结尾，避免将重要信息置于中间位置。

---

## 7. Proactive vs. On-Demand Retrieval

### 7.1 The Design Decision

When should memory be retrieved?

- **On-demand (reactive)**: Retrieve when the user asks a question or when a keyword triggers retrieval
- **Proactive (automatic)**: Retrieve preemptively at the start of every turn, based on predicted information need

### 7.2 Proactive Memory Injection

**The dominant production approach** as of 2025-2026: retrieve memories automatically at the start of every turn, before the LLM sees the user message.

```python
async def handle_user_turn(user_message: str, user_id: str) -> str:
    """Proactive memory injection pipeline."""

    # 1. Immediately embed the user's message
    query_emb = await embed_model.encode_async(user_message)

    # 2. Proactively retrieve relevant memories (async, doesn't block)
    memories = await memory_store.retrieve(
        user_id=user_id,
        query_embedding=query_emb,
        query_text=user_message,
        top_k=10,
        strategy="hybrid_rrf"
    )

    # 3. Rerank to select best 5
    top_memories = reranker.rerank(user_message, memories, top_k=5)

    # 4. Assemble context (async-built while waiting for retrieval)
    context = context_builder.assemble(
        system_prompt=SYSTEM_PROMPT,
        user_profile=user_profiles[user_id],
        memories=top_memories,
        recent_history=conversation_store.get_recent(user_id, turns=20),
        current_message=user_message
    )

    # 5. Generate response
    response = await llm.generate_async(context)

    # 6. Write-back: update memory store async (don't block response)
    asyncio.create_task(
        memory_updater.process_turn(user_id, user_message, response)
    )

    return response
```

**Key architectural choice**: Memory writes are **async-by-default** — they run after the response is sent, so they never add latency.

### 7.3 LLM-Driven Self-Selection (MemGPT Model)

In MemGPT/Letta's architecture, the agent itself decides what to retrieve:

```python
# MemGPT-style: agent issues retrieval tool calls
AVAILABLE_TOOLS = [
    "archival_memory_search(query: str) -> list[Memory]",
    "conversation_search(query: str, page: int) -> list[Turn]",
    "memory_replace(block_label: str, new_content: str) -> None",
]

# The LLM sees tool schemas and can call:
# archival_memory_search("user's sister") -> returns relevant facts
# Then incorporate results into its response
```

**Tradeoffs of LLM self-selection**:

| Factor | Proactive Injection | LLM Self-Selection |
|--------|--------------------|--------------------|
| Latency | Fixed retrieval cost | Variable (1+ tool calls) |
| Recall | Depends on retrieval quality | Agent can refine queries |
| Precision | May include irrelevant memories | Agent filters by judgment |
| Cost | Lower (one retrieval round) | Higher (multiple LLM calls) |
| Failure mode | Silently misses relevant memories | Agent may hallucinate retrieval |
| Control | Predictable | Hard to audit |

### 7.4 MemGuide: Intent-Driven Memory Selection (2025)

MemGuide (arXiv 2505.20231, 2025) proposes a middle ground: use a lightweight intent classifier to determine which memory categories are relevant, then retrieve only from those categories.

```python
class MemGuide:
    """Intent-driven selective memory retrieval."""

    MEMORY_CATEGORIES = [
        "biographical",     # name, age, location
        "relationships",    # family, friends, partner
        "preferences",      # food, hobbies, entertainment
        "goals",            # career, personal aspirations
        "emotional_state",  # current feelings, recent mood
        "events",           # recent life events
    ]

    def select_categories(self, user_message: str) -> list[str]:
        """Lightweight classifier: which memory categories are relevant?"""
        prompt = f"""
        User message: "{user_message}"

        Which memory categories are most relevant to answer this message?
        Return only the relevant category names as a JSON list.
        Categories: {self.MEMORY_CATEGORIES}
        """
        relevant = self.fast_llm.classify(prompt)  # cheap, fast model
        return relevant

    async def retrieve(self, user_message: str, user_id: str) -> list[Memory]:
        categories = self.select_categories(user_message)
        memories = []
        for cat in categories:
            cat_memories = await memory_store.retrieve_by_category(
                user_id, cat, query=user_message, top_k=3
            )
            memories.extend(cat_memories)
        return memories
```

### 7.5 Recommendation for Companion AI

**Use proactive injection as the default**, with optional LLM-driven fallback:

```
RETRIEVAL DECISION FLOW
════════════════════════

User message arrives
      │
      ▼
[Always] Proactive: retrieve top-5 memories via hybrid search
      │
      ▼
Confidence check: are retrieved memories relevant? (threshold: > 0.6)
      │
    High ─────────────────────────────────────────────────────────────►
    confidence                                               Inject & generate
      │
    Low confidence or user asks about unfamiliar topic
      │
      ▼
[Optional] LLM receives retrieved memories + permission to call
           archival_memory_search() tool for additional retrieval
      │
      ▼
LLM issues additional retrieval calls if needed (max 2 rounds)
      │
      ▼
Generate response with enriched context
```

用中文总结: 本章比较了两种记忆检索触发机制：主动注入（每次对话自动检索）和LLM自主选择（让模型决定检索什么）。对AI伴侣产品，推荐以主动注入为默认策略——在每轮对话开始时自动检索top-5相关记忆，记忆写入操作异步执行以不增加响应延迟。当置信度较低时，可允许LLM通过工具调用进行补充检索（最多2轮）。MemGuide提供了一种中间方案：先用轻量分类器判断哪些记忆类别与当前查询相关，再仅从相关类别检索，降低噪音。

---

## 8. Evaluation Metrics & Benchmarks

### 8.1 Retrieval Quality Metrics

**Standard IR metrics applied to memory retrieval:**

| Metric | Formula | What It Measures |
|--------|---------|-----------------|
| **Precision@k** | relevant in top-k / k | How many retrieved memories are actually useful |
| **Recall@k** | relevant in top-k / total relevant | How many relevant memories were found |
| **NDCG@k** | Normalized discounted cumulative gain | Ranking quality, rewards relevant items ranked higher |
| **MRR** | 1 / rank of first relevant result | Quality of top-1 result |
| **MAP@k** | Mean average precision across queries | Overall system quality |

```python
def precision_at_k(retrieved: list, relevant: set, k: int) -> float:
    """Fraction of top-k retrieved items that are relevant."""
    top_k = retrieved[:k]
    relevant_retrieved = sum(1 for item in top_k if item in relevant)
    return relevant_retrieved / k

def recall_at_k(retrieved: list, relevant: set, k: int) -> float:
    """Fraction of relevant items found in top-k."""
    top_k = set(retrieved[:k])
    relevant_retrieved = len(top_k & relevant)
    return relevant_retrieved / len(relevant) if relevant else 0.0

def mean_reciprocal_rank(retrieved_lists: list, relevant_sets: list) -> float:
    """MRR across multiple queries."""
    reciprocal_ranks = []
    for retrieved, relevant in zip(retrieved_lists, relevant_sets):
        for rank, item in enumerate(retrieved, start=1):
            if item in relevant:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)
```

### 8.2 End-to-End Memory Quality Metrics

Beyond retrieval quality, production systems need end-to-end quality metrics:

| Metric | Description | How Measured |
|--------|-------------|--------------|
| **QA Accuracy** | Does retrieved memory enable correct answers? | Ground-truth Q&A pairs; LLM-as-judge |
| **Hallucination Rate on Recalled Facts** | Does the model invent memories? | Human annotation; contradiction detection |
| **Staleness Rate** | What % of injected memories are outdated? | Compare with ground truth user state |
| **Coverage Rate** | What % of user-disclosed facts are retrievable? | Inject known facts; test retrieval |
| **Context Efficiency** | Token cost per quality unit | Tokens used / QA accuracy |
| **Contradiction Rate** | Does injected memory contradict other memory? | Pairwise consistency check |

### 8.3 LongMemEval Benchmark (ICLR 2025)

The most important benchmark for long-term memory evaluation as of 2025-2026.

**arXiv 2410.10813** | ICLR 2025 | Xiaowu et al.

**Five evaluated abilities:**
1. **Information Extraction**: Retrieve specific facts from history
2. **Multi-Session Reasoning**: Connect facts across separate sessions
3. **Temporal Reasoning**: Understand "before/after," date-relative queries
4. **Knowledge Updates**: Recognize when information changes
5. **Abstention**: Correctly decline when queried information doesn't exist

**Dataset stats:**
- 500 curated questions
- LongMemEval_S: ~115k token histories
- LongMemEval_M: up to 1.5M token histories

**Key published results (from Zep paper and others):**

| System | Accuracy | Latency | Token Cost |
|--------|----------|---------|------------|
| Full context (GPT-4o) | ~72.9% | 17s P95 | Very high |
| Mem0 selective | 66.9% | 1.44s P95 | 90% lower |
| Zep (Graphiti) | +18.5% over baseline | < 300ms retrieval | Low |
| EMem-G | 77.9% | Fast | 1-3.6k tokens |
| HyperMem | 92.73% (LoCoMo) | Not reported | Low |

### 8.4 LoCoMo Benchmark

**SNAP Research, arXiv 2402.17753** | Multi-session dialogue evaluation

- Very long-term dialogues: up to 35 sessions, 9K tokens average, 300 turns
- Evaluates: single-hop, multi-hop, temporal reasoning, open-domain QA
- Key finding: RAG-based agents outperform long-context baselines, but "primary bottleneck is retrieval quality"

### 8.5 The Four-Layer Metric Stack

(From arXiv 2603.07670, Memory for Autonomous LLM Agents, 2026)

```
LAYER 4: Governance Metrics
  - Privacy leakage rate
  - Deletion compliance (right to be forgotten)
  - Access violation rate

LAYER 3: Efficiency Metrics
  - Latency per retrieval operation (target: < 300ms)
  - Tokens consumed per turn
  - Retrieval calls per session

LAYER 2: Memory Quality Metrics
  - Retrieval precision@5, recall@5
  - Contradiction rate
  - Staleness distribution

LAYER 1: Task Effectiveness Metrics
  - QA accuracy on known user facts
  - User satisfaction (via CSAT/NPS)
  - Factual consistency across sessions
```

### 8.6 Companion AI-Specific Evaluation

For an emotion-focused companion, additional metrics matter:

| Metric | Description | Measurement Method |
|--------|-------------|-------------------|
| **Emotional Continuity Score** | Does the system remember emotional context across sessions? | Human raters; session-to-session consistency |
| **Embarrassing Memory Rate** | How often does the system cite a fact the user changed? | A/B test with seed facts deliberately updated |
| **Proactive Reference Quality** | When the system volunteers a memory ("I remember you mentioned..."), is it appropriate? | Human annotation; user feedback |
| **False Memory Rate** | Does the system fabricate memories? | Inject false facts; test if they appear in responses |

用中文总结: 本章介绍了记忆检索质量的评估体系。标准IR指标（Precision@k、Recall@k、MRR、NDCG）适用于检索质量评估；端到端指标包括QA准确率、幻觉率、覆盖率和矛盾率。最重要的基准测试是LongMemEval（ICLR 2025），评估5项核心能力：信息提取、跨会话推理、时间推理、知识更新和拒答。对AI伴侣产品，还需关注情感连续性评分、"尴尬记忆率"（引用用户已更新的过时信息的频率）和虚假记忆率等专项指标。

---

## 9. Recommended Architecture for AI Companion

### 9.1 System Overview

Combining all research findings, the recommended read-path architecture for a Chinese AI companion:

```
COMPANION AI READ PATH ARCHITECTURE
═════════════════════════════════════

┌─────────────────────────────────────────────────────────────────┐
│                        USER TURN ARRIVES                         │
│                    (Chinese text message)                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  QUERY ANALYZER │
                    │                 │
                    │ - Detect type:  │
                    │   factual/      │
                    │   emotional/    │
                    │   biographical/ │
                    │   general       │
                    │ - Detect temporal│
                    │   references    │
                    │ - Embed query   │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
     ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
     │  BM25 Index │ │Dense Vector │ │  Temporal   │
     │  (keyword   │ │   Store     │ │   Filter    │
     │   search)   │ │ (semantic   │ │ (if temporal│
     │             │ │  search)    │ │  query)     │
     └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
            │               │               │
            └───────────────┼───────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │  RRF FUSION     │
                   │  top-50 merging │
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │   RERANKER      │
                   │ (BGE-Reranker   │
                   │  v2-m3 or       │
                   │  Cohere)        │
                   └────────┬────────┘
                            │
                            ▼ top-8 memories
                   ┌─────────────────────────────────────┐
                   │       MEMORY SCORER                  │
                   │                                      │
                   │  score = 0.40 * relevance            │
                   │        + 0.35 * recency_decay        │
                   │        + 0.25 * importance           │
                   │                                      │
                   │  Apply type-specific half-lives:     │
                   │   mood: 3 days                       │
                   │   events: 14 days                    │
                   │   preferences: 90 days               │
                   │   biography: no decay (pinned)       │
                   └────────┬────────────────────────────┘
                            │ top-5 scored memories
                            ▼
                   ┌─────────────────────────────────────┐
                   │     CONTEXT ASSEMBLER                │
                   │                                      │
                   │  Placement (anti-lost-in-middle):    │
                   │   [TOP] system prompt + user profile │
                   │   [TOP] highest-scored memory        │
                   │   [MID] session summaries (2-3)      │
                   │   [MID] memories #2-4                │
                   │   [BOT] recent 20 turns verbatim     │
                   │   [BOT] memory #5 + current query    │
                   └────────┬────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │     LLM       │
                    │  (optional:   │
                    │  memory tool  │
                    │  calls if     │
                    │  low confidence│
                    └───────┬───────┘
                            │
                            ▼
                        RESPONSE
                            │
                            ▼
               [Async] Memory write-back pipeline
               (signal extraction → lifecycle update
                → entity resolution → store)
```

### 9.2 Complete Pseudocode

```python
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

# ─── Data Models ───────────────────────────────────────────────

@dataclass
class Memory:
    memory_id: str
    user_id: str
    content: str                    # human-readable fact
    memory_type: str                # "fact", "episode", "preference", "mood"
    entity_refs: list[str]          # entity IDs this memory relates to
    embedding: list[float]          # dense vector (1536-dim)
    keywords: list[str]             # for BM25 indexing
    created_at: datetime
    updated_at: datetime
    importance: float               # 1.0–10.0, LLM-assigned at write time
    pinned: bool = False            # if True, always include, no decay
    valid_until: Optional[datetime] = None  # for time-limited facts
    source_session_id: str = ""

    def half_life_hours(self) -> float:
        """Type-specific decay rate."""
        rates = {
            "mood": 72,         # 3 days
            "event": 336,       # 14 days
            "preference": 2160, # 90 days
            "fact": 4320,       # 180 days
            "biography": float('inf'),  # permanent
        }
        return rates.get(self.memory_type, 720)  # default 30 days

    def recency_score(self, now: datetime) -> float:
        if self.pinned:
            return 1.0
        hl = self.half_life_hours()
        if hl == float('inf'):
            return 1.0
        hours_elapsed = (now - self.created_at).total_seconds() / 3600
        return 0.5 ** (hours_elapsed / hl)

@dataclass
class ContextBudget:
    total: int
    system_prompt: int
    user_profile: int
    retrieved_memories: int
    session_summaries: int
    recent_history: int
    output_buffer: int

@dataclass
class SessionSummary:
    session_id: str
    user_id: str
    summary_text: str           # LLM-generated, 150-300 tokens
    key_facts: list[str]        # bullet list of new facts disclosed
    emotional_tone: str         # "positive", "neutral", "distressed"
    session_date: datetime
    topics: list[str]

# ─── Core Read Pipeline ─────────────────────────────────────────

class CompanionReadPipeline:

    def __init__(self,
                 embed_model,
                 bm25_index,
                 vector_store,
                 reranker,
                 memory_store,
                 conversation_store,
                 llm,
                 budget: ContextBudget):
        self.embed_model = embed_model
        self.bm25 = bm25_index
        self.vectors = vector_store
        self.reranker = reranker
        self.memory_store = memory_store
        self.conv_store = conversation_store
        self.llm = llm
        self.budget = budget

    async def handle_turn(self, user_id: str, message: str) -> str:

        now = datetime.utcnow()

        # ── Step 1: Query analysis ──────────────────────────────
        query_type = self._classify_query(message)       # fast heuristic
        temporal_range = self._extract_temporal(message) # None or (start, end)
        query_emb = await self.embed_model.encode_async(message)

        # ── Step 2: Retrieve candidates (parallel) ──────────────
        bm25_hits, dense_hits = await asyncio.gather(
            self.bm25.search_async(message, user_id=user_id, top_k=25),
            self.vectors.search_async(query_emb, user_id=user_id, top_k=25)
        )

        # Apply temporal filter if query references a time
        if temporal_range:
            bm25_hits = self._temporal_filter(bm25_hits, temporal_range)
            dense_hits = self._temporal_filter(dense_hits, temporal_range)

        # ── Step 3: Fuse with RRF ───────────────────────────────
        candidates = self._rrf_merge(bm25_hits, dense_hits, k=60)[:50]

        # ── Step 4: Rerank top-50 → top-8 ──────────────────────
        reranked = await self.reranker.rerank_async(
            message, candidates, top_k=8
        )

        # ── Step 5: Score by recency + importance + relevance ───
        scored = []
        for mem, rerank_score in reranked:
            final_score = (
                0.40 * rerank_score +                      # relevance
                0.35 * mem.recency_score(now) +            # recency
                0.25 * (mem.importance / 10.0)             # importance
            )
            scored.append((mem, final_score))
        scored.sort(key=lambda x: x[1], reverse=True)

        top_memories = [mem for mem, _ in scored[:5]]

        # Always include pinned memories (add at top, don't count against budget)
        pinned = await self.memory_store.get_pinned_async(user_id)

        # ── Step 6: Retrieve session summaries ─────────────────
        recent_summaries = await self.conv_store.get_session_summaries_async(
            user_id, last_n=3
        )

        # ── Step 7: Get recent verbatim history ─────────────────
        recent_turns = await self.conv_store.get_recent_turns_async(
            user_id, last_n=20
        )

        # ── Step 8: Assemble context (anti-lost-in-middle) ───────
        context = self._assemble_context(
            pinned_memories=pinned,
            top_memories=top_memories,
            session_summaries=recent_summaries,
            recent_turns=recent_turns,
            current_message=message,
            query_type=query_type,
            budget=self._dynamic_budget(query_type)
        )

        # ── Step 9: Generate response ────────────────────────────
        response = await self.llm.generate_async(
            context=context,
            tools=["archival_memory_search"] if self._needs_tool(scored) else []
        )

        # ── Step 10: Async write-back (non-blocking) ─────────────
        asyncio.create_task(
            self._write_back(user_id, message, response, now)
        )

        return response

    def _assemble_context(self, pinned_memories, top_memories,
                          session_summaries, recent_turns,
                          current_message, query_type, budget) -> str:
        """
        Assemble context with anti-lost-in-middle placement:
        - Critical content at TOP and BOTTOM
        - Less critical content in MIDDLE
        """
        parts = []

        # TOP: system prompt + pinned profile
        parts.append(SYSTEM_PROMPT)
        parts.append(self._format_pinned(pinned_memories))

        # TOP: highest-scored memory (position 1 = high attention)
        if top_memories:
            parts.append(f"[Key memory] {top_memories[0].content}")

        # MIDDLE: session summaries (lower attention is OK for summaries)
        for summary in session_summaries:
            parts.append(f"[{summary.session_date.strftime('%Y-%m-%d')} session]: "
                        f"{summary.summary_text}")

        # MIDDLE: remaining memories (positions 2-4)
        for mem in top_memories[1:]:
            parts.append(f"[Memory] {mem.content}")

        # BOTTOM: recent turns (high attention due to recency in transformer)
        parts.append(self._format_turns(recent_turns, budget.recent_history))

        # BOTTOM: current user message (very high attention)
        parts.append(f"User: {current_message}")

        # BOTTOM: closing instruction (highest attention position)
        parts.append("Reply to the user above. Be warm, personalized, and consistent "
                     "with what you know about them.")

        return "\n\n".join(parts)

    def _dynamic_budget(self, query_type: str) -> ContextBudget:
        """Adjust allocation based on what this query needs."""
        base_total = self.budget.total
        reserved = self.budget.system_prompt + self.budget.user_profile + self.budget.output_buffer
        remaining = base_total - reserved

        allocations = {
            "factual":           (0.60, 0.10, 0.30),  # (memories, summaries, history)
            "emotional_support": (0.20, 0.15, 0.65),
            "biographical":      (0.50, 0.15, 0.35),
            "general":           (0.25, 0.15, 0.60),
        }
        m_frac, s_frac, h_frac = allocations.get(query_type, (0.25, 0.15, 0.60))

        return ContextBudget(
            total=base_total,
            system_prompt=self.budget.system_prompt,
            user_profile=self.budget.user_profile,
            retrieved_memories=int(remaining * m_frac),
            session_summaries=int(remaining * s_frac),
            recent_history=int(remaining * h_frac),
            output_buffer=self.budget.output_buffer
        )

    def _rrf_merge(self, list1, list2, k=60):
        from collections import defaultdict
        scores = defaultdict(float)
        id_to_obj = {}
        for ranked_list in [list1, list2]:
            for rank, (obj, _) in enumerate(ranked_list, start=1):
                scores[obj.memory_id] += 1.0 / (k + rank)
                id_to_obj[obj.memory_id] = obj
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [(id_to_obj[mid], scores[mid]) for mid in sorted_ids]

    def _needs_tool(self, scored_memories) -> bool:
        """Should we give the LLM retrieval tools for further search?"""
        if not scored_memories:
            return True
        top_score = scored_memories[0][1]
        return top_score < 0.55  # low confidence → allow tool calls

    async def _write_back(self, user_id, message, response, now):
        """Async memory update — runs after response is sent."""
        await signal_extractor.process(user_id, message, response, now)
        # Triggers lifecycle manager and entity resolver in pipeline

    def _temporal_filter(self, memories, time_range):
        start, end = time_range
        return [(m, s) for m, s in memories
                if start <= m.created_at <= end]

    def _classify_query(self, message: str) -> str:
        """Fast heuristic query classifier (no LLM needed)."""
        emotional_keywords = {"感觉", "心情", "难过", "开心", "压力", "焦虑", "担心"}
        factual_keywords = {"是什么", "多少", "哪里", "什么时候", "谁", "为什么"}
        biographical_keywords = {"我的", "家人", "工作", "学校", "住"}

        msg_lower = message.lower()
        if any(k in msg_lower for k in emotional_keywords):
            return "emotional_support"
        elif any(k in msg_lower for k in factual_keywords):
            return "factual"
        elif any(k in msg_lower for k in biographical_keywords):
            return "biographical"
        return "general"

    def _extract_temporal(self, message: str):
        """Extract date ranges from temporal language."""
        # e.g., "上周" → (last_monday, last_sunday)
        # "最近" → (7 days ago, now)
        # Implementation: regex + date parser
        pass  # Full implementation in production code
```

### 9.3 Session Summary Generation

Generated at session end, stored for future retrieval:

```python
SESSION_SUMMARY_PROMPT = """
You are summarizing a conversation for a Chinese AI companion's memory system.
The companion needs to remember what the user shared for future conversations.

Focus on:
1. New personal facts the user disclosed (name, relationships, events)
2. The user's current emotional state and concerns
3. Topics discussed and any preferences expressed
4. Any significant life events or changes mentioned

Write a concise summary (150-250 words) that will help the companion be
personalized and caring in future conversations. Write in natural language.

Also list:
- KEY_FACTS: [list of discrete facts to store separately]
- EMOTIONAL_TONE: [positive/neutral/distressed/mixed]
- TOPICS: [list of topics covered]

Conversation:
{conversation}
"""

async def generate_session_summary(session_turns, user_id, llm):
    summary_response = await llm.generate_async(
        SESSION_SUMMARY_PROMPT.format(
            conversation=format_turns(session_turns)
        )
    )
    # Parse and store in SessionSummary; extract key_facts for memory store
    return parse_summary(summary_response, user_id)
```

### 9.4 Technology Stack Recommendations

| Component | Recommended Option | Alternative |
|-----------|-------------------|-------------|
| **Embedding model** | Qwen3-Embedding-4B (self-hosted) | OpenAI text-embedding-3-small |
| **Vector store** | Qdrant (self-hosted) | Weaviate, Pinecone |
| **BM25 index** | Elasticsearch / OpenSearch | BM25s Python library |
| **Reranker** | BGE-Reranker-v2-m3 | Cohere Rerank API |
| **Session summarizer** | Qwen2.5-7B (cost efficient) | GPT-4o-mini |
| **Main LLM** | Claude Sonnet / GPT-4o | Qwen2.5-72B |
| **Memory store** | PostgreSQL + pgvector | MongoDB Atlas |
| **Orchestration** | Custom async pipeline | Letta framework |

### 9.5 Performance Targets

| Metric | Target | Stretch Target |
|--------|--------|----------------|
| Memory retrieval latency (P50) | < 100ms | < 50ms |
| Memory retrieval latency (P95) | < 300ms | < 150ms |
| Context assembly latency | < 50ms | < 20ms |
| Total time-to-first-token | < 800ms | < 500ms |
| Memory QA accuracy (LongMemEval-style) | > 70% | > 80% |
| Retrieval Recall@5 | > 75% | > 85% |
| Hallucination rate on recalled facts | < 5% | < 2% |

用中文总结: 本章提供了AI伴侣"读取路径"的完整推荐架构，包括：（1）查询分析器——分类查询类型、检测时间引用、嵌入查询；（2）混合检索——并行BM25关键词检索和Dense向量检索，通过RRF融合；（3）重排序——从top-50精选到top-8；（4）记忆评分——综合相关性（0.40）、时近性（0.35，按记忆类型设置不同半衰期）和重要性（0.25）；（5）上下文组装——遵循"anti-lost-in-middle"原则，将最重要内容放在头部和尾部；（6）异步写回——响应发送后异步更新记忆存储，不增加延迟。推荐的技术栈包括：Qwen3-Embedding嵌入模型、Qdrant向量存储、BGE-Reranker-v2-m3重排序模型。

---

## 10. References

### Papers (arXiv and Peer-Reviewed)

1. **Liu et al., 2024** — "Lost in the Middle: How Language Models Use Long Contexts." *Transactions of the Association for Computational Linguistics* (TACL 2024). arXiv:2307.03172. Foundational work on positional bias in LLM attention. [Link](https://arxiv.org/abs/2307.03172)

2. **Packer et al., 2023** — "MemGPT: Towards LLMs as Operating Systems." arXiv:2310.08560. Introduced virtual context management with two-tier memory architecture. [Link](https://arxiv.org/abs/2310.08560)

3. **Rasmussen et al., 2025** — "Zep: A Temporal Knowledge Graph Architecture for Agent Memory." arXiv:2501.13956 (January 2025). Graphiti temporal knowledge graph; 94.8% DMR, 18.5% LongMemEval improvement. [Link](https://arxiv.org/abs/2501.13956)

4. **Wang et al., 2025** — "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory." arXiv:2410.10813. ICLR 2025. Key benchmark with five evaluation dimensions. [Link](https://arxiv.org/abs/2410.10813)

5. **Zhong et al., 2025** — "Recursively Summarizing Enables Long-Term Dialogue Memory in Large Language Models." arXiv:2308.15022. Published in *Neurocomputing* 2025. [Link](https://arxiv.org/abs/2308.15022)

6. **Anonymous, 2025** — "A Simple Yet Strong Baseline for Long-Term Conversational Memory of LLM Agents" (EMem paper). arXiv:2511.17208 (November 2025). EDU-based memory, 77.9% LongMemEval_S accuracy. [Link](https://arxiv.org/abs/2511.17208)

7. **Anonymous, 2026** — "HyperMem: Hypergraph Memory for Long-Term Conversations." arXiv:2604.08256 (April 2026). Three-level hypergraph hierarchy; 92.73% LoCoMo accuracy. [Link](https://arxiv.org/abs/2604.08256)

8. **Anonymous, 2025** — "SGMem: Sentence Graph Memory for Long-Term Conversational Agents." arXiv:2509.21212 (September 2025). Sentence-level graph memory. [Link](https://arxiv.org/abs/2509.21212)

9. **Anonymous, 2025** — "Solving Freshness in RAG: A Simple Recency Prior and the Limits of Heuristic Trend Detection." arXiv:2509.19376. Fused semantic-temporal scoring formula. [Link](https://arxiv.org/abs/2509.19376)

10. **Anonymous, 2025** — "Relevance Isn't All You Need." arXiv:2504.07104. ICLR 2025. Multi-criteria reranking beyond relevance. [Link](https://arxiv.org/pdf/2504.07104)

11. **Park et al., 2023** — "Generative Agents: Interactive Simulacra of Human Behavior." (Foundational work on recency + importance + relevance scoring formula for memory.) ACM CHI 2023.

12. **Anonymous, 2026** — "Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers." arXiv:2603.07670 (March 2026). Comprehensive survey; four-layer metric stack; three architectural patterns. [Link](https://arxiv.org/abs/2603.07670)

13. **Anonymous, 2025** — "MemGuide: Intent-Driven Memory Selection for Goal-Oriented Multi-Session LLM Agents." arXiv:2505.20231 (May 2025). [Link](https://arxiv.org/abs/2505.20231)

14. **Anonymous, 2026** — "Cognitive Memory in Large Language Models." arXiv:2504.02441 (April 2026). Cognitive architecture framework for LLM memory. [Link](https://arxiv.org/abs/2504.02441)

15. **MemoRAG, 2025** — "MemoRAG: Boosting Long Context Processing with Global Memory-Enhanced Retrieval Augmentation." *ACM Web Conference 2025*. arXiv:2409.05591. [Link](https://arxiv.org/abs/2409.05591)

### Production Systems & Blogs

16. **Letta/MemGPT** (2024–2026). "Memory Blocks: The Key to Agentic Context Management." [https://www.letta.com/blog/memory-blocks](https://www.letta.com/blog/memory-blocks)

17. **Letta** (2025). "Anatomy of a Context Window: A Guide to Context Engineering." [https://www.letta.com/blog/guide-to-context-engineering](https://www.letta.com/blog/guide-to-context-engineering)

18. **Mem0** (2026). "State of AI Agent Memory 2026." [https://mem0.ai/blog/state-of-ai-agent-memory-2026](https://mem0.ai/blog/state-of-ai-agent-memory-2026)

19. **Getzep/Graphiti** (2025). GitHub repo. [https://github.com/getzep/graphiti](https://github.com/getzep/graphiti)

20. **OpenAI** (2024–2025). "Memory and New Controls for ChatGPT." [https://openai.com/index/memory-and-new-controls-for-chatgpt/](https://openai.com/index/memory-and-new-controls-for-chatgpt/)

21. **Weaviate** (2025). "Context Engineering — LLM Memory and Retrieval for AI Agents." [https://weaviate.io/blog/context-engineering](https://weaviate.io/blog/context-engineering)

22. **Factory.ai** (2025). "Compressing Context." [https://factory.ai/news/compressing-context](https://factory.ai/news/compressing-context)

23. **SNAP Research** (2024). "LoCoMo: Evaluating Very Long-Term Conversational Memory of LLM Agents." [https://snap-research.github.io/locomo/](https://snap-research.github.io/locomo/)

24. **Frontiers in Psychology** (2025). "Enhancing Memory Retrieval in Generative Agents through LLM-Trained Cross Attention Networks." PMC12092450. [Link](https://pmc.ncbi.nlm.nih.gov/articles/PMC12092450/)

---

*Document authored: April 2026*
*Research tracks covered: Signal Extraction (Track 1), Lifecycle Management (Track 2), Entity Resolution (Track 3), Context Management (Track 4 — this document)*
*Next step: Integrate Track 4 findings into the Final Synthesis document.*
