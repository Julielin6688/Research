# Memory System Architecture: Final Synthesis
# 记忆系统架构：综合研究报告

> Synthesized from four parallel research tracks | April 2026
> For: AI Companion Product Team

---

## Executive Summary

We researched four core problems in user memory management for a Chinese-language AI companion product. This document synthesizes findings from all four tracks — signal extraction, lifecycle management, entity resolution, and context window management — into a single unified architecture recommendation.

**Key finding**: The four problems form two complementary paths. The **write path** (signal extraction → entity resolution → lifecycle management) handles how memories are created, updated, and kept coherent. The **read path** (context window management → retrieval → assembly) handles how memories are selected and delivered to the LLM at inference time. Both paths must be designed together, as retrieval quality is only as good as what was written, and write-path quality is wasted if retrieval fails to surface the right memories.

用中文总结: 四个子问题（信号提取、生命周期管理、实体消歧、上下文窗口管理）构成两条互补路径：写入路径（记忆如何生成、更新、去重）和读取路径（推理时如何检索并装配正确的记忆）。本文档将四项研究的核心发现整合为统一的架构建议。

---

## 1. Summary of Findings per Track

### 1.1 Signal Extraction (What to Remember)

**Core finding**: A two-stage pipeline — cheap rule-based pre-filter followed by an LLM extractor — achieves the best cost/quality tradeoff. The six-category taxonomy (biographical, preference, relationship, goal, emotional state, event) covers ~95% of memory-worthy content in companion settings.

Key numbers from research:
- Pre-filtering eliminates **40–60%** of utterances before LLM extraction
- Self-disclosure detection (first-person + personal pronoun patterns) achieves **~78% precision**
- Extracting every **2–3 turns** (rather than every turn) reduces latency and gives the LLM more context to work with

**Chinese-specific implication**: Chinese companions generate more implicit self-disclosure (statements without 我/I due to pro-drop), so the pre-filter must be tuned differently — don't discard short utterances that lack explicit pronouns.

用中文总结: 两阶段管道（规则预过滤+LLM提取）为最优方案。六类分类体系覆盖大多数有价值信号。中文因零代词特性，预过滤规则需要针对性调整，不能简单过滤掉不含"我"的短句。

### 1.2 Lifecycle Management (Keeping Memories Fresh)

**Core finding**: Append-only versioning (never hard-delete, only transition state) combined with per-category TTL and decay is the right foundation. The staleness problem is solved by a combination of (a) automatic conflict detection at write time and (b) explicit user correction signals treated as the highest-trust input.

Key design decisions from production systems:
- **Mem0** favors a single LLM call that does extraction + conflict detection together — simpler, but loses history
- **Letta/MemGPT** is agent-driven — powerful but slow and non-deterministic
- **Zep** maintains a versioned fact graph — most aligned with what we need

The right resolution: versioned storage (like Zep) with single-call LLM update detection (like Mem0) for speed.

**Decay policy recommendation**:

| Category | TTL | Decay Half-Life |
|---|---|---|
| Biographical | Never | ∞ |
| Preference | Never | 180 days |
| Relationship | Never | ∞ |
| Goal | 180 days | 60 days |
| Emotional state | 14 days | 7 days |
| Event | 365 days | 90 days |

用中文总结: 仅追加版本控制+按类别TTL衰减是核心架构。写入时检测冲突，用户主动纠正视为最高信任信号。Zep的版本图思路最接近需求，结合Mem0的单次LLM调用效率。

### 1.3 Context Window Management (The Read Path)

**Core finding**: Writing high-quality memories is necessary but not sufficient — retrieval quality is the primary bottleneck in production systems (LoCoMo benchmark finding, 2025). The read path must solve three competing forces: **relevance** (surface topically related memories), **recency** (prefer recent updates over stale facts), and **token budget** (fit everything in the LLM's context window without degrading attention quality).

Key results from research:
- "Lost in the Middle" (Liu et al., TACL 2024): LLMs show a U-shaped attention pattern — content placed in the **middle** of long contexts suffers a **30%+ accuracy drop**. Critical memories must be placed at the top or bottom of context.
- Pure semantic (dense) retrieval achieves only **~49% recall**. Hybrid search (BM25 + dense, fused via Reciprocal Rank Fusion) achieves **~53% recall** — Zep's production system shows **18.5% accuracy improvement** and **90% latency reduction** over full-context approaches.
- Temporal decay is essential, not marginal: a simple recency-weighted scoring formula (half-life per memory type) achieved perfect accuracy on freshness tasks where pure semantic retrieval scored 0.00.
- **Token budget recommendation** (16k window): system prompt 9%, user profile 4%, retrieved memories 12.5%, session summaries 6%, recent verbatim history 56%, output buffer 12.5%.

**Three-level memory hierarchy**: working memory (current session verbatim turns) → session summaries (LLM-generated at session end, 150-300 tokens each) → long-term fact store (retrieved via hybrid search). The EMem system (2025) shows EDU-based compression reduces context from **101k → 1-3.6k tokens** while maintaining 77.9% QA accuracy.

用中文总结: 读取路径的核心挑战是在相关性、时近性和令牌预算之间取得平衡。生产系统的检索瓶颈远比存储本身重要。混合检索（BM25+Dense+重排序）是当前最优方案，Zep的生产数据显示它比全上下文方法准确率高18.5%、延迟低90%。"lost in the middle"现象要求将关键记忆放置在上下文首尾，而非中间。分层记忆架构（当前会话→会话摘要→长期事实库）是管理无限增长历史的核心设计。

### 1.4 Entity Resolution (Keeping People Straight)

**Core finding**: Chinese entity resolution is significantly harder than English due to six compounding factors: pro-drop, no word boundaries, high name-collision rates (top 3 surnames cover 22% of population), no capitalization, multiple surface forms per person, and granular kinship terms. The disambiguation pipeline must handle all six.

Recommended disambiguation scoring weights:
- Alias / name string match: **35%**
- Embedding cosine similarity: **30%**
- Relationship role consistency: **20%**
- Recency of last mention: **10%**
- Attribute coherence: **5%**

**Critical design principle**: Default to splitting, not merging. Creating an extra entity is cheap; conflating two real people causes irreversible trust damage. Merge only with confidence > 0.9 or explicit user confirmation.

用中文总结: 中文实体消歧面临零代词、无词间空格、同名率高等六大独特挑战。消歧打分综合五个信号，权重以别名匹配和语义相似度为主。核心原则：默认拆分而非合并，避免将不同人混淆。

---

## 2. The Four Core Problems Are Interconnected

```
                   ┌──────────────────────────┐
                   │    User Conversation Turn │
                   └──────────┬───────────────┘
                              │
                    ┌─────────▼──────────┐
               ┌────│  Signal Extractor  │────┐
               │    └─────────┬──────────┘    │
               │              │               │
               │    "User's boyfriend Zhang   │
               │     Wei got promoted"        │
               │              │               │
               │    ┌─────────▼──────────┐    │
               │    │  Entity Resolver   │◄───┘
               │    └─────────┬──────────┘
               │              │
               │    Entity: Zhang Wei (ent_001)
               │    Relation: ROMANTIC_PARTNER
               │              │
               │    ┌─────────▼──────────┐
               └───►│ Lifecycle Manager  │
                    └─────────┬──────────┘
                              │
                  ┌───────────┼───────────┐
                  ▼           ▼           ▼
            [Check for   [Update Zhang  [Update
             conflicts]   Wei's career   relationship
                          attribute]     temporal edge]
```

The three write-path modules share a single pipeline. The extractor identifies what happened; the resolver identifies to whom; the lifecycle manager decides how to store it. The fourth module — context management — operates on the **read path**: it retrieves the right memories from the store and assembles them into the LLM context at inference time.

```
                   ┌─────────────────────────────┐
                   │       READ PATH              │
                   │  User message arrives        │
                   │         │                    │
                   │  [Query Analyzer]             │
                   │  embed + classify + temporal  │
                   │         │                    │
                   │  ┌──────┴──────┐             │
                   │  BM25    Dense │ (parallel)   │
                   │  search  search│             │
                   │  └──────┬──────┘             │
                   │   RRF Fusion → Reranker       │
                   │         │                    │
                   │  Recency+Importance+Relevance │
                   │  scoring (type-specific decay)│
                   │         │                    │
                   │  [Context Assembler]          │
                   │  anti-lost-in-middle placement│
                   │         │                    │
                   │     LLM Response              │
                   │         │                    │
                   │  [Async write-back]           │
                   │  → WRITE PATH below ──────────┤
                   └─────────────────────────────┘
```

用中文总结: 四个模块构成两条路径：写入路径（信号提取→实体解析→生命周期管理）负责记忆的生成和存储；读取路径（查询分析→混合检索→重排序→上下文装配）负责在推理时选取正确记忆。写入操作异步执行，不影响响应延迟。

---

## 3. Unified Architecture

### 3.1 System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         API LAYER                               │
│  POST /memory/process-turn  │  GET /memory/retrieve  │  PATCH   │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                      PROCESSING PIPELINE                         │
│                                                                   │
│  1. Pre-filter      2. Signal Extract    3. Entity Resolve       │
│  ┌──────────┐       ┌──────────────┐    ┌──────────────────┐    │
│  │Rule-based│──────►│LLM extractor │───►│ NER → Candidate  │    │
│  │gate      │       │(Haiku/mini)  │    │ Retrieval →      │    │
│  │~50% drop │       │2-3 turns     │    │ Score → Decide   │    │
│  └──────────┘       └──────┬───────┘    └────────┬─────────┘    │
│                            │                     │              │
│                    extracted memories     resolved entity IDs   │
│                            │                     │              │
│                    4. Lifecycle Manager           │              │
│                    ┌────────────────────────────┐│              │
│                    │ Conflict detection          ││              │
│                    │ → supersede / merge / flag  ││              │
│                    │ Versioned write             ││              │
│                    │ TTL sweep (daily)           ││              │
│                    └────────────┬───────────────┘│              │
│                                 │                │              │
└─────────────────────────────────┼────────────────┼─────────────┘
                                  │                │
┌─────────────────────────────────▼────────────────▼─────────────┐
│                         STORAGE LAYER                            │
│                                                                   │
│  ┌─────────────────┐  ┌───────────────┐  ┌───────────────────┐  │
│  │  Vector Store   │  │  Memory Store │  │   Entity Graph    │  │
│  │  (embeddings    │  │  (versioned   │  │   (Neo4j /        │  │
│  │   for retrieval)│  │   facts, TTL) │  │    temporal       │  │
│  │  Qdrant / FAISS │  │  PostgreSQL / │  │    edges)         │  │
│  └─────────────────┘  │  SQLite       │  └───────────────────┘  │
│                        └───────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Read Path Architecture

The read path operates at inference time, before each LLM response. It must complete within the latency budget (target: < 300ms for retrieval, < 2s total).

```
READ PATH (per user turn)
══════════════════════════

User message
    │
    ▼
[1] Query Analyzer (< 10ms, rule-based)
    ├── Classify: factual / emotional / biographical / general
    ├── Detect temporal references ("last week", "recently")
    └── Embed query (async, BGE-M3 or Qwen3-Embedding)
    │
    ▼
[2] Hybrid Retrieval (parallel, < 100ms)
    ├── BM25 keyword search → top-25 candidates
    └── Dense vector ANN search → top-25 candidates
    │
    ├── [If temporal query] → apply temporal date-range filter first
    │
    ▼
[3] RRF Fusion → top-50 merged candidates
    │
    ▼
[4] Reranker (BGE-Reranker-v2-m3 or Cohere) → top-8
    │
    ▼
[5] Final Scorer: 0.40 × relevance + 0.35 × recency_decay + 0.25 × importance
    Per-type half-lives:
      mood: 72h | event: 14d | preference: 90d | biography: ∞ (pinned)
    → top-5 memories selected
    │
    ▼
[6] Context Assembler (anti-lost-in-middle ordering)
    ┌───────────────────────────────────────────────────┐
    │ [TOP]    System prompt + agent persona             │
    │ [TOP]    Core user profile (always pinned)         │
    │ [TOP]    Highest-scored memory (#1)               │
    │ [MIDDLE] Session summaries (2–3 most recent)      │
    │ [MIDDLE] Memories #2–#4                           │
    │ [BOTTOM] Last 20 verbatim turns                   │
    │ [BOTTOM] Memory #5 + current user message         │
    └───────────────────────────────────────────────────┘
    │
    ▼
LLM Response (+ optional tool-call retrieval if confidence < 0.6)
    │
    ▼ [Async, non-blocking]
Write-back pipeline (signal extraction → entity resolution → lifecycle write)
```

**Token budget (16k window):**

| Slot | Tokens | % |
|---|---|---|
| System prompt | 1,500 | 9.4% |
| Core user profile (pinned) | 500 | 3.1% |
| Retrieved long-term memories (top-5) | 2,000 | 12.5% |
| Session summaries (2-3 sessions) | 1,000 | 6.3% |
| Recent verbatim turns (last ~20) | 9,000 | 56.3% |
| Output buffer | 2,000 | 12.5% |

**Dynamic allocation**: For emotional support queries, increase recent history to 65% and reduce retrieved memories to 20%. For factual/biographical queries, flip the ratio: retrieved memories 50%, recent history 35%.

用中文总结: 读取路径在每次推理前执行，目标延迟<300ms。核心步骤：查询分析（分类+时间检测+向量化）→并行BM25+Dense混合检索（各top-25）→RRF融合→重排序（top-8）→三分量评分（相关性0.40+时近性0.35+重要性0.25）→上下文装配（按anti-lost-in-middle顺序排列）。对话历史（56%）是最大令牌消费项，按查询类型动态调整各槽位比例。

### 3.3 Unified Data Model

```typescript
// The central record type — connects all four subsystems
interface MemoryRecord {
  // Identity
  id: string;                      // Immutable UUID
  user_id: string;

  // Content (from Signal Extractor)
  content: string;                 // Normalized fact, third-person
  raw_source: string;              // Original utterance
  source_turn_id: string;
  category: MemoryCategory;        // biographical | preference | relationship | goal | emotional_state | event
  confidence: number;              // Extractor confidence, 0–1

  // Entity links (from Entity Resolver)
  entities: EntityRef[];           // canonical entity IDs + roles
  primary_entity_id?: string;      // The main person this memory is about

  // Lifecycle (from Lifecycle Manager)
  version: number;
  status: "active" | "superseded" | "archived" | "expired" | "flagged";
  parent_id?: string;              // Memory this supersedes
  superseded_by?: string;
  created_at: Date;
  expires_at?: Date;
  salience: number;                // Computed, decays over time
  access_count: number;

  // Context retrieval (from Context Manager)
  embedding: float[];              // Dense vector for semantic retrieval (BGE-M3 / Qwen3-Embedding)
  keywords: string[];              // BM25 index terms (extracted at write time)
  importance: number;              // 1.0–10.0, LLM-assigned at write time; used in retrieval scoring
  pinned: boolean;                 // If true, always included in context; no decay applied
  half_life_type: "mood" | "event" | "preference" | "fact" | "biography";
                                   // Controls recency decay rate at retrieval time
}

interface EntityRef {
  canonical_id: string;            // Points to Entity record
  role_in_memory: string;          // "subject", "mentioned", "location"
  surface_form: string;            // How user referred to them
  resolution_confidence: number;
}
```

### 3.4 Unified Write Pipeline (Pseudocode)

```python
def process_turn(
    turn: ConversationTurn,
    user_id: str,
    recent_context: list[ConversationTurn],
    stores: StorageLayer
) -> ProcessingResult:

    # Stage 1: Pre-filter (rule-based, fast)
    if not pre_filter(turn.user_text):
        return ProcessingResult(skipped=True, reason="pre_filter")

    # Stage 2: Signal extraction (LLM, batched every 2-3 turns)
    raw_memories = extract_signals(
        turns=recent_context[-3:] + [turn],
        llm=fast_llm  # Claude Haiku / GPT-4o-mini
    )
    if not raw_memories:
        return ProcessingResult(skipped=True, reason="no_signal")

    results = []
    for raw in raw_memories:

        # Stage 3: Entity resolution
        raw["entities"] = resolve_entities(
            content=raw["content"],
            context=recent_context,
            entity_store=stores.entity_graph
        )
        raw["embedding"] = embed(raw["content"])

        # Stage 4: Conflict detection + lifecycle write
        conflict_candidates = get_conflict_candidates(raw, stores.memory_store)
        conflicts = detect_conflicts(raw, conflict_candidates, llm=fast_llm)

        if not conflicts:
            stores.memory_store.insert(raw, status="active")
            action = "inserted"
        else:
            resolution = resolve_conflicts(raw, conflicts)
            apply_resolution(resolution, stores.memory_store)
            action = resolution["action"]

        results.append({"memory": raw["content"], "action": action})

    return ProcessingResult(processed=len(results), actions=results)
```

用中文总结: 统一架构将四个阶段串联：规则预过滤→LLM信号提取→实体解析→冲突检测与版本化写入。所有记忆共享统一数据模型，通过EntityRef链接实体图，通过版本字段支持生命周期管理。同时为每条记忆写入关键词索引（BM25用）、重要性评分和半衰期类型（检索时衰减用）。

### 3.5 Unified Read Pipeline (Pseudocode)

```python
import asyncio

async def handle_user_turn(
    user_id: str,
    message: str,
    stores: StorageLayer,
    budget: ContextBudget
) -> str:
    now = datetime.utcnow()

    # ── Step 1: Query analysis ──────────────────────────────────
    query_type = classify_query(message)          # fast heuristic, no LLM
    temporal_range = extract_temporal_ref(message) # None or (start, end)
    query_emb = await embed_model.encode_async(message)  # BGE-M3 / Qwen3

    # ── Step 2: Parallel hybrid retrieval ──────────────────────
    bm25_hits, dense_hits = await asyncio.gather(
        stores.bm25.search(message, user_id=user_id, top_k=25),
        stores.vectors.search(query_emb, user_id=user_id, top_k=25)
    )
    if temporal_range:
        bm25_hits = [m for m in bm25_hits if temporal_range[0] <= m.created_at <= temporal_range[1]]
        dense_hits = [m for m in dense_hits if temporal_range[0] <= m.created_at <= temporal_range[1]]

    # ── Step 3: RRF fusion + reranking ──────────────────────────
    candidates = rrf_merge(bm25_hits, dense_hits, k=60)[:50]
    reranked = await reranker.rerank_async(message, candidates, top_k=8)

    # ── Step 4: Final scoring (relevance + recency + importance) ─
    scored = []
    for mem, rerank_score in reranked:
        hours_elapsed = (now - mem.created_at).total_seconds() / 3600
        hl = mem.half_life_hours()  # type-specific; ∞ for biography
        recency = 1.0 if mem.pinned else (0.5 ** (hours_elapsed / hl))
        final = 0.40 * rerank_score + 0.35 * recency + 0.25 * (mem.importance / 10.0)
        scored.append((mem, final))
    top_memories = [m for m, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:5]]

    # ── Step 5: Dynamic budget allocation ────────────────────────
    alloc = allocate_budget(budget.total, query_type)

    # ── Step 6: Assemble context (anti-lost-in-middle placement) ─
    context = assemble_context(
        system_prompt=SYSTEM_PROMPT,                          # TOP
        user_profile=stores.profiles.get(user_id),           # TOP
        top_memory=top_memories[0] if top_memories else None, # TOP
        session_summaries=stores.sessions.get_recent(user_id, n=3),  # MIDDLE
        middle_memories=top_memories[1:4],                    # MIDDLE
        recent_history=stores.conv.get_recent(user_id, budget=alloc["recent_history"]),  # BOTTOM
        bottom_memory=top_memories[4] if len(top_memories) > 4 else None,  # BOTTOM
        current_message=message,                              # BOTTOM
    )

    # ── Step 7: LLM call ─────────────────────────────────────────
    response = await llm.generate_async(context)

    # ── Step 8: Async write-back (never blocks response) ─────────
    asyncio.create_task(
        write_back_pipeline(user_id, message, response, stores)
    )

    return response
```

---

## 4. Key Design Decisions

### Decision 1: Model Selection by Stage

| Stage | Model | Rationale |
|---|---|---|
| Pre-filter | Rule-based (no LLM) | Cost: $0; handles ~50% of filtering |
| Signal extraction | Claude Haiku / GPT-4o-mini | Fast, cheap, structured output |
| Conflict detection | Same small model | Single combined prompt with extraction |
| Entity disambiguation (ambiguous cases) | Claude Sonnet / GPT-4o | Only for <15% of cases; worth the cost |
| Embedding (retrieval) | BGE-M3 or Qwen3-Embedding | Strong Chinese + multilingual; top MTEB Chinese performance |
| Reranking | BGE-Reranker-v2-m3 or Cohere Rerank | Cross-encoder precision; BGE-v2-m3 has strong Chinese support |
| Session summarization | Claude Haiku / GPT-4o-mini | Triggered at session end; not latency-critical |
| Conversation | Claude Sonnet / Opus | Main conversation quality |

### Decision 2: Extraction Frequency

Extract after every **2–3 user turns**, not every turn. This:
- Gives the extractor more context
- Reduces latency impact on conversation
- Allows the extractor to see if the user contradicts themselves within a short window

### Decision 3: Merge vs. Split Default

**Always default to splitting (creating a new entity).** Merge only when:
- User explicitly says "X is the same as Y"
- Embedding cosine similarity > 0.92 AND shared attributes > 3

This is the most important design decision for trust — confusing two real people in a user's life is severely trust-damaging.

### Decision 4: Emotional State Handling

Do **not** persist emotional states long-term. Emotional states:
- Expire after 14 days automatically
- Are stored at session level for within-session continuity
- Are referenced in conversation context but not in long-term memory retrieval

This is both a technical decision (stale emotional states are misleading) and a safety/privacy decision.

### Decision 5: User Control Surface

Users should be able to:
- View their stored memories ("what do you remember about me?")
- Correct a memory ("actually, Zhang Wei is my colleague, not my friend")
- Delete a memory ("forget that I said that")
- Confirm an entity disambiguation ("yes, that's the same Zhang Wei")

User corrections feed directly into the lifecycle manager as highest-trust signals.

### Decision 6: Retrieval Strategy

Use **hybrid search (BM25 + dense, fused via RRF) + cross-encoder reranking** as the standard retrieval path. Do not use pure vector search — it achieves only ~49% recall and fails completely on freshness tasks.

| Retrieval Method | Recall | Freshness | Chinese Names | Latency | Decision |
|---|---|---|---|---|---|
| BM25 only | ~22% | Good | Good | Very fast | Insufficient recall |
| Dense only | ~49% | Poor (0.00 on freshness) | Fair | Medium | Never use alone |
| Hybrid (BM25+Dense) | ~53% | Good | Good | Medium | **Default** |
| Hybrid + reranking | ~53% | Good | Good | Medium+ | **Production standard** |
| Full-context dump | ~73% | Good | Good | 17s P95 | Too slow/expensive |

### Decision 7: Proactive vs. On-Demand Memory Injection

**Always inject memories proactively** (retrieve at the start of every turn). Do not wait for the LLM to ask for memories — it won't know what it doesn't have. Allow optional LLM-driven supplemental retrieval (via tool calls, max 2 rounds) only when retrieval confidence is low (relevance score < 0.6).

Memory writes are **always async** — they run after the response is sent and never add to user-perceived latency.

### Decision 8: Memory Compression Strategy

Use a **three-level hierarchy**: (1) verbatim recent turns in context, (2) LLM-generated session summaries triggered at session end, (3) long-term fact store retrieved via hybrid search. For users with < 50 sessions, skip session summarization and use verbatim history + retrieval. For > 500 sessions, consider EDU-based compression (EMem approach).

用中文总结: 八个关键决策：(1) 按阶段选模型（嵌入用BGE-M3，重排序用BGE-Reranker-v2-m3）；(2) 每2-3轮写入一次；(3) 默认拆分实体而非合并；(4) 情绪状态不长期持久化；(5) 暴露用户记忆管理界面；(6) 使用BM25+Dense混合检索+重排序，绝不单独使用向量检索；(7) 主动注入记忆（每轮自动检索），写入操作异步执行；(8) 三层记忆压缩（原始对话→会话摘要→长期事实库）。

---

## 5. Chinese Language Adaptations

The standard pipeline requires these modifications for Chinese:

| Problem | Standard Solution | Chinese Adaptation |
|---|---|---|
| Pronoun-less references | Pronoun detection | Discourse-level subject tracking; don't filter short utterances without 我 |
| NER | spaCy / standard NER | HanLP 2.x or LTP 4.x for segmentation + NER; add KINSHIP label |
| Entity matching | Name string match | Role-based match (妈妈 → MOTHER role) as primary signal |
| Name collisions | Name deduplication | Require attribute corroboration (workplace + city + role) to merge |
| Kinship terms | Generic synonym list | Chinese kinship role map (妈妈/老妈/母亲 → MOTHER; 外婆 vs 奶奶 distinct) |
| Coreference | Standard coref model | CDLM or LLM-based cross-session resolution; handle zero-pronoun |
| Embedding model | OpenAI ada-002 | BGE-M3 or Qwen3-Embedding; outperform ada-002 on colloquial/emotional Chinese text |
| BM25 tokenization | Whitespace tokenization | Must segment Chinese text (Jieba / HanLP) before BM25 indexing |
| Temporal expressions | English date parsing | Chinese temporal patterns: "上周" (last week), "前几天" (a few days ago), "去年" (last year) — require dedicated parser |

用中文总结: 中文适配要点（含检索层）：不过滤无主语短句（零代词）、使用HanLP/LTP做NER、以关系角色为主要实体匹配信号、嵌入模型使用BGE-M3或Qwen3-Embedding（对中文口语/情感文本优于OpenAI模型）、BM25索引前必须先分词（Jieba/HanLP）、时间表达式需专门处理中文时间词（"上周"/"前几天"等）。

---

## 6. Implementation Roadmap

### Phase 1 — Foundation (Weeks 1–6)
- [ ] Signal extraction pipeline with rule pre-filter + LLM extractor
- [ ] Basic memory store (PostgreSQL) with versioning schema + BM25 keyword field + importance score field
- [ ] 6-category taxonomy + salience scoring
- [ ] Simple keyword-based conflict detection (no LLM)
- [ ] Basic read path: sliding window context assembly (no retrieval yet)

### Phase 2 — Retrieval & Entity Graph (Weeks 7–12)
- [ ] Chinese NER integration (HanLP or LTP)
- [ ] Entity store with alias management + kinship role normalization
- [ ] BGE-M3 embedding pipeline for all stored memories
- [ ] BM25 index (Jieba segmentation + Elasticsearch or SQLite FTS5)
- [ ] Hybrid retrieval: RRF fusion of BM25 + dense → reranker (BGE-Reranker-v2-m3)
- [ ] Three-component memory scorer (relevance + recency + importance)
- [ ] Proactive retrieval integration: inject top-5 memories per turn
- [ ] Multi-signal entity scorer + confidence thresholds
- [ ] Clarification queue ("这是哪位张伟？")

### Phase 3 — Lifecycle Intelligence & Compression (Weeks 13–18)
- [ ] LLM-based conflict detection at write time
- [ ] Per-category TTL + daily expiry sweep
- [ ] Entity merge/split operations with audit trail
- [ ] User memory management API (view / correct / delete)
- [ ] Session summarization: LLM-generated summaries at session end
- [ ] Three-level memory hierarchy: verbatim turns → session summaries → long-term store
- [ ] Dynamic token budget allocator (per query-type)
- [ ] Temporal query detection + date-range pre-filter for retrieval

### Phase 4 — Quality Loop (Weeks 19–24)
- [ ] Human annotation pipeline for extraction quality evaluation
- [ ] A/B testing: memory-enabled vs. memory-disabled conversations
- [ ] Retrieval quality dashboard: Precision@5, Recall@5, staleness rate
- [ ] LongMemEval-style evaluation suite (adapted for Chinese companion domain)
- [ ] Correction rate tracking + retraining signal collection
- [ ] Embarrassing memory rate monitoring (detecting stale fact citations)

用中文总结: 四阶段实施路线图（24周，更新为含检索路径）：第一阶段搭建信号提取基础和基础上下文装配；第二阶段建立实体图和完整检索路径（BGE-M3嵌入+BM25+混合检索+重排序）；第三阶段加入生命周期管理和会话摘要压缩；第四阶段建立质量评估体系，含检索质量指标和中文伴侣场景专项评估。

---

## 7. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Entity conflation (confusing two people) | Medium | High | Default-split policy; require high confidence to merge |
| Emotional state overpersistence | Medium | High | Hard 14-day TTL; never surface stale emotional states in context |
| LLM extraction hallucination | Low-Medium | Medium | Store raw source alongside normalized content; human audit sampling |
| Memory store bloat | High (over time) | Low-Medium | TTL + salience-based archival; BM25/vector index maintenance |
| Chinese NER segmentation errors | Medium | Medium | Two-model ensemble; fall back to LLM-based extraction for Chinese NER |
| User privacy concerns | Medium | High | User-facing delete; no cross-user data; emotional states session-only |
| Clarification over-burden | Low-Medium | Medium | Cap clarification requests at <5% of turns; prefer implicit resolution |
| Stale memory citation ("embarrassing memory") | Medium | High | Type-specific recency decay; staleness rate monitoring dashboard |
| Retrieval miss on relevant facts | Medium | Medium | Hybrid BM25+dense retrieval; evaluate with Recall@5 on known-fact test set |
| "Lost in the middle" attention loss | Medium | Medium | Anti-lost-in-middle context ordering; keep per-memory snippets under 150 tokens |
| Retrieval latency spikes | Low-Medium | Medium | Pre-compute embeddings at write time; async retrieval with timeout fallback to recency-only |

用中文总结: 新增检索层风险：过时记忆引用（类型特定衰减+监控）、检索遗漏（混合检索+Recall@5评估）、"lost in the middle"注意力损失（上下文排序+短片段）、检索延迟峰值（写入时预计算嵌入+降级策略）。原有风险均保留。

---

## 8. Open Questions for the Product Team

1. **User transparency level**: How much should users know about what's stored? A "memory book" UI builds trust but increases product complexity.

2. **Cold start strategy**: For new users with no history, how does the companion behave? Consider explicitly asking memory-seeding questions early in the relationship.

3. **Multi-modal signals**: Do we have access to metadata beyond text (e.g., time of day, message response latency as a proxy for emotional engagement)? These can augment salience scoring.

4. **Cross-session emotional continuity**: Should the system remember "last time we talked you were stressed about your exam"? This requires short-lived emotional persistence (e.g., 48h window) rather than full TTL=0.

5. **Regulatory considerations**: China's PIPL (Personal Information Protection Law) requires explicit consent for storing personal information. Ensure the memory system's legal basis is established before launch.

6. **Retrieval latency budget**: Given a 2-second P95 response target, how should the budget be split between retrieval (target < 300ms), LLM generation (target < 1.5s), and write-back (async)? Consider caching embedding computations for returning users with stable profiles.

7. **Memory hierarchy depth**: When does session-level summarization become necessary? The research suggests < 50 sessions needs no summarization, but our companion may have users with hundreds of sessions within the first year. Plan for the hierarchy upfront even if not immediately activated.

用中文总结: 新增两个开放问题：(6) 在2秒响应预算内如何分配检索延迟（目标<300ms）和LLM生成延迟；(7) 何时启用会话摘要层——产品上线后用户会快速积累大量对话，应提前规划分层记忆架构，即使暂不激活。原有五个问题均保留。

---

## References

All findings synthesized from:

**Signal Extraction track:**
- Park et al. (2023). *Generative Agents*. ACM CHI.
- Zhong et al. (2024). *MemoryBank*. AAAI 2024.
- Mem0 (`mem0ai/mem0`). GitHub, 2024–2025.
- A-MEM (2025). *Agentic Memory for LLM Agents*. arXiv 2502.

**Lifecycle Management track:**
- Packer et al. (2023). *MemGPT*. NeurIPS 2023.
- Letta (2024). Technical documentation.
- Zep (2024). *Temporal Knowledge Graphs for Conversational AI*.
- LangChain Memory (2024). Conceptual guide.

**Entity Resolution track:**
- HanLP 2.x (`hankcs/HanLP`). GitHub.
- LTP 4.x (`HIT-SCIR/ltp`). Harbin Institute of Technology.
- Caciularu et al. (2021). *CDLM: Cross-Document Language Modeling*.
- Zeng et al. (2024). *Towards Long-Term Memory in Conversational AI*.
- NLPCC 2022–2024. Chinese Person Name Disambiguation shared tasks.
- Apple Intelligence (2024–2025). On-device knowledge graph design.

**Context Window Management & Retrieval track:**
- Liu et al. (2024). *Lost in the Middle: How Language Models Use Long Contexts*. TACL.
- Hsieh et al. (2024). *RULER: What's the Real Context Window Size of Your LLM?*
- Rasmussen et al. (2025). *Zep/Graphiti: Temporal Knowledge Graphs for LLM Memory*. arXiv 2501.13956.
- Xiaowu et al. (2025). *LongMemEval: Benchmarking Long-Term Memory in Conversational AI*. arXiv 2410.10813. ICLR 2025.
- Chen et al. (2025). *EMem: Efficient Memory Management for LLM Agents*. arXiv 2511.17208.
- arXiv 2604.08256 (2026). *HyperMem: Hierarchical Memory with Hypergraph Retrieval*.
- arXiv 2603.07670 (2026). *Memory for Autonomous LLM Agents: A Survey*.
- arXiv 2509.19376 (2025). *Temporal-Semantic Fusion for RAG Retrieval*.
- arXiv 2505.20231 (2025). *MemGuide: Intent-Driven Memory Selection*.
- arXiv 2308.15022 / Neurocomputing 2025. *Recursively Summarizing Enables Long-Term Dialogue Memory*.
- Chen et al. (2024). *BGE-M3: Multi-Lingual, Multi-Functionality Embedding Model*. BAAI.
- SNAP Research (2024). *LoCoMo: Long-Context Multi-Session Conversation Benchmark*. arXiv 2402.17753.
- Packer et al. (2023). *MemGPT: Towards LLMs as Operating Systems*. NeurIPS 2023. (also cited in Lifecycle track)
- Letta (2024). Open-source release and technical documentation. GitHub: `letta-ai/letta`.
