# Memory System Architecture: Final Synthesis
# 记忆系统架构：综合研究报告

> Synthesized from three parallel research tracks | April 2026
> For: AI Companion Product Team

---

## Executive Summary

We researched three core problems in user memory management for a Chinese-language AI companion product. This document synthesizes findings from all three tracks — signal extraction, lifecycle management, and entity resolution — into a single unified architecture recommendation.

**Key finding**: The three problems are deeply interdependent. An entity store without lifecycle management grows stale. A lifecycle manager without signal extraction operates on noise. Signal extraction without entity resolution creates fragmented, ambiguous records. The system must be designed as an integrated pipeline, not three independent modules.

用中文总结: 三个子问题（信号提取、生命周期管理、实体消歧）本质上是相互依赖的，必须作为一个整体管道来设计。本文档将三项研究的核心发现整合为统一的架构建议。

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

### 1.3 Entity Resolution (Keeping People Straight)

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

## 2. The Three Core Problems Are Interconnected

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

The three modules share a single write path. The extractor identifies what happened; the resolver identifies to whom; the lifecycle manager decides how to store it.

用中文总结: 三个模块共享同一条写入路径：信号提取器识别"发生了什么"，实体解析器识别"涉及谁"，生命周期管理器决定"如何存储"。三者必须串联，而非并行独立运行。

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

### 3.2 Unified Data Model

```typescript
// The central record type — connects all three subsystems
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

  // Versioning provenance
  embedding: float[];              // For semantic retrieval
}

interface EntityRef {
  canonical_id: string;            // Points to Entity record
  role_in_memory: string;          // "subject", "mentioned", "location"
  surface_form: string;            // How user referred to them
  resolution_confidence: number;
}
```

### 3.3 Unified Write Pipeline (Pseudocode)

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

用中文总结: 统一架构将四个阶段串联：规则预过滤→LLM信号提取→实体解析→冲突检测与版本化写入。所有记忆共享统一数据模型，通过EntityRef链接实体图，通过版本字段支持生命周期管理。

---

## 4. Key Design Decisions

### Decision 1: LLM Model Selection

| Stage | Model | Rationale |
|---|---|---|
| Pre-filter | Rule-based (no LLM) | Cost: $0; handles ~50% of filtering |
| Signal extraction | Claude Haiku / GPT-4o-mini | Fast, cheap, structured output |
| Conflict detection | Same small model | Single combined prompt with extraction |
| Entity disambiguation (ambiguous cases) | Claude Sonnet / GPT-4o | Only for <15% of cases; worth the cost |
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

用中文总结: 五个关键决策：(1) 小模型做提取/冲突检测，大模型做消歧兜底；(2) 每2-3轮提取一次；(3) 默认拆分实体而非合并；(4) 情绪状态不长期持久化（14天过期，隐私+准确性考量）；(5) 向用户暴露记忆管理界面，用户纠正为最高优先级输入。

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

用中文总结: 中文适配要点：不过滤无主语短句（零代词）、使用HanLP/LTP做分词和NER、以关系角色（MOTHER/COLLEAGUE等）为主要实体匹配信号、姓名相同不足以合并实体（需要额外属性佐证）、建立完整的中文亲属称谓规范化字典。

---

## 6. Implementation Roadmap

### Phase 1 — Foundation (Weeks 1–6)
- [ ] Signal extraction pipeline with rule pre-filter + LLM extractor
- [ ] Basic memory store (PostgreSQL) with versioning schema
- [ ] 6-category taxonomy + salience scoring
- [ ] Simple keyword-based conflict detection (no LLM)

### Phase 2 — Entity Graph (Weeks 7–12)
- [ ] Chinese NER integration (HanLP or LTP)
- [ ] Entity store with alias management + kinship role normalization
- [ ] Embedding-based candidate retrieval (FAISS for MVP, Qdrant for production)
- [ ] Multi-signal scorer + confidence thresholds
- [ ] Clarification queue ("这是哪位张伟？")

### Phase 3 — Lifecycle Intelligence (Weeks 13–18)
- [ ] LLM-based conflict detection at write time
- [ ] Per-category TTL + daily expiry sweep
- [ ] Entity merge/split operations with audit trail
- [ ] User memory management API (view / correct / delete)

### Phase 4 — Quality Loop (Weeks 19–24)
- [ ] Human annotation pipeline for extraction quality evaluation
- [ ] A/B testing: memory-enabled vs. memory-disabled conversations
- [ ] Staleness rate dashboard
- [ ] Correction rate tracking + retraining signal collection

用中文总结: 四阶段实施路线图（24周）：第一阶段搭建信号提取基础；第二阶段建立实体图和中文NER；第三阶段加入LLM冲突检测和生命周期管理；第四阶段建立质量评估和持续改进循环。

---

## 7. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Entity conflation (confusing two people) | Medium | High | Default-split policy; require high confidence to merge |
| Emotional state overpersistence | Medium | High | Hard 14-day TTL; never surface stale emotional states in context |
| LLM extraction hallucination | Low-Medium | Medium | Store raw source alongside normalized content; human audit sampling |
| Memory store bloat | High (over time) | Low-Medium | TTL + salience-based archival; index optimization |
| Chinese NER segmentation errors | Medium | Medium | Two-model ensemble; fall back to LLM-based extraction for Chinese NER |
| User privacy concerns | Medium | High | User-facing delete; no cross-user data; emotional states session-only |
| Clarification over-burden | Low-Medium | Medium | Cap clarification requests at <5% of turns; prefer implicit resolution |

用中文总结: 主要风险：实体混淆（严重度高，用默认拆分缓解）、情绪状态过期（强制14天TTL）、LLM提取幻觉（保留原始来源+人工抽样）、中文分词错误（双模型集成）、用户隐私（提供删除功能，情绪仅会话内）。

---

## 8. Open Questions for the Product Team

1. **User transparency level**: How much should users know about what's stored? A "memory book" UI builds trust but increases product complexity.

2. **Cold start strategy**: For new users with no history, how does the companion behave? Consider explicitly asking memory-seeding questions early in the relationship.

3. **Multi-modal signals**: Do we have access to metadata beyond text (e.g., time of day, message response latency as a proxy for emotional engagement)? These can augment salience scoring.

4. **Cross-session emotional continuity**: Should the system remember "last time we talked you were stressed about your exam"? This requires short-lived emotional persistence (e.g., 48h window) rather than full TTL=0.

5. **Regulatory considerations**: China's PIPL (Personal Information Protection Law) requires explicit consent for storing personal information. Ensure the memory system's legal basis is established before launch.

用中文总结: 需要产品团队决策的五个开放问题：用户透明度（记忆本UI）、冷启动策略、多模态信号扩展、跨会话情绪连续性、以及PIPL合规（个人信息保护法要求明确用户同意）。

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

