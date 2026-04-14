# Memory Lifecycle Management: Updates, Conflicts, and Versioning
# 记忆生命周期管理：更新、冲突与版本控制

> Research for AI Companion Memory System | April 2026

---

## 1. Problem Framing

A memory system is not a write-once store. Users change: they move cities, switch jobs, start and end relationships, shift opinions, heal from illness. The core lifecycle challenge is ensuring that as the user's reality evolves, the memory store evolves with it — without either (a) clinging to stale facts or (b) discarding history prematurely.

The three sub-problems:

1. **Update detection**: New utterance X contradicts stored memory M. How do we recognize this?
2. **Conflict resolution**: When X and M conflict, which wins? Always X? Or does context matter?
3. **Versioning**: Should we delete M, mutate it, or keep it alongside X with a temporal marker?

The "staleness problem" is the most user-visible failure: the user mentioned something new, but the AI keeps referencing the old information. This destroys the feeling of being truly known.

用中文总结: 记忆管理的核心挑战是让存储的信息随用户现实变化而演进。三个子问题：(1) 更新检测——如何识别新信息与旧记忆的矛盾；(2) 冲突解决——谁赢？；(3) 版本控制——删除、覆盖还是保留历史？"过时问题"是最影响用户感知的失败模式。

---

## 2. Memory Lifecycle States

```
                    ┌─────────────┐
                    │  CANDIDATE  │  ← extracted from conversation
                    └──────┬──────┘
                           │ passes dedup + confidence check
                           ▼
                    ┌─────────────┐
                    │   ACTIVE    │  ← in use, retrieved in context
                    └──────┬──────┘
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │SUPERSEDED│ │ ARCHIVED │ │ EXPIRED  │
        └──────────┘ └──────────┘ └──────────┘
        (newer fact   (low salience, (TTL elapsed,
         replaced it)  rarely used)   category-based)
```

State transitions are explicit and logged — we never hard-delete, only soft-delete, to preserve auditability and enable rollback.

用中文总结: 记忆有五个状态：候选（刚提取）、活跃（正在使用）、已取代（被新事实替换）、已归档（显著性低）、已过期（TTL到期）。所有转换均软删除、可审计、可回滚。

---

## 3. Update Detection

### 3.1 Semantic Contradiction Detection

The core approach: when new memories are extracted, run a contradiction check against existing active memories with overlapping entities and categories.

```python
CONTRADICTION_CHECK_PROMPT = """
You are checking if two memory statements conflict with each other.

Memory A (existing): "{memory_a}"
Memory B (new): "{memory_b}"

Respond with JSON:
{{
  "relationship": "contradicts" | "updates" | "refines" | "unrelated",
  "confidence": 0.0–1.0,
  "explanation": "one sentence"
}}

- "contradicts": they cannot both be true at the same time
- "updates": B supersedes A (A was true, now B is true)
- "refines": B adds detail to A without invalidating it
- "unrelated": different topics entirely
"""

def detect_update(new_memory: dict, candidate_existing: list[dict], llm) -> list[dict]:
    """
    Returns list of existing memories that conflict with or are updated by new_memory.
    Only checks memories with overlapping entities + same category.
    """
    conflicts = []
    for existing in candidate_existing:
        result = llm.complete(CONTRADICTION_CHECK_PROMPT.format(
            memory_a=existing["content"],
            memory_b=new_memory["content"]
        ))
        parsed = json.loads(result)
        if parsed["relationship"] in ("contradicts", "updates") and parsed["confidence"] > 0.75:
            conflicts.append({
                "existing_memory": existing,
                "relationship": parsed["relationship"],
                "confidence": parsed["confidence"]
            })
    return conflicts
```

### 3.2 Candidate Filtering (Pre-LLM)

Before the expensive LLM check, filter to likely candidates using:
1. **Same category** (don't compare `relationship` memories against `preference`)
2. **Shared named entities** (same person, place, or topic)
3. **Embedding similarity** > 0.6 (cosine distance in vector store)

```python
def get_conflict_candidates(new_memory: dict, memory_store) -> list[dict]:
    # Vector similarity search — fast, approximate
    similar = memory_store.search(
        embedding=new_memory["embedding"],
        top_k=10,
        filter={"category": new_memory["category"], "status": "active"}
    )
    # Filter to those sharing at least one entity
    new_entity_ids = {e["canonical_id"] for e in new_memory["entities"]}
    return [
        m for m in similar
        if {e["canonical_id"] for e in m["entities"]} & new_entity_ids
    ]
```

用中文总结: 更新检测分两步：先用向量相似度+实体过滤快速筛选候选记忆，再用LLM判断具体关系类型（矛盾/更新/细化/无关）。只对同类别、共享实体的记忆对进行LLM检查，控制成本。

---

## 4. Conflict Resolution

When a conflict is detected, the resolution strategy depends on the relationship type:

```python
def resolve_conflict(
    new_memory: dict,
    existing_memory: dict,
    relationship: str,
    context: dict
) -> dict:
    """
    Returns an action dict describing what to do.
    """
    if relationship == "updates":
        # New info supersedes old — standard case
        # Example: "I moved to Beijing" supersedes "User lives in Shanghai"
        return {
            "action": "supersede",
            "supersede_id": existing_memory["id"],
            "activate_id": new_memory["id"],
            "note": f"Superseded by newer information on {now().date()}"
        }

    elif relationship == "contradicts":
        # Genuine contradiction — apply trust hierarchy
        trust_score_new = compute_trust(new_memory, context)
        trust_score_old = compute_trust(existing_memory, context)

        if trust_score_new > trust_score_old + 0.2:
            # New is clearly more trustworthy
            return {"action": "supersede", "supersede_id": existing_memory["id"], ...}
        else:
            # Ambiguous — flag both as uncertain, don't discard either
            return {
                "action": "flag_conflict",
                "memory_ids": [existing_memory["id"], new_memory["id"]],
                "note": "Conflicting information; user clarification may help"
            }

    elif relationship == "refines":
        # Merge additional detail into existing
        return {
            "action": "merge",
            "base_id": existing_memory["id"],
            "append_detail": new_memory["content"]
        }


def compute_trust(memory: dict, context: dict) -> float:
    score = 0.0
    score += memory["confidence"] * 0.4          # extraction confidence
    score += recency_score(memory["created_at"]) * 0.4  # newer = more trusted
    score += 0.2 if context.get("explicit_correction") else 0.0
    # "Actually, I moved" or "wait, I meant" = explicit correction = high trust
    return score
```

**Trust hierarchy for conflict resolution:**
1. Explicit corrections ("actually, I meant...") — highest trust
2. More recent statements
3. Higher extraction confidence
4. Statements with more contextual detail

用中文总结: 冲突解决策略基于关系类型和信任层级：(1) 更新关系→新取代旧；(2) 矛盾关系→计算信任分，分差>0.2则取代，否则标记冲突待澄清；(3) 细化关系→合并补充细节。信任层级：用户主动纠正>时间更新>提取置信度>细节丰富度。

---

## 5. Versioning Strategy

We use an **append-only versioned log** — memories are never mutated or hard-deleted, only transitioned between states with timestamps. This mirrors event-sourcing patterns.

```typescript
interface VersionedMemory {
  id: string;                      // Immutable UUID
  user_id: string;
  content: string;
  category: MemoryCategory;
  entities: Entity[];
  confidence: number;
  status: "active" | "superseded" | "archived" | "expired" | "flagged";

  // Versioning fields
  version: number;                 // 1, 2, 3... incremented on supersede
  parent_id?: string;              // ID of memory this supersedes
  superseded_by?: string;          // ID of memory that superseded this
  superseded_at?: Date;

  // Lifecycle timestamps
  created_at: Date;
  status_changed_at: Date;
  expires_at?: Date;               // For TTL-based expiry

  // Provenance
  source_conversation_id: string;
  source_turn_id: string;
}
```

**Retrieval rule**: only `status = "active"` memories are returned by default. Superseded versions are accessible for debugging, auditing, or "what did the user say before?" queries.

用中文总结: 版本控制采用仅追加日志（append-only），记忆永不硬删除，只做状态转换并记录时间戳。每条记忆有version字段、parent_id（取代了哪条）和superseded_by（被哪条取代）。检索时默认只返回active状态的记忆。

---

## 6. Decay and Expiration Policies

Different memory categories should have different TTLs and decay behaviors:

```python
TTL_POLICIES = {
    "biographical": None,          # Never expires automatically
    "preference": None,            # Never expires; but salience decays
    "relationship": None,          # Never expires; but may be updated
    "goal": timedelta(days=180),   # Goals become stale after ~6 months without reinforcement
    "emotional_state": timedelta(days=14),  # Emotional states are short-lived
    "event": timedelta(days=365),  # Events fade to archival after a year
}

DECAY_HALF_LIVES = {
    "biographical": float("inf"),
    "preference": 180,             # days
    "relationship": float("inf"),
    "goal": 60,
    "emotional_state": 7,
    "event": 90,
}

def get_decayed_salience(memory: dict) -> float:
    base_salience = memory["salience"]
    half_life = DECAY_HALF_LIVES[memory["category"]]
    if half_life == float("inf"):
        return base_salience
    days_old = (now() - memory["created_at"]).days
    return base_salience * (0.5 ** (days_old / half_life))

def run_expiry_sweep(memory_store, user_id: str):
    """Run periodically (e.g., daily) to expire/archive stale memories."""
    active = memory_store.get_active(user_id)
    for memory in active:
        policy = TTL_POLICIES[memory["category"]]
        if policy and (now() - memory["created_at"]) > policy:
            memory_store.transition(memory["id"], "expired")
        elif get_decayed_salience(memory) < ARCHIVE_THRESHOLD:
            memory_store.transition(memory["id"], "archived")
```

用中文总结: 不同类别采用不同TTL和衰减策略：传记事实永不过期；情绪状态14天后过期；目标180天后过期；事件1年后过期。显著性采用指数衰减，半衰期按类别设定。建议每日运行过期清扫任务。

---

## 7. Production System Observations

**Mem0 (2024–2025):**
- Uses an LLM to decide in a single pass: "add", "update", "delete", or "no change"
- Maintains a `metadata.update_history` field tracking what changed and when
- Deduplication uses embedding similarity + an LLM "is this the same fact?" check
- No explicit versioning — old memories are overwritten (simpler but loses history)

**Letta / MemGPT (2024–2025):**
- The agent itself decides when to write/edit/delete archival memories using tool calls
- Edit operations are explicit: `core_memory_replace(old_content, new_content)`
- This is powerful but means update quality depends on the agent's reasoning
- History is preserved in the agent's conversation log even if memory is overwritten

**Zep (2024):**
- Maintains a "fact version graph" — each fact has a linked list of versions
- Uses temporal metadata: `valid_from`, `valid_until` per fact
- Conflict resolution is primarily recency-based (most recent wins)
- Provides an explicit "memory diff" API for debugging

**LangChain Memory (2024):**
- Multiple backends: `ConversationBufferMemory`, `ConversationSummaryMemory`, `EntityMemory`
- No built-in conflict resolution — application layer responsibility
- Entity memory tracks per-entity summaries that are updated by re-summarization

用中文总结: 主流系统对比：Mem0单次LLM调用决定增删改，无版本历史；Letta/MemGPT由Agent自主决定记忆操作，灵活但依赖Agent推理质量；Zep维护"事实版本图"，有valid_from/valid_until时间元数据；LangChain Memory冲突解决依赖应用层，不内置。

---

## 8. Unified Update Pipeline

```python
def process_new_memories(
    new_memories: list[dict],
    user_id: str,
    memory_store,
    llm
) -> dict:
    """
    Full lifecycle pipeline for newly extracted memories.
    Returns summary of actions taken.
    """
    actions = []

    for new_mem in new_memories:
        # 1. Embed the new memory
        new_mem["embedding"] = embed(new_mem["content"])

        # 2. Find conflict candidates
        candidates = get_conflict_candidates(new_mem, memory_store)

        if not candidates:
            # No conflicts — simple insert
            memory_store.insert(new_mem, status="active")
            actions.append({"type": "insert", "memory": new_mem["content"]})
            continue

        # 3. Detect conflicts via LLM
        conflicts = detect_update(new_mem, candidates, llm)

        if not conflicts:
            # Similar but not conflicting — check for near-duplicate
            if is_near_duplicate(new_mem, candidates[0]):
                actions.append({"type": "skip_duplicate", "memory": new_mem["content"]})
                continue
            memory_store.insert(new_mem, status="active")
            actions.append({"type": "insert", "memory": new_mem["content"]})
        else:
            # 4. Resolve each conflict
            for conflict in conflicts:
                resolution = resolve_conflict(
                    new_mem,
                    conflict["existing_memory"],
                    conflict["relationship"],
                    context={}
                )
                apply_resolution(resolution, new_mem, memory_store)
                actions.append(resolution)

    return {"user_id": user_id, "actions": actions, "processed": len(new_memories)}
```

用中文总结: 统一更新管道五步：(1) 嵌入新记忆；(2) 向量搜索候选冲突；(3) LLM检测冲突类型；(4) 无冲突则插入（检查近重复）；(5) 有冲突则按解决策略应用（取代/标记/合并）。

---

## 9. Recommendations

1. **Use append-only versioning from day one** — retrofitting is painful; losing history is permanent.
2. **Implement per-category TTL and decay** — emotional states should not persist like biographical facts.
3. **Treat explicit user corrections as the highest trust signal** — detect phrases like "actually", "I meant", "wait, no" and weight them strongly.
4. **Run a daily expiry sweep** — don't let the memory store grow unbounded.
5. **Expose a "memory diff" tool** — invaluable for debugging staleness reports from users.
6. **Never hard-delete** — soft-delete with status transitions, keeping history for at least 90 days before permanent removal.
7. **Monitor staleness rate** — metric: "% of retrieved memories that were superseded in the last 30 days". Target < 5%.

用中文总结: 核心建议：从第一天就用仅追加版本控制；按类别差异化设置TTL；将用户主动纠正视为最高信任信号；每日运行过期清扫；提供"记忆差分"调试工具；永远软删除；监控过时率（目标<5%）。

---

## References

- Packer et al. (2023). *MemGPT: Towards LLMs as Operating Systems*. NeurIPS 2023.
- Letta (2024). *Stateful Agents with Core and Archival Memory*. Technical docs.
- Mem0 (2024–2025). `mem0ai/mem0`. GitHub. Apache 2.0.
- Zep (2024). *Temporal Knowledge Graphs for Conversational AI*. Technical blog.
- Zhong et al. (2024). *MemoryBank: Enhancing LLMs with Long-Term Memory*. AAAI 2024.
- LangChain (2024). *Memory: Conceptual guide*. LangChain docs.
- Berners-Lee et al. (classic). *Linked Data* — temporal RDF patterns applied to memory versioning.
