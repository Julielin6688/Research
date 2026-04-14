# Entity Resolution in User Memory Systems for AI Companions

**Date:** 2026-04-13
**Scope:** Entity disambiguation, coreference resolution, and knowledge graph design for conversational AI memory — with special focus on Chinese-language challenges.

---

## Table of Contents

1. [Problem Framing](#1-problem-framing)
2. [Chinese-Specific Challenges](#2-chinese-specific-challenges)
3. [Entity Data Model / Schema](#3-entity-data-model--schema)
4. [Disambiguation Algorithms and Pseudocode](#4-disambiguation-algorithms-and-pseudocode)
5. [Relationship Graph Design](#5-relationship-graph-design)
6. [Lifecycle Management: Entity Splitting and Merging](#6-lifecycle-management-entity-splitting-and-merging)
7. [Open-Source Tooling](#7-open-source-tooling)
8. [How Real Products Handle This](#8-how-real-products-handle-this)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Recommended Architecture Summary](#10-recommended-architecture-summary)

---

## 1. Problem Framing

### The Core Problem

In a long-running AI companion relationship, users casually mention many people. Over hundreds of conversations, the system accumulates references like:

- "my mom," "妈妈," "my mother," "老妈" — all potentially the same person
- "Zhang Wei (张伟) from work" and "Zhang Wei (张伟), my college roommate" — definitely different people
- "my boyfriend" at month 1 and "my ex" at month 6 — same entity, changed relationship status

Without robust entity resolution, the memory system will:
- **Conflate** distinct people (false merge): treat two "Zhang Wei"s as one person, causing embarrassing or harmful confusions
- **Fragment** one person into many records (false split): store "妈妈" and "mom" and "my mother" as three different entities, losing accumulated context
- **Stale data**: fail to update "boyfriend" → "ex-boyfriend," presenting outdated facts to the user

### Terminology

| Term | Definition |
|---|---|
| **Named Entity Recognition (NER)** | Detecting spans in text that refer to people, places, orgs |
| **Coreference Resolution** | Determining which mentions across a conversation refer to the same entity |
| **Entity Linking** | Mapping a mention to a canonical entity record in a knowledge base |
| **Entity Disambiguation** | Choosing the correct entity when multiple candidates exist |
| **Entity Merging** | Combining two entity records that were incorrectly split |
| **Entity Splitting** | Separating one entity record that was incorrectly merged |

### Why This Is Hard in AI Companion Contexts

Unlike structured databases, conversations are:
- **Implicit**: "She called again" assumes the system already knows who "she" is
- **Context-dependent**: The same word means different things across sessions
- **Temporally evolving**: Relationships, names, and roles change
- **User-idiosyncratic**: Each user has a unique vocabulary for referring to the same people

用中文总结:
> 实体消歧的核心挑战在于：用户在对话中会用多种不同的方式提及同一个人（如"妈妈"、"老妈"、"我妈"），同时又可能用相同的称呼指代不同的人（如两个叫"张伟"的人）。AI 伴侣系统必须跨越多次对话保持正确的实体映射，并随着时间推移更新关系状态。

---

## 2. Chinese-Specific Challenges

### 2.1 Pro-Drop Language

Chinese is a **pro-drop** language: pronouns are frequently omitted when the referent is clear from context. This is far more extreme than in English.

- English: "I called my mom. **She** said she was fine."
- Chinese: "我打电话给妈妈了。说没事。" (No explicit "she" — subject dropped)

This means coreference resolution cannot rely on pronoun detection. The system must infer dropped subjects from discourse-level context, verb semantics, and conversational state.

### 2.2 No Word Boundaries

Chinese text has no spaces. "张伟去了" (Zhang Wei left) is a single string. Segmentation errors propagate into NER errors. The name boundary between a person name and surrounding context must be learned, not rule-based.

**Risk:** A segmenter might split "我妈妈今天" as ["我妈", "妈今天"] rather than ["我", "妈妈", "今天"], producing a phantom entity "妈" instead of recognizing "妈妈" as a single kinship term.

### 2.3 High Name Collision Rates

Chinese has a very small surname inventory. The top 3 surnames — 李 (Li), 王 (Wang), 张 (Zhang) — collectively cover approximately 22% of the Chinese population. Given a name like "张伟" (Zhang Wei), the probability that a user knows multiple people with this name is statistically significant.

Consequences:
- Cannot use the name string alone as an entity key
- Must rely on contextual attributes: workplace, relationship role, age, city, etc.
- The system must prompt for disambiguation when confidence is low

### 2.4 No Capitalization

English capitalizes proper nouns, providing a cheap signal for NER. Chinese has no such convention. The character sequence "王小红" looks identical in casing to any other three characters. NER models must rely entirely on character embeddings, context, and learned patterns.

### 2.5 Homonyms and Multiple Representations

The same person can be referred to via:
- Full name: 张伟
- Nickname: 小张, 阿伟, 大伟
- Relationship role: 我同事, 我室友
- Honorifics: 张总, 张老师
- Transliterations or mixed scripts: "Zhang Wei," "阿 Wei"

Any of these can refer to the same entity. The system must learn that user-specific nicknames map to canonical entity records.

### 2.6 Kinship Term Ambiguity

Chinese kinship vocabulary is extremely granular: 外婆 (maternal grandmother) vs. 奶奶 (paternal grandmother), 舅舅 (maternal uncle) vs. 叔叔 (paternal uncle). While this provides more signal when used precisely, users often use these terms loosely or inconsistently, and the system must track which kinship term a given user applies to which entity.

用中文总结:
> 中文对实体消歧带来独特挑战：一是零代词（主语经常省略），二是无词间空格导致分词错误，三是姓氏集中度高（"李王张"覆盖全国约22%人口）导致同名率极高，四是无大小写区别，五是同一个人可通过昵称、称谓、角色等多种方式被提及。系统不能单靠姓名字符串来唯一标识一个人。

---

## 3. Entity Data Model / Schema

### 3.1 Core Entity Record

```json
{
  "entity_id": "ent_uuid_v4",
  "canonical_name": "张伟",
  "entity_type": "PERSON",
  "created_at": "2025-03-01T10:23:00Z",
  "last_updated_at": "2026-02-14T18:45:00Z",

  "aliases": [
    { "surface": "小张", "source": "user_utterance", "confidence": 0.95 },
    { "surface": "张总", "source": "user_utterance", "confidence": 0.88 },
    { "surface": "Zhang Wei", "source": "user_utterance", "confidence": 0.99 }
  ],

  "attributes": {
    "gender": { "value": "male", "confidence": 0.97, "last_seen": "2025-08-10" },
    "workplace": { "value": "字节跳动", "confidence": 0.90, "last_seen": "2026-01-05" },
    "city": { "value": "北京", "confidence": 0.85, "last_seen": "2025-11-20" },
    "age_approx": { "value": 32, "confidence": 0.70, "last_seen": "2025-06-01" }
  },

  "relationship_to_user": {
    "current_role": "colleague",
    "role_history": [
      { "role": "colleague", "start": "2024-01-01", "end": null, "confidence": 0.95 }
    ]
  },

  "disambiguation_fingerprint": {
    "embedding_vector": "[float32 array, dim=768]",
    "key_facts": ["works at 字节跳动", "male", "Beijing", "user's team lead"]
  },

  "source_turns": ["turn_id_001", "turn_id_047", "turn_id_203"],

  "merge_history": [],
  "split_from": null,
  "confidence_score": 0.91,
  "status": "active"
}
```

### 3.2 Mention Record

```json
{
  "mention_id": "men_uuid",
  "turn_id": "turn_001",
  "surface_form": "小张",
  "span_start": 2,
  "span_end": 4,
  "mention_type": "NICKNAME",
  "resolved_entity_id": "ent_uuid_v4",
  "resolution_confidence": 0.88,
  "resolution_method": "alias_match + context_embedding",
  "ambiguous": false,
  "candidate_entities": ["ent_uuid_v4"]
}
```

### 3.3 Relationship Record

```json
{
  "relationship_id": "rel_uuid",
  "subject_entity_id": "user_self",
  "object_entity_id": "ent_uuid_v4",
  "relation_type": "COLLEAGUE",
  "relation_label_zh": "同事",
  "confidence": 0.95,
  "valid_from": "2024-01-01",
  "valid_until": null,
  "superseded_by": null
}
```

用中文总结:
> 实体数据模型包含三层结构：实体记录（包含别名列表、属性、关系历史和消歧指纹）、提及记录（将对话中的具体表述映射到实体）、以及关系记录（记录实体间关系及其时效性）。每个属性都带有置信度分数和最近观察时间戳，以支持生命周期管理。

---

## 4. Disambiguation Algorithms and Pseudocode

### 4.1 Overview of the Resolution Pipeline

The pipeline runs on each new conversational turn and has five stages:

1. **Extract mentions** from the new turn (NER + kinship/role detection)
2. **Generate candidates** from the entity store
3. **Score candidates** using a multi-signal ranker
4. **Decide**: resolve, create new, or defer (ask user)
5. **Update** the entity store

### 4.2 Full Pipeline Pseudocode

```python
def resolve_entities_in_turn(turn_text: str, user_id: str, conversation_context: list[str]) -> list[MentionRecord]:
    """
    Main entry point. Returns resolved mention records for all person
    references detected in a single conversational turn.
    """
    entity_store = load_entity_store(user_id)
    resolved_mentions = []

    # Stage 1: Extract candidate mentions
    raw_mentions = extract_mentions(turn_text, conversation_context)
    # raw_mentions: list of {surface, span, mention_type, context_window}

    for mention in raw_mentions:
        # Stage 2: Generate candidate entities from store
        candidates = retrieve_candidates(
            mention=mention,
            entity_store=entity_store,
            top_k=10
        )

        if not candidates:
            # No match found — create new entity
            new_entity = create_entity_from_mention(mention, turn_text, conversation_context)
            entity_store.add(new_entity)
            resolved_mentions.append(MentionRecord(mention, new_entity.id, confidence=1.0, method="new_entity"))
            continue

        # Stage 3: Score candidates
        scored = score_candidates(mention, candidates, conversation_context)
        top_candidate, top_score = scored[0]

        # Stage 4: Decide
        if top_score >= CONFIDENT_THRESHOLD:          # e.g., 0.85
            resolved_entity = top_candidate
            method = "confident_match"
        elif top_score >= AMBIGUOUS_THRESHOLD:        # e.g., 0.55
            # Low confidence: check if we can disambiguate from context
            resolved_entity = disambiguate_with_context(
                mention, scored, conversation_context, entity_store
            )
            if resolved_entity is None:
                # Defer: surface a clarifying question to user
                enqueue_clarification(user_id, mention, scored[:3])
                resolved_mentions.append(MentionRecord(mention, entity_id=None, confidence=top_score, method="deferred"))
                continue
            method = "context_resolved"
        else:
            # Below threshold — treat as new entity
            resolved_entity = create_entity_from_mention(mention, turn_text, conversation_context)
            entity_store.add(resolved_entity)
            method = "new_entity_low_score"

        # Stage 5: Update entity with new evidence
        update_entity(resolved_entity, mention, turn_text)
        resolved_mentions.append(MentionRecord(mention, resolved_entity.id, top_score, method))

    return resolved_mentions


def extract_mentions(text: str, context: list[str]) -> list[Mention]:
    """
    Uses Chinese NER + kinship/role pattern detection.
    For Chinese: runs HanLP or LTP for segmentation + NER,
    then applies role patterns (妈妈, 同事, 男友, etc.)
    """
    segments = chinese_segmenter.segment(text)
    ner_spans = ner_model.predict(segments)

    mentions = []
    for span in ner_spans:
        if span.label in ("PERSON", "ROLE", "KINSHIP"):
            mention_type = classify_mention_type(span.surface)
            context_window = get_surrounding_sentences(text, span, window=3)
            mentions.append(Mention(
                surface=span.surface,
                span=(span.start, span.end),
                mention_type=mention_type,   # PROPER_NAME | KINSHIP | ROLE | PRONOUN | NICKNAME
                context_window=context_window
            ))
    return mentions


def retrieve_candidates(mention: Mention, entity_store, top_k: int) -> list[Entity]:
    """
    Multi-strategy retrieval: alias match, embedding similarity, role match.
    """
    candidates = set()

    # Exact alias match
    alias_matches = entity_store.find_by_alias(mention.surface)
    candidates.update(alias_matches)

    # Embedding similarity (semantic match for role/kinship terms)
    mention_embedding = embed(mention.surface + " " + mention.context_window)
    embedding_matches = entity_store.similarity_search(mention_embedding, top_k=top_k)
    candidates.update(embedding_matches)

    # Role-based match (e.g., mention_type=KINSHIP "妈妈" → find entities with role=MOTHER)
    if mention.mention_type in ("KINSHIP", "ROLE"):
        role_matches = entity_store.find_by_relationship_role(
            normalize_role(mention.surface)
        )
        candidates.update(role_matches)

    return list(candidates)[:top_k]


def score_candidates(mention: Mention, candidates: list[Entity], context: list[str]) -> list[tuple[Entity, float]]:
    """
    Multi-signal scoring: weighted combination of signals.
    """
    scored = []
    for entity in candidates:
        score = 0.0

        # Signal 1: Alias / name string match
        alias_score = compute_alias_score(mention.surface, entity.aliases)
        score += 0.35 * alias_score

        # Signal 2: Embedding similarity to entity's disambiguation fingerprint
        mention_emb = embed(mention.surface + " " + mention.context_window)
        emb_score = cosine_similarity(mention_emb, entity.disambiguation_fingerprint.embedding_vector)
        score += 0.30 * emb_score

        # Signal 3: Relationship role consistency
        role_score = check_role_consistency(mention, entity, context)
        score += 0.20 * role_score

        # Signal 4: Recency boost — recently mentioned entities are more likely referents
        recency_score = compute_recency_score(entity.last_updated_at)
        score += 0.10 * recency_score

        # Signal 5: Attribute coherence with contextual clues in current turn
        attr_score = check_attribute_coherence(mention.context_window, entity.attributes)
        score += 0.05 * attr_score

        scored.append((entity, score))

    return sorted(scored, key=lambda x: x[1], reverse=True)


def disambiguate_with_context(mention, scored_candidates, context, entity_store):
    """
    Uses broader conversation context and LLM reasoning to break ties.
    Called when confidence is in the ambiguous range [0.55, 0.85).
    """
    top_2 = scored_candidates[:2]
    if top_2[0][1] - top_2[1][1] > 0.15:
        # Clear winner despite low absolute score
        return top_2[0][0]

    # Ask LLM to reason over context + candidate summaries
    prompt = build_disambiguation_prompt(mention, top_2, context)
    llm_choice = llm.classify(prompt)   # returns entity_id or "unknown"

    if llm_choice == "unknown":
        return None
    return entity_store.get(llm_choice)
```

### 4.3 Role Normalization for Chinese

```python
KINSHIP_ROLE_MAP = {
    "妈妈": "MOTHER", "老妈": "MOTHER", "妈": "MOTHER", "母亲": "MOTHER",
    "爸爸": "FATHER", "老爸": "FATHER", "爸": "FATHER", "父亲": "FATHER",
    "男友": "ROMANTIC_PARTNER", "男朋友": "ROMANTIC_PARTNER", "老公": "SPOUSE_M",
    "女友": "ROMANTIC_PARTNER", "女朋友": "ROMANTIC_PARTNER", "老婆": "SPOUSE_F",
    "前男友": "EX_ROMANTIC_PARTNER", "前女友": "EX_ROMANTIC_PARTNER",
    "闺蜜": "CLOSE_FRIEND_F", "好友": "CLOSE_FRIEND", "死党": "CLOSE_FRIEND",
    "同事": "COLLEAGUE", "领导": "SUPERIOR", "老板": "SUPERIOR",
    "室友": "ROOMMATE", "同学": "CLASSMATE"
}

def normalize_role(surface: str) -> str:
    return KINSHIP_ROLE_MAP.get(surface, "UNKNOWN_ROLE")
```

用中文总结:
> 消歧管道分五个阶段：提及提取→候选生成→多信号打分→置信度决策→实体更新。打分综合了别名匹配（35%）、语义向量相似度（30%）、关系角色一致性（20%）、近期提及加成（10%）和属性连贯性（5%）。置信度低时，优先用上下文推断；仍不确定时，向用户发起澄清提问。

---

## 5. Relationship Graph Design

### 5.1 Graph Schema

The entity graph is a **property graph** (suitable for Neo4j, Amazon Neptune, or a lightweight embedded store like SQLite with adjacency tables).

```
Nodes:
  (:Person {entity_id, canonical_name, status})
  (:User   {user_id})

Edges:
  (:User)-[:KNOWS {role, confidence, valid_from, valid_until}]->(:Person)
  (:Person)-[:KNOWS {role, context}]->(:Person)
  (:Person)-[:WORKS_AT]->(:Organization)
  (:Person)-[:LIVES_IN]->(:Location)
```

### 5.2 Temporal Edges

Relationships must be **bitemporal**: tracking both when the fact was true in the world, and when the system learned it.

```python
class TemporalRelationship:
    subject_id: str
    object_id: str
    relation_type: str          # "ROMANTIC_PARTNER", "COLLEAGUE", etc.
    valid_from: date            # when this relationship became true
    valid_until: date | None    # None = currently valid
    recorded_at: datetime       # when system learned this
    confidence: float
    superseded_by: str | None   # relationship_id of the successor record
```

### 5.3 Example: Handling "Boyfriend Becomes Ex"

```
Turn 50 (2025-01-10):  "我男友昨天说..."
  → Upsert edge: (User)-[:ROMANTIC_PARTNER {role:"boyfriend", valid_from:"2024-06-01", valid_until:null}]->(Entity_A)

Turn 247 (2025-08-20): "我前男友张伟总是这样..."
  → Detect role change: "前男友" ≠ "男友"
  → Close old edge: valid_until = "2025-08-20" (approximate)
  → Create new edge: (User)-[:EX_ROMANTIC_PARTNER {role:"ex-boyfriend", valid_from:"2025-08-20"}]->(Entity_A)
  → Flag for user confirmation if high-stakes
```

### 5.4 Graph Queries for Context Injection

```cypher
-- Get all current relationships of a user (for memory injection)
MATCH (u:User {user_id: $uid})-[r:KNOWS]->(p:Person)
WHERE r.valid_until IS NULL
RETURN p.canonical_name, r.role, r.confidence
ORDER BY r.confidence DESC

-- Find potential duplicate persons (same name, different records)
MATCH (p1:Person), (p2:Person)
WHERE p1.canonical_name = p2.canonical_name
  AND p1.entity_id <> p2.entity_id
RETURN p1, p2
```

用中文总结:
> 关系图采用属性图模型，支持双时态记录（事实在现实中的有效期 + 系统记录时间）。关系边在用户提及新的角色描述时自动更新，如"男友"变"前男友"时关闭旧边并新建一条记录，而非直接修改。这样可以保留关系历史，避免信息丢失。

---

## 6. Lifecycle Management: Entity Splitting and Merging

### 6.1 Entity Merging

Triggered when the system detects two entity records likely refer to the same person.

**Merge triggers:**
- User explicitly says "小张就是我说的那个张伟" (Little Zhang is the Zhang Wei I mentioned)
- Two entities share a high cosine similarity in their fingerprint embeddings (> 0.92)
- Overlapping attributes (same workplace + same approximate age + compatible relationship role)

```python
def merge_entities(entity_a_id: str, entity_b_id: str, reason: str):
    a = entity_store.get(entity_a_id)
    b = entity_store.get(entity_b_id)

    merged = Entity(
        entity_id=generate_new_id(),
        canonical_name=pick_canonical_name(a, b),   # prefer more complete / more frequent
        aliases=deduplicate(a.aliases + b.aliases),
        attributes=merge_attributes(a.attributes, b.attributes),  # take higher-confidence value
        relationship_to_user=merge_relationships(a, b),
        source_turns=a.source_turns + b.source_turns,
        merge_history=[{"merged_from": [a.id, b.id], "reason": reason, "timestamp": now()}]
    )

    entity_store.add(merged)
    entity_store.tombstone(a.id, successor=merged.entity_id)
    entity_store.tombstone(b.id, successor=merged.entity_id)
    relink_all_mentions(old_ids=[a.id, b.id], new_id=merged.entity_id)
```

### 6.2 Entity Splitting

Triggered when the system detects a single entity record conflates two distinct people.

**Split triggers:**
- Contradictory attributes appear: entity has both "works in Beijing" and "lives in Shanghai, never traveled" within the same time window
- User disambiguates: "哦不，我说的那个张伟是我大学同学，不是公司的那个"
- Embedding cluster analysis reveals two distinct sub-clusters in the entity's mention history

```python
def split_entity(entity_id: str, partition: dict):
    """
    partition: {"entity_a_turns": [...turn_ids], "entity_b_turns": [...turn_ids]}
    """
    original = entity_store.get(entity_id)

    entity_a = rebuild_entity_from_turns(partition["entity_a_turns"])
    entity_b = rebuild_entity_from_turns(partition["entity_b_turns"])

    entity_store.add(entity_a)
    entity_store.add(entity_b)
    entity_store.tombstone(entity_id, successor=None, reason="split",
                           split_into=[entity_a.id, entity_b.id])
```

用中文总结:
> 实体合并在发现两条记录指向同一个人时触发（用户明确声明、嵌入向量相似度过高、属性大量重叠），合并后原记录被标记为"已归档"并指向新记录。实体拆分在发现单条记录混入两个人的信息时触发（属性矛盾、用户主动纠正），系统从原始对话轮次重建两个独立实体。

---

## 7. Open-Source Tooling

### 7.1 Chinese NER and Segmentation

| Tool | Notes |
|---|---|
| **HanLP 2.x** | State-of-the-art Chinese NLP library; multi-task model covers NER, dependency parsing, coreference; supports both CPU and GPU. 2024 models fine-tuned on OntoNotes 5.0 Chinese. |
| **LTP (Language Technology Platform) 4.x** | Harbin Institute of Technology; modular pipeline with dedicated NER; fast inference; good support for named person entities. |
| **spaCy + zh_core_web_trf** | Transformer-based Chinese model; integrates easily into Python pipelines; NER covers PERSON, ORG, GPE. |
| **BERT-based Chinese NER** | Fine-tune `hfl/chinese-roberta-wwm-ext` (HuggingFace) on domain-specific NER data. For companion apps, add a custom label `KINSHIP` for 妈妈, 老公, etc. |

### 7.2 Coreference Resolution

- **CorefBERT / CorefRoBERTa (Chinese)**: Pre-trained models on Chinese CoNLL data. Handle within-document coreference. Less capable across session boundaries.
- **CDLM (Cross-Document Language Model)**: Extends coreference across documents (sessions), relevant for long-term memory.
- **LLM-as-resolver**: GPT-4 class models (including Claude) can be prompted to perform zero-shot coreference resolution across a few turns. Effective but costly. Best used as a fallback for low-confidence cases.

### 7.3 Vector Stores for Entity Fingerprints

- **Qdrant / Weaviate / Milvus**: Production-grade vector stores for embedding-based candidate retrieval
- **FAISS** (Meta): Local, no server required; good for smaller user bases

### 7.4 Graph Storage

- **Neo4j Community Edition**: Full property graph; Cypher query language; good for development
- **SQLite + adjacency tables**: Simpler, serverless, sufficient for single-user companion apps
- **Amazon Neptune** / **Azure Cosmos DB (Gremlin API)**: Managed graph databases for cloud deployment

用中文总结:
> 推荐技术栈：中文分词与NER使用HanLP 2.x或LTP 4.x，实体嵌入使用chinese-roberta-wwm-ext微调，候选检索使用Qdrant或FAISS，关系图存储使用Neo4j（生产）或SQLite（轻量级）。对于低置信度的消歧情况，可以回退到大语言模型的零样本推理。

---

## 8. How Real Products Handle This

### 8.1 Character.AI and Similar Companion Platforms

Based on public information and research papers through 2025, companion AI systems typically use a combination of:

- **Structured extraction**: At end of conversation, an LLM pass extracts structured facts ("user's mom is named 李秀英, lives in Chengdu") rather than storing raw text
- **Conservative merging**: New entities are created by default; merging requires explicit evidence to avoid false conflation
- **User-confirmable disambiguation**: When a new mention is ambiguous, the system makes a soft assumption and notes it conversationally ("你之前说的张伟是公司的那位对吧？")

### 8.2 MemGPT / Letta Architecture (2024–2025)

MemGPT (UC Berkeley, open-sourced as Letta) introduced a **paged memory** model where the LLM manages its own memory via tool calls. For entity management:
- Entities are stored in an archival store indexed by embedding
- The LLM decides when to write, update, or query entities
- Does not have a dedicated entity disambiguation module — relies on LLM judgment, which can be inconsistent across long sessions

### 8.3 WeChat / DingTalk

Enterprise messaging platforms avoid the disambiguation problem through **explicit contact graph**: contacts are pre-disambiguated by the OS phonebook or enterprise directory. AI features (like DingTalk's AI assistant) inherit this structured identity layer rather than solving NER from scratch.

**Lesson for companion apps**: Where possible, allow users to explicitly tag entities ("这是我妈" → link to a named entity). Reduce reliance on automatic disambiguation for high-stakes relationships.

### 8.4 Apple Intelligence / iOS Memory

Apple's on-device approach (2024–2025) uses a local knowledge graph built from Mail, Messages, Photos, and Calendar. Persons are disambiguated using the Contacts app as a ground truth anchor. Inferences about relationships are stored with confidence scores and surfaced only when high-confidence.

用中文总结:
> 主流产品的共同策略是：默认保守（宁可拆分不随意合并）、利用已有身份锚点（通讯录、企业目录）、结构化提取而非存储原始对话、以及在低置信度时向用户确认。伴侣 AI 产品可以借鉴这些策略，尤其是"允许用户主动标注实体"这一做法。

---

## 9. Evaluation Metrics

### 9.1 Entity Resolution Quality

| Metric | Formula | What It Measures |
|---|---|---|
| **MUC F1** | Precision/Recall on mention links | Coreference chain completeness |
| **B³ F1** | Entity-level precision/recall | Individual mention accuracy |
| **CEAFe** | Entity mapping quality | Bijective entity alignment |
| **CoNLL F1** | Mean of MUC + B³ + CEAFe | Standard coreference benchmark |
| **False Merge Rate** | Merges that are wrong / all merges | Cost of conflating two people |
| **False Split Rate** | Splits that are wrong / all splits | Cost of fragmenting one person |

### 9.2 Recommended Test Suite Design

```
Test class 1: Same-name disambiguation
  - Create 2 entities with identical names; verify system keeps them separate
  - Input: "张伟 (colleague) called" vs "Zhang Wei (roommate) texted"
  - Pass: two distinct entity records with no merge

Test class 2: Alias resolution
  - Single entity referred to via 5 different aliases across 10 turns
  - Pass: all mentions resolve to same entity_id

Test class 3: Relationship evolution
  - Turn 1: "my boyfriend"  →  Turn 50: "my ex"
  - Pass: entity unchanged; relationship edge updated; old edge closed

Test class 4: Pro-drop resolution
  - Chinese input with dropped subjects across turns
  - Pass: correct entity assigned to zero-pronoun mentions ≥80% accuracy

Test class 5: Low-confidence deferral
  - Ambiguous mention with no prior context
  - Pass: system enqueues clarification rather than guessing
```

### 9.3 User-Perceived Quality

Beyond automated metrics, track:
- **Correction rate**: How often users correct the system's entity assumptions
- **Clarification burden**: How often the system asks for disambiguation (should be < 5% of turns)
- **Stale fact surfacing rate**: How often the system presents outdated relationship facts

用中文总结:
> 评估指标分两层：自动化指标（CoNLL F1用于消歧链质量，错误合并率/错误拆分率用于实体边界质量）和用户感知指标（纠正率、澄清负担、过时信息曝出率）。测试集应覆盖同名消歧、别名解析、关系演变、零代词解析和低置信度回退五类场景。

---

## 10. Recommended Architecture Summary

```
┌─────────────────────────────────────────────────────────┐
│                   Conversation Turn                      │
└───────────────────────────┬─────────────────────────────┘
                            │
                     ┌──────▼──────┐
                     │  NER + Role │  ← HanLP / LTP / fine-tuned BERT
                     │  Detection  │    (segment → tag → span extraction)
                     └──────┬──────┘
                            │  raw mentions
                     ┌──────▼──────┐
                     │  Candidate  │  ← alias index + embedding ANN search
                     │  Retrieval  │    (Qdrant / FAISS)
                     └──────┬──────┘
                            │  top-k candidates
                     ┌──────▼──────┐
                     │  Multi-     │  ← alias score + embedding cosine +
                     │  Signal     │    role consistency + recency + attr
                     │  Scorer     │    coherence
                     └──────┬──────┘
                            │  scored candidates
              ┌─────────────▼──────────────┐
              │       Decision Engine       │
              │  high conf → resolve        │
              │  mid conf  → LLM reasoning  │
              │  low conf  → defer/ask user │
              └─────────────┬──────────────┘
                            │
               ┌────────────▼────────────┐
               │    Entity Store Update   │
               │  (create / update /      │
               │   merge / split)         │
               └────────────┬────────────┘
                            │
               ┌────────────▼────────────┐
               │   Property Graph (Neo4j) │
               │   Temporal Relationships │
               └─────────────────────────┘
```

### Key Design Principles

1. **Default to splitting, not merging.** Creating an extra entity is cheaper than conflating two people. Merge only with strong evidence.
2. **Confidence is first-class.** Every fact, every alias, every relationship carries a confidence score. Never treat any inference as ground truth.
3. **User corrections are gold labels.** When a user corrects the system, treat that as the highest-confidence signal and propagate it.
4. **Kinship roles are user-relative.** "妈妈" always means the *user's* mother, not any person named "Ma." Store kinship terms as relationship types, not names.
5. **Temporal edges, never overwrites.** Never overwrite a relationship; always close the old one and open a new one. This preserves history and enables rollback.
6. **Clarify proactively, not reactively.** A well-timed "你说的张伟是哪位？" is far better than silently making the wrong assumption for 200 turns.

用中文总结:
> 推荐架构将消歧流程分为NER提取→候选检索→多信号打分→决策→图更新五个环节。六条核心设计原则：默认拆分而非合并、置信度贯穿全链、用户纠正为最高优先级信号、亲属称谓按关系而非姓名存储、关系用时态边记录不覆盖、主动而非被动澄清。

---

## References and Further Reading

- **OntoNotes 5.0**: Standard NER + coreference benchmark dataset; Chinese portion covers news, web, and broadcast
- **HanLP GitHub**: `hankcs/HanLP` — multi-task Chinese NLP; 2024 models include NER, SRL, coref
- **LTP**: `HIT-SCIR/ltp` — Language Technology Platform from Harbin Institute of Technology
- **MemGPT / Letta**: `cpacker/MemGPT` — agentic memory management with tool-call based storage
- **CorefBERT**: "Coreferential Reasoning Learning for Language Representation" (Ye et al., 2020) — adapted for Chinese in subsequent work
- **CDLM**: "Cross-Document Language Modeling" (Caciularu et al., 2021) — cross-session coreference
- **Zeng et al. (2024)**: "Towards Long-Term Memory in Conversational AI" — surveys memory architectures in LLM-based systems, including entity-centric approaches
- **Chinese Name Disambiguation**: Multiple shared tasks at NLPCC (Chinese National Conference on NLP) 2022–2024 focus specifically on person name disambiguation in Chinese social media
- **Temporal Knowledge Graphs**: "TeMP: Temporal Message Passing for Temporal Knowledge Graph Completion" — relevant for modeling evolving relationships
