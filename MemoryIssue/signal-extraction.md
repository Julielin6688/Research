# Signal Extraction: What's Worth Remembering?
# 信号提取：什么值得被记忆？

> Research for AI Companion Memory System | April 2026

---

## 1. Problem Framing

In a conversational AI companion, users produce a continuous stream of utterances — most of which are ephemeral social exchanges ("haha", "ok", "what do you think?") but some carry durable, identity-relevant information ("I have a dog named Biscuit", "I'm terrified of flying", "my sister just got engaged"). The core challenge is distinguishing **signal** (memory-worthy facts) from **noise** (transient chatter) in real time, without interrupting the conversation flow.

The failure modes are symmetric:
- **Under-extraction**: The system forgets things the user clearly stated, creating a feeling of being unheard.
- **Over-extraction**: The system stores garbage ("the user said 'fine'") bloating the memory store and degrading retrieval quality.

用中文总结: 核心挑战是从对话流中区分"有价值的信号"（如用户偏好、事实、情感状态）和"短暂噪音"（如语气词、闲聊）。过度提取会污染记忆库，提取不足则让用户感到不被理解。

---

## 2. Taxonomy of Memory-Worthy Signals

Based on production systems (Mem0, MemGPT/Letta, Zep, Character.AI) and research (MemoryBank AAAI 2024, Generative Agents 2023, A-MEM 2025), memory-worthy signals fall into six categories:

| Category | Examples | Persistence |
|---|---|---|
| **Biographical facts** | name, age, location, occupation | Long-term |
| **Preferences & tastes** | hates cilantro, loves jazz, prefers dark mode | Long-term |
| **Relationships** | "my mom Zhang Wei", "my ex called Liu Yang" | Long-term, mutable |
| **Goals & plans** | "trying to quit smoking", "applying to grad school" | Medium-term |
| **Emotional states** | "been really anxious lately", "today was great" | Short-term |
| **Events & experiences** | "just got back from Tokyo", "lost my job last week" | Fades over time |

**Non-memory signals (discard):**
- Filler words and backchannel responses: "yeah", "ok", "lol", "hmm"
- Pure questions with no self-disclosure: "what's 2+2?"
- Meta-conversational acts: "say that again", "never mind"
- Hypotheticals clearly marked as such: "imagine if I were rich"

用中文总结: 值得记忆的信号分六类：传记事实、偏好、人际关系、目标/计划、情绪状态、事件经历。应丢弃填充词、纯疑问句、元对话行为和明显的假设性陈述。

---

## 3. Technical Approaches

### 3.1 LLM-Based Extraction (Dominant 2024–2026 Approach)

The current state of the art uses an LLM as a classifier + extractor in a single pass. **Mem0** (open-source, 2024–2025) exemplifies this pattern:

```python
EXTRACTION_PROMPT = """
You are a memory extraction system for an AI companion.
Given the conversation below, extract ONLY information that is:
1. A durable fact about the user (not the AI)
2. Explicitly stated (not inferred)
3. Likely to be relevant in future conversations

Return a JSON list. Each item has:
  - "content": the normalized fact (third-person, e.g. "User has a dog named Biscuit")
  - "category": one of [biographical, preference, relationship, goal, emotional_state, event]
  - "confidence": 0.0–1.0
  - "entities": list of named entities mentioned

If nothing is memory-worthy, return [].

Conversation:
{conversation_turns}
"""

def extract_signals(conversation: list[dict], llm) -> list[dict]:
    prompt = EXTRACTION_PROMPT.format(
        conversation_turns=format_turns(conversation)
    )
    raw = llm.complete(prompt)
    candidates = json.loads(raw)
    # Filter low-confidence candidates
    return [c for c in candidates if c["confidence"] >= 0.7]
```

**Key design choices in Mem0:**
- Extraction runs after each conversation turn (not in batch)
- Uses a cheaper/faster model for extraction, expensive model for conversation
- Deduplication happens downstream in the storage layer

### 3.2 Rule-Based Pre-filtering

Before hitting the LLM extractor, a rule-based gate can filter obvious non-signals cheaply:

```python
DISCARD_PATTERNS = [
    r"^(ok|okay|yeah|yep|nope|haha|lol|hmm|uh|um|right|sure|fine)[\.\!\?]?$",
    r"^\?+$",                          # pure question marks
    r"^(thanks?|thank you|ty)[\.\!]?$",
    r"^(lol+|haha+|hehe+|😂+)$",
]

MIN_CONTENT_TOKENS = 4  # utterances shorter than 4 tokens rarely carry signal

def pre_filter(utterance: str) -> bool:
    """Returns True if utterance should be passed to LLM extractor."""
    text = utterance.strip().lower()
    if token_count(text) < MIN_CONTENT_TOKENS:
        return False
    for pattern in DISCARD_PATTERNS:
        if re.fullmatch(pattern, text, re.IGNORECASE):
            return False
    return True
```

This reduces LLM calls by ~40–60% in typical companion conversations (based on Zep's published benchmarks, 2024).

### 3.3 Salience Scoring

Not all signals are equally important. A salience score helps prioritize retrieval and decide eviction order:

```python
def compute_salience(memory: dict) -> float:
    score = 0.0

    # Category weights (tunable)
    category_weight = {
        "biographical": 1.0,
        "preference": 0.85,
        "relationship": 0.95,
        "goal": 0.80,
        "emotional_state": 0.50,  # decays fast
        "event": 0.60,
    }
    score += category_weight.get(memory["category"], 0.5)

    # Recency boost (exponential decay, half-life = 30 days)
    days_old = (now() - memory["created_at"]).days
    score *= math.exp(-0.693 * days_old / 30)

    # Access frequency boost
    score += 0.1 * math.log1p(memory["access_count"])

    # Confidence from extraction
    score *= memory["confidence"]

    return score
```

### 3.4 Self-Disclosure Detection

Research from the HCI and NLP communities (Ravichander & Black 2018, updated by several 2024 companion AI papers) shows that **first-person statements with personal pronouns** are the strongest predictor of memory-worthy content:

```python
SELF_DISCLOSURE_INDICATORS = [
    r"\bI (am|was|have|had|love|hate|like|dislike|want|need|feel|think)\b",
    r"\bmy (mom|dad|sister|brother|friend|dog|cat|job|boss|partner|ex)\b",
    r"\bI'm (afraid|scared|excited|worried|happy|sad|tired|stressed)\b",
]

def has_self_disclosure(text: str) -> bool:
    for pattern in SELF_DISCLOSURE_INDICATORS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False
```

This heuristic achieves ~78% precision, ~65% recall as a standalone signal (per MemoryBank ablation studies).

用中文总结: 主流技术方案包括：(1) LLM直接提取（Mem0的做法，准确但有成本）；(2) 规则预过滤（正则表达式快速丢弃明显噪音，可减少40-60% LLM调用）；(3) 显著性评分（综合类别权重、时间衰减、访问频率打分）；(4) 自我披露检测（第一人称陈述是最强预测信号）。

---

## 4. Data Model

```typescript
interface ExtractedMemory {
  id: string;                    // UUID
  user_id: string;
  content: string;               // Normalized fact, third-person
  raw_source: string;            // Original user utterance
  source_turn_id: string;        // Which conversation turn
  category: MemoryCategory;
  entities: Entity[];            // Named entities (people, places, etc.)
  confidence: number;            // 0.0–1.0, from extractor
  salience: number;              // Computed score
  created_at: Date;
  last_accessed_at: Date;
  access_count: number;
  status: "active" | "superseded" | "expired";
  superseded_by?: string;        // ID of the memory that replaced this one
}

type MemoryCategory =
  | "biographical"
  | "preference"
  | "relationship"
  | "goal"
  | "emotional_state"
  | "event";

interface Entity {
  surface_form: string;          // As mentioned by user
  canonical_id: string;          // Links to entity store
  entity_type: "person" | "place" | "org" | "pet" | "concept";
  role?: string;                 // "mother", "friend", "colleague"
}
```

用中文总结: 数据模型的核心字段：标准化内容（第三人称）、原始来源、分类、实体列表、置信度、显著性评分、时间戳、访问计数、状态（active/superseded/expired）。实体单独抽象，通过canonical_id链接到实体存储。

---

## 5. Evaluation Metrics

| Metric | Definition | Target |
|---|---|---|
| **Extraction Precision** | % of extracted memories that are genuinely useful | ≥ 85% |
| **Extraction Recall** | % of user-stated facts that were captured | ≥ 75% |
| **Noise Rate** | % of stored memories that are ephemeral/useless | ≤ 10% |
| **Retrieval Hit Rate** | % of relevant memories surfaced when needed | ≥ 80% |
| **User Perceived Memory** | User survey: "does it remember things about me?" | ≥ 4/5 |

**Evaluation dataset approach**: Sample real conversation logs, have human annotators label each utterance as memory-worthy or not (using the taxonomy above as the rubric), then measure system performance against this gold standard.

用中文总结: 评估指标分两层：自动化指标（提取精确率/召回率、噪音率、检索命中率）和用户感知指标（用户调研问卷）。建议构建人工标注的黄金数据集进行评测。

---

## 6. Production System Observations

**Mem0 (open-source, 2024–2025):**
- Uses GPT-4o-mini for extraction, keeps costs low
- Maintains a vector store for semantic retrieval + a graph for relationship tracking
- Performs extraction + deduplication in a single LLM call using structured output

**MemGPT / Letta (2024–2025):**
- Memory is organized in explicit tiers: in-context (working memory), archival (long-term)
- Agent decides when to write to archival memory using tool calls
- More agentic but higher latency and cost

**Zep (2024):**
- Focuses on session-level memory, with a separate "fact extraction" pipeline
- Uses smaller fine-tuned models for extraction speed
- Strong emphasis on structured entity extraction

**Character.AI / Replika (product-level):**
- Details are proprietary, but published UX research shows they store persona traits and relationship milestones
- Emotional state is tracked at session level, not persisted long-term (privacy/safety consideration)

用中文总结: 主流产品对比：Mem0用GPT-4o-mini做提取，成本低，结合向量库+图谱；MemGPT/Letta采用分层记忆架构，更智能但延迟高；Zep专注会话级记忆，用微调小模型提速；Character.AI/Replika出于隐私考虑，情绪状态仅在会话内保留，不长期持久化。

---

## 7. Recommendations for the Companion Product

1. **Use a two-stage pipeline**: cheap rule-based pre-filter → LLM extractor (e.g., Claude Haiku or GPT-4o-mini). This balances cost and quality.
2. **Extract after every 2–3 turns**, not every turn — batching reduces latency and gives the LLM more context.
3. **Use the six-category taxonomy** above; tune salience weights based on your specific product (a romantic companion weights `relationship` higher; a journaling companion weights `emotional_state` higher).
4. **Store raw source alongside normalized content** — enables re-extraction if your taxonomy evolves.
5. **Track emotional state at the session level only** initially — persisting emotional states long-term raises privacy and safety concerns.
6. **Human eval loop**: weekly sampling of extracted memories for quality review, especially in early stages.

用中文总结: 产品建议：(1) 两阶段管道（规则预过滤+LLM提取）；(2) 每2-3轮提取一次而非每轮；(3) 使用六类分类体系，根据产品类型调整权重；(4) 始终保存原始来源；(5) 情绪状态仅在会话内保留；(6) 建立人工评估循环。

---

## References

- Park et al. (2023). *Generative Agents: Interactive Simulacra of Human Behavior*. ACM CHI 2023.
- Zhong et al. (2024). *MemoryBank: Enhancing Large Language Models with Long-Term Memory*. AAAI 2024.
- Mem0 GitHub (2024–2025). `mem0ai/mem0`. Apache 2.0.
- Zep (2024). *Zep: Long-Term Memory for AI Assistants*. Technical blog.
- Letta/MemGPT (2024). *MemGPT: Towards LLMs as Operating Systems*. NeurIPS 2023 + 2024 updates.
- A-MEM (2025). *Agentic Memory for LLM Agents*. arXiv 2502.
- Ravichander & Black (2018). *An Empirical Study of Self-Disclosure in Spoken Dialogue Systems*.
