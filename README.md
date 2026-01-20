# 🔍 Agentic RAG Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Advanced Retrieval-Augmented Generation with specialized agents** for hybrid search, query classification, answer fusion, and self-correction. Implements SELF-RAG patterns for production-grade RAG systems.

---

## 🌟 Features

- **Hybrid Search** - Vector + BM25 + Metadata with RRF fusion
- **Query Classification** - Adaptive retrieval based on query type
- **Answer Fusion** - Multi-source synthesis with voting
- **Cross-Reference Validation** - Fact verification across sources
- **Source Citation** - APA, MLA, Chicago, IEEE formatting
- **Knowledge Gap Detection** - Iterative retrieval for missing info

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Agentic RAG Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐                                            │
│  │ Query            │  Classify: factual, analytical,           │
│  │ Classification   │  comparative, procedural                   │
│  └────────┬─────────┘                                            │
│           ↓                                                      │
│  ┌──────────────────┐                                            │
│  │ Hybrid Search    │  Vector + BM25 + Metadata                  │
│  │ (RRF Fusion)     │  Reciprocal Rank Fusion                    │
│  └────────┬─────────┘                                            │
│           ↓                                                      │
│  ┌──────────────────┐                                            │
│  │ Knowledge Gap    │  Detect missing info                       │
│  │ Detection        │  Trigger re-retrieval                      │
│  └────────┬─────────┘                                            │
│           ↓                                                      │
│  ┌──────────────────┐                                            │
│  │ Answer Fusion    │  Combine multiple sources                  │
│  │ (Voting/Hybrid)  │  Consistency analysis                      │
│  └────────┬─────────┘                                            │
│           ↓                                                      │
│  ┌──────────────────┐                                            │
│  │ Cross-Reference  │  Verify facts across sources               │
│  │ Validation       │                                            │
│  └────────┬─────────┘                                            │
│           ↓                                                      │
│  ┌──────────────────┐                                            │
│  │ Source Citation  │  APA, MLA, Chicago, IEEE                   │
│  └──────────────────┘                                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/yourusername/agentic-rag-framework.git
cd agentic-rag-framework
pip install -r requirements.txt
```

### Basic Usage

```python
from rag_engine.agents import HybridSearchAgent, QueryClassificationAgent

# Classify query for adaptive retrieval
classifier = QueryClassificationAgent(llm_client=my_llm)
classification = await classifier.execute(QueryClassificationRequest(
    query="Compare Python vs JavaScript for web development"
))
print(classification.query_type)  # "comparative"
print(classification.suggested_strategy)  # "multi_source_comparison"

# Hybrid search with RRF fusion
searcher = HybridSearchAgent(
    vector_store=my_vector_db,
    keyword_index=my_bm25_index
)
results = await searcher.execute(HybridSearchRequest(
    query="machine learning best practices",
    semantic_weight=0.5,
    keyword_weight=0.3,
    metadata_weight=0.2,
    fusion_strategy=FusionStrategy.RRF
))
```

---

## 📚 Agents

### HybridSearchAgent
Combines vector, keyword, and metadata search with score fusion.

```python
from rag_engine.agents import HybridSearchAgent, FusionStrategy

agent = HybridSearchAgent(vector_store=vs, keyword_index=ki)
result = await agent.execute(HybridSearchRequest(
    query="quantum computing applications",
    fusion_strategy=FusionStrategy.RRF,  # Reciprocal Rank Fusion
    use_reranking=True
))
```

### QueryClassificationAgent
Classifies queries by type, complexity, and intent.

```python
from rag_engine.agents import QueryClassificationAgent

agent = QueryClassificationAgent(llm_client=llm)
result = await agent.execute(QueryClassificationRequest(
    query="How do I implement a binary search tree?"
))
print(result.classification.query_type)    # PROCEDURAL
print(result.classification.complexity)    # MODERATE
print(result.classification.intent)        # LEARNING
```

### AnswerFusionAgent
Combines answers from multiple sources using ensemble techniques.

```python
from rag_engine.agents import AnswerFusionAgent, FusionStrategy

agent = AnswerFusionAgent(llm_client=llm)
result = await agent.execute(AnswerFusionRequest(
    answers=[answer1, answer2, answer3],
    query="What is the capital of France?",
    strategy=FusionStrategy.VOTING
))
print(result.fused_answer)
print(result.consistency_score)
```

### CrossReferenceValidationAgent
Validates facts across multiple sources.

```python
from rag_engine.agents import CrossReferenceValidationAgent

agent = CrossReferenceValidationAgent(llm_client=llm)
result = await agent.execute(CrossReferenceRequest(
    primary_content="Paris is the capital of France",
    reference_sources=[source1, source2, source3]
))
print(result.overall_reliability)
print(result.inconsistencies)
```

### SourceCitationAgent
Generates properly formatted citations.

```python
from rag_engine.agents import SourceCitationAgent, CitationStyle

agent = SourceCitationAgent()
result = await agent.execute(CitationRequest(
    sources=[source1, source2],
    style=CitationStyle.APA
))
print(result.bibliography)
```

---

## 📁 Project Structure

```
agentic-rag-framework/
├── rag_engine/
│   ├── __init__.py
│   └── agents/
│       ├── hybrid_search_agent.py
│       ├── query_classification_agent.py
│       ├── answer_fusion_agent.py
│       ├── cross_reference_validation_agent.py
│       └── source_citation_agent.py
├── examples/
├── tests/
├── requirements.txt
└── README.md
```

---

## 📄 License

MIT License - See [LICENSE](LICENSE)

---

## 📬 Contact

**Ravi Teja K** - AI/ML Engineer
- GitHub: [@TEJA4704](https://github.com/TEJA4704)
