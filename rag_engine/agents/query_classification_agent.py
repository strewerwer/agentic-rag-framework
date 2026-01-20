"""
Query Classification Agent

Classifies queries by type, intent, and complexity to enable
query-aware retrieval strategies and response generation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import logging

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries."""
    FACTUAL = "factual"  # Who, what, when, where
    ANALYTICAL = "analytical"  # Why, how, explain
    COMPARATIVE = "comparative"  # Compare, contrast, difference
    PROCEDURAL = "procedural"  # How-to, step-by-step
    OPINION = "opinion"  # Should, recommend, best
    AGGREGATION = "aggregation"  # Summarize, list all
    BOOLEAN = "boolean"  # Yes/no questions
    CONVERSATIONAL = "conversational"  # Chat, follow-up


class QueryComplexity(Enum):
    """Query complexity levels."""
    SIMPLE = "simple"  # Single fact, direct lookup
    MODERATE = "moderate"  # Multiple facts, some reasoning
    COMPLEX = "complex"  # Multi-step, deep reasoning
    MULTI_HOP = "multi_hop"  # Requires chaining multiple sources


class QueryIntent(Enum):
    """User intent behind query."""
    INFORMATION_SEEKING = "information_seeking"
    PROBLEM_SOLVING = "problem_solving"
    DECISION_MAKING = "decision_making"
    LEARNING = "learning"
    VERIFICATION = "verification"
    EXPLORATION = "exploration"


@dataclass
class QueryClassificationRequest:
    """Request for query classification."""
    
    query: str
    context: Optional[str] = None
    conversation_history: List[str] = field(default_factory=list)


@dataclass
class QueryClassification:
    """Classification result for a query."""
    
    query_type: QueryType
    complexity: QueryComplexity
    intent: QueryIntent
    requires_context: bool
    requires_multiple_sources: bool
    suggested_strategy: str
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)


@dataclass
class QueryClassificationResult:
    """Result from query classification."""
    
    classification: QueryClassification
    metadata: Dict = field(default_factory=dict)


class QueryClassificationAgent:
    """
    Universal Query Classification agent for RAG.
    
    Features:
    - Query type identification (factual, analytical, comparative, etc.)
    - Complexity assessment (simple, moderate, complex, multi-hop)
    - Intent detection (information seeking, problem solving, etc.)
    - Entity and keyword extraction
    - Retrieval strategy recommendation
    - Context requirement analysis
    
    Best for: Query routing, adaptive retrieval, personalized responses
    """

    def __init__(self, llm_client=None, mcp_client=None):
        """Initialize Query Classification agent."""
        self.llm_client = llm_client
        self.mcp_client = mcp_client
        logger.info("Query Classification Agent initialized")

    async def execute(self, request: QueryClassificationRequest) -> QueryClassificationResult:
        """
        Classify a query.
        
        Args:
            request: QueryClassificationRequest
            
        Returns:
            QueryClassificationResult with classification
        """
        logger.info("Classifying query", query=request.query)
        
        # Classify query type
        query_type, type_confidence = await self._classify_query_type(request.query)
        
        # Assess complexity
        complexity = await self._assess_complexity(request.query)
        
        # Detect intent
        intent = await self._detect_intent(request.query, request.context)
        
        # Extract entities and keywords
        entities = await self._extract_entities(request.query)
        keywords = await self._extract_keywords(request.query)
        
        # Determine requirements
        requires_context = self._requires_context(query_type, complexity)
        requires_multiple_sources = self._requires_multiple_sources(query_type, complexity)
        
        # Suggest retrieval strategy
        suggested_strategy = self._suggest_strategy(query_type, complexity, intent)
        
        classification = QueryClassification(
            query_type=query_type,
            complexity=complexity,
            intent=intent,
            requires_context=requires_context,
            requires_multiple_sources=requires_multiple_sources,
            suggested_strategy=suggested_strategy,
            confidence_scores={"query_type": type_confidence},
            entities=entities,
            keywords=keywords,
        )
        
        logger.info(
            "Query classified",
            type=query_type.value,
            complexity=complexity.value,
            intent=intent.value,
        )
        
        return QueryClassificationResult(
            classification=classification,
            metadata={"query": request.query},
        )

    async def _classify_query_type(self, query: str) -> tuple:
        """Classify the query type using LLM."""
        
        # Always use LLM for accurate classification
        if self.llm_client or self.mcp_client:
            return await self._llm_classify_query_type(query)
        
        # Fallback: simple rule-based classification
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["who", "what", "when", "where", "which"]):
            return QueryType.FACTUAL, 0.6
        elif any(word in query_lower for word in ["why", "how", "explain", "describe"]):
            return QueryType.ANALYTICAL, 0.6
        elif any(word in query_lower for word in ["compare", "contrast", "difference"]):
            return QueryType.COMPARATIVE, 0.6
        else:
            return QueryType.CONVERSATIONAL, 0.5

    async def _llm_classify_query_type(self, query: str) -> tuple:
        """Use LLM to classify query type."""
        
        prompt = f"""Classify this query into ONE of these types:
- factual (who, what, when, where)
- analytical (why, how, explain)
- comparative (compare, contrast)
- procedural (how-to, steps)
- opinion (should, recommend, best)
- aggregation (summarize, list all)
- boolean (yes/no)
- conversational (chat, follow-up)

Query: {query}

Type (one word only):"""

        try:
            if self.llm_client:
                response = await self.llm_client.generate(prompt=prompt, max_tokens=20)
                result = response.content.strip().lower()
            elif self.mcp_client:
                response = await self.mcp_client.call_tool("llm_generate", {"prompt": prompt})
                result = response.get("content", "conversational").strip().lower()
            
            # Map to enum
            type_map = {
                "factual": QueryType.FACTUAL,
                "analytical": QueryType.ANALYTICAL,
                "comparative": QueryType.COMPARATIVE,
                "procedural": QueryType.PROCEDURAL,
                "opinion": QueryType.OPINION,
                "aggregation": QueryType.AGGREGATION,
                "boolean": QueryType.BOOLEAN,
                "conversational": QueryType.CONVERSATIONAL,
            }
            
            return type_map.get(result, QueryType.CONVERSATIONAL), 0.75
        except:
            return QueryType.CONVERSATIONAL, 0.5

    async def _assess_complexity(self, query: str) -> QueryComplexity:
        """Assess query complexity using LLM."""
        
        # Use LLM for accurate complexity assessment
        if self.llm_client or self.mcp_client:
            prompt = f"""Assess the complexity of this query:

Query: {query}

Complexity levels:
- SIMPLE: Single fact, direct lookup, one-step reasoning
- MODERATE: Multiple facts, some reasoning required
- COMPLEX: Deep reasoning, multiple concepts, analysis needed
- MULTI_HOP: Requires chaining information from multiple sources

Respond with ONE word only (SIMPLE, MODERATE, COMPLEX, or MULTI_HOP):"""
            
            try:
                if self.llm_client:
                    response = await self.llm_client.generate(prompt=prompt, max_tokens=10, temperature=0.3)
                    complexity_str = response.content.strip().upper()
                elif self.mcp_client:
                    response = await self.mcp_client.call_tool(
                        "llm_generate", {"prompt": prompt, "max_tokens": 10, "temperature": 0.3}
                    )
                    complexity_str = response.get("content", "").strip().upper()
                
                # Map to enum
                if "MULTI" in complexity_str or "HOP" in complexity_str:
                    return QueryComplexity.MULTI_HOP
                elif "COMPLEX" in complexity_str:
                    return QueryComplexity.COMPLEX
                elif "SIMPLE" in complexity_str:
                    return QueryComplexity.SIMPLE
                else:
                    return QueryComplexity.MODERATE
            except:
                pass
        
        # Fallback: simple heuristics
        word_count = len(query.split())
        if word_count < 5:
            return QueryComplexity.SIMPLE
        elif word_count < 15:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.COMPLEX

    async def _detect_intent(self, query: str, context: Optional[str]) -> QueryIntent:
        """Detect user intent using LLM."""
        
        # Use LLM for accurate intent detection
        if self.llm_client or self.mcp_client:
            context_str = f"\nContext: {context}" if context else ""
            
            prompt = f"""Detect the user's primary intent behind this query:

Query: {query}{context_str}

Intent categories:
- INFORMATION_SEEKING: Looking for facts or information
- PROBLEM_SOLVING: Trying to solve a specific problem
- DECISION_MAKING: Needs help making a decision
- LEARNING: Wants to learn or understand a concept
- VERIFICATION: Checking or validating information
- EXPLORATION: Discovering or researching broadly

Respond with ONE intent only:"""
            
            try:
                if self.llm_client:
                    response = await self.llm_client.generate(prompt=prompt, max_tokens=20, temperature=0.3)
                    intent_str = response.content.strip().upper()
                elif self.mcp_client:
                    response = await self.mcp_client.call_tool(
                        "llm_generate", {"prompt": prompt, "max_tokens": 20, "temperature": 0.3}
                    )
                    intent_str = response.get("content", "").strip().upper()
                
                # Map to enum
                if "LEARNING" in intent_str:
                    return QueryIntent.LEARNING
                elif "VERIFICATION" in intent_str or "VALIDAT" in intent_str:
                    return QueryIntent.VERIFICATION
                elif "PROBLEM" in intent_str or "SOLVING" in intent_str:
                    return QueryIntent.PROBLEM_SOLVING
                elif "DECISION" in intent_str:
                    return QueryIntent.DECISION_MAKING
                elif "EXPLOR" in intent_str:
                    return QueryIntent.EXPLORATION
                else:
                    return QueryIntent.INFORMATION_SEEKING
            except:
                pass
        
        # Fallback: keyword matching
        query_lower = query.lower()
        if any(word in query_lower for word in ["learn", "understand", "explain"]):
            return QueryIntent.LEARNING
        elif any(word in query_lower for word in ["decide", "choose", "recommend"]):
            return QueryIntent.DECISION_MAKING
        else:
            return QueryIntent.INFORMATION_SEEKING

    async def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities using LLM."""
        
        # Use LLM for accurate entity extraction
        if self.llm_client or self.mcp_client:
            prompt = f"""Extract all named entities from this query:

Query: {query}

Extract entities like: people, places, organizations, products, dates, etc.
List them separated by commas:"""
            
            try:
                if self.llm_client:
                    response = await self.llm_client.generate(prompt=prompt, max_tokens=100, temperature=0.3)
                    entities_str = response.content.strip()
                elif self.mcp_client:
                    response = await self.mcp_client.call_tool(
                        "llm_generate", {"prompt": prompt, "max_tokens": 100, "temperature": 0.3}
                    )
                    entities_str = response.get("content", "").strip()
                
                # Parse entities
                entities = [e.strip() for e in entities_str.split(",") if e.strip()]
                return entities[:5]  # Top 5
            except:
                pass
        
        # Fallback: simple capitalization heuristic
        words = query.split()
        entities = [word for word in words if word and word[0].isupper() and len(word) > 1]
        return entities[:5]

    async def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords using LLM."""
        
        # Use LLM for semantic keyword extraction
        if self.llm_client or self.mcp_client:
            prompt = f"""Extract the 3-5 most important keywords from this query:

Query: {query}

Extract keywords that capture the core concepts and topics.
List them separated by commas:"""
            
            try:
                if self.llm_client:
                    response = await self.llm_client.generate(prompt=prompt, max_tokens=50, temperature=0.3)
                    keywords_str = response.content.strip()
                elif self.mcp_client:
                    response = await self.mcp_client.call_tool(
                        "llm_generate", {"prompt": prompt, "max_tokens": 50, "temperature": 0.3}
                    )
                    keywords_str = response.get("content", "").strip()
                
                # Parse keywords
                keywords = [k.strip().lower() for k in keywords_str.split(",") if k.strip()]
                return keywords[:5]
            except:
                pass
        
        # Fallback: simple stop word removal
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "what", "how", "why", "when", "where", "who"}
        words = query.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 3]
        return keywords[:5]

    def _requires_context(self, query_type: QueryType, complexity: QueryComplexity) -> bool:
        """Determine if query requires contextual information."""
        
        if complexity in [QueryComplexity.COMPLEX, QueryComplexity.MULTI_HOP]:
            return True
        
        if query_type in [QueryType.ANALYTICAL, QueryType.COMPARATIVE]:
            return True
        
        return False

    def _requires_multiple_sources(self, query_type: QueryType, complexity: QueryComplexity) -> bool:
        """Determine if query requires multiple sources."""
        
        if complexity == QueryComplexity.MULTI_HOP:
            return True
        
        if query_type in [QueryType.COMPARATIVE, QueryType.AGGREGATION]:
            return True
        
        return False

    def _suggest_strategy(
        self, query_type: QueryType, complexity: QueryComplexity, intent: QueryIntent
    ) -> str:
        """Suggest retrieval strategy based on classification."""
        
        strategies = {
            (QueryType.FACTUAL, QueryComplexity.SIMPLE): "dense_retrieval",
            (QueryType.FACTUAL, QueryComplexity.MODERATE): "hybrid_search",
            (QueryType.ANALYTICAL, QueryComplexity.COMPLEX): "multi_query_with_reranking",
            (QueryType.COMPARATIVE, QueryComplexity.MODERATE): "parallel_retrieval_with_fusion",
            (QueryType.PROCEDURAL, QueryComplexity.MODERATE): "sequential_retrieval",
            (QueryType.AGGREGATION, QueryComplexity.COMPLEX): "comprehensive_search_with_summarization",
        }
        
        # Try exact match
        key = (query_type, complexity)
        if key in strategies:
            return strategies[key]
        
        # Fallback based on complexity
        if complexity == QueryComplexity.SIMPLE:
            return "dense_retrieval"
        elif complexity == QueryComplexity.MODERATE:
            return "hybrid_search"
        elif complexity == QueryComplexity.MULTI_HOP:
            return "iterative_retrieval_with_decomposition"
        else:
            return "multi_query_with_reranking"
