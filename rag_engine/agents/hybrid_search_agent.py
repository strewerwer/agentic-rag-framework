"""
Hybrid Search Agent

Combines semantic (vector), keyword (BM25), and metadata filtering
for comprehensive retrieval with score fusion.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

import logging

logger = logging.getLogger(__name__)


class FusionStrategy(Enum):
    """Score fusion strategies."""
    RRF = "reciprocal_rank_fusion"  # Reciprocal Rank Fusion
    WEIGHTED = "weighted_sum"
    MAX = "max_score"
    MIN = "min_score"


@dataclass
class SearchResult:
    """A search result with scores."""
    
    content: str
    doc_id: str
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    metadata_score: float = 0.0
    final_score: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class HybridSearchRequest:
    """Request for hybrid search."""
    
    query: str
    top_k: int = 10
    semantic_weight: float = 0.5
    keyword_weight: float = 0.3
    metadata_weight: float = 0.2
    fusion_strategy: FusionStrategy = FusionStrategy.RRF
    metadata_filters: Dict[str, Any] = field(default_factory=dict)
    use_reranking: bool = True


@dataclass
class HybridSearchResult:
    """Result from hybrid search."""
    
    results: List[SearchResult]
    total_results: int
    fusion_strategy: str
    metadata: Dict = field(default_factory=dict)


class HybridSearchAgent:
    """
    Universal Hybrid Search agent for RAG.
    
    Features:
    - Semantic search (vector similarity)
    - Keyword search (BM25/TF-IDF)
    - Metadata filtering
    - Multiple score fusion strategies
    - Optional reranking
    - Configurable weights
    
    Fusion strategies:
    - RRF (Reciprocal Rank Fusion) - position-based
    - Weighted Sum - score-based with weights
    - Max/Min - take best/worst scores
    
    Best for: Comprehensive retrieval combining multiple signals
    """

    def __init__(
        self,
        vector_store=None,
        keyword_index=None,
        llm_client=None,
        mcp_client=None,
    ):
        """
        Initialize Hybrid Search agent.
        
        Args:
            vector_store: Vector database for semantic search
            keyword_index: BM25/keyword index
            llm_client: LLM for reranking
            mcp_client: MCP client
        """
        self.vector_store = vector_store
        self.keyword_index = keyword_index
        self.llm_client = llm_client
        self.mcp_client = mcp_client
        logger.info("Hybrid Search Agent initialized")

    async def execute(self, request: HybridSearchRequest) -> HybridSearchResult:
        """
        Execute hybrid search.
        
        Args:
            request: HybridSearchRequest
            
        Returns:
            HybridSearchResult with fused results
        """
        logger.info(
            "Starting hybrid search",
            query=request.query,
            strategy=request.fusion_strategy.value,
        )
        
        # Perform semantic search
        semantic_results = await self._semantic_search(
            request.query, top_k=request.top_k * 2
        )
        
        # Perform keyword search
        keyword_results = await self._keyword_search(
            request.query, top_k=request.top_k * 2
        )
        
        # Apply metadata filters
        if request.metadata_filters:
            semantic_results = self._apply_metadata_filters(
                semantic_results, request.metadata_filters
            )
            keyword_results = self._apply_metadata_filters(
                keyword_results, request.metadata_filters
            )
        
        # Use LLM to determine optimal weights based on query
        if self.llm_client or self.mcp_client:
            optimal_weights = await self._llm_determine_weights(
                request.query,
                request.semantic_weight,
                request.keyword_weight,
                request.metadata_weight,
            )
        else:
            optimal_weights = {
                "semantic": request.semantic_weight,
                "keyword": request.keyword_weight,
                "metadata": request.metadata_weight,
            }
        
        # Fuse results with LLM-optimized weights
        fused_results = await self._fuse_results(
            semantic_results,
            keyword_results,
            request.fusion_strategy,
            optimal_weights,
        )
        
        # Take top_k
        fused_results = fused_results[:request.top_k]
        
        # Optional reranking
        if request.use_reranking and len(fused_results) > 0:
            fused_results = await self._rerank_results(request.query, fused_results)
        
        logger.info("Hybrid search completed", results=len(fused_results))
        
        return HybridSearchResult(
            results=fused_results,
            total_results=len(fused_results),
            fusion_strategy=request.fusion_strategy.value,
            metadata={
                "semantic_weight": request.semantic_weight,
                "keyword_weight": request.keyword_weight,
            },
        )

    async def _semantic_search(self, query: str, top_k: int) -> List[SearchResult]:
        """Perform semantic vector search."""
        
        # Placeholder: integrate with actual vector store
        # In production, this would query ChromaDB, Pinecone, Weaviate, etc.
        
        logger.debug("Performing semantic search", query=query)
        
        try:
            # Mock semantic results
            results = []
            for i in range(min(top_k, 5)):
                results.append(SearchResult(
                    content=f"Semantic result {i+1} for: {query}",
                    doc_id=f"sem_doc_{i+1}",
                    semantic_score=0.9 - (i * 0.1),
                    metadata={"source": "semantic", "rank": i+1},
                ))
            return results
        except Exception as e:
            logger.error("Semantic search failed", error=str(e))
            return []

    async def _keyword_search(self, query: str, top_k: int) -> List[SearchResult]:
        """Perform keyword/BM25 search."""
        
        logger.debug("Performing keyword search", query=query)
        
        try:
            # Mock keyword results
            results = []
            for i in range(min(top_k, 5)):
                results.append(SearchResult(
                    content=f"Keyword result {i+1} for: {query}",
                    doc_id=f"kw_doc_{i+1}",
                    keyword_score=0.85 - (i * 0.12),
                    metadata={"source": "keyword", "rank": i+1},
                ))
            return results
        except Exception as e:
            logger.error("Keyword search failed", error=str(e))
            return []

    def _apply_metadata_filters(
        self, results: List[SearchResult], filters: Dict[str, Any]
    ) -> List[SearchResult]:
        """Apply metadata filters to results."""
        
        filtered = []
        for result in results:
            match = True
            for key, value in filters.items():
                if key not in result.metadata or result.metadata[key] != value:
                    match = False
                    break
            if match:
                filtered.append(result)
        
        logger.debug("Metadata filtering", before=len(results), after=len(filtered))
        return filtered

    async def _llm_determine_weights(
        self, query: str, default_semantic: float, default_keyword: float, default_metadata: float
    ) -> Dict[str, float]:
        """Use LLM to determine optimal fusion weights based on query characteristics."""
        
        prompt = f"""Analyze this search query and suggest optimal weights for hybrid search:

Query: "{query}"

Consider:
- Conceptual/abstract queries need high semantic weight (0.6-0.8)
- Queries with specific keywords/names need high keyword weight (0.5-0.7)
- Queries about metadata (dates, authors, sources) need high metadata weight (0.2-0.4)

Provide weights that sum to 1.0 in this exact format:
semantic=X.X, keyword=X.X, metadata=X.X

Weights:"""
        
        try:
            if self.llm_client:
                response = await self.llm_client.generate(prompt=prompt, max_tokens=50, temperature=0.5)
                weights_str = response.content.strip()
            elif self.mcp_client:
                response = await self.mcp_client.call_tool(
                    "llm_generate", {"prompt": prompt, "max_tokens": 50, "temperature": 0.5}
                )
                weights_str = response.get("content", "").strip()
            
            # Parse weights
            weights = {"semantic": default_semantic, "keyword": default_keyword, "metadata": default_metadata}
            for part in weights_str.split(","):
                if "=" in part:
                    key, value = part.split("=")
                    key = key.strip().lower()
                    if key in weights:
                        weights[key] = float(value.strip())
            
            # Normalize to sum to 1.0
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}
            
            logger.info("LLM-optimized weights", weights=weights)
            return weights
            
        except Exception as e:
            logger.warning(f"LLM weight determination failed: {e}, using defaults")
            return {"semantic": default_semantic, "keyword": default_keyword, "metadata": default_metadata}

    async def _fuse_results(
        self,
        semantic_results: List[SearchResult],
        keyword_results: List[SearchResult],
        strategy: FusionStrategy,
        weights: Dict[str, float],
    ) -> List[SearchResult]:
        """Fuse results from multiple sources."""
        
        # Merge results by doc_id
        all_results = {}
        
        for result in semantic_results:
            all_results[result.doc_id] = result
        
        for result in keyword_results:
            if result.doc_id in all_results:
                # Update existing
                all_results[result.doc_id].keyword_score = result.keyword_score
            else:
                # Add new
                all_results[result.doc_id] = result
        
        # Apply fusion strategy
        fused = list(all_results.values())
        
        if strategy == FusionStrategy.RRF:
            fused = self._reciprocal_rank_fusion(semantic_results, keyword_results)
        
        elif strategy == FusionStrategy.WEIGHTED:
            for result in fused:
                result.final_score = (
                    result.semantic_score * weights["semantic"] +
                    result.keyword_score * weights["keyword"] +
                    result.metadata_score * weights["metadata"]
                )
        
        elif strategy == FusionStrategy.MAX:
            for result in fused:
                result.final_score = max(
                    result.semantic_score,
                    result.keyword_score,
                    result.metadata_score,
                )
        
        elif strategy == FusionStrategy.MIN:
            for result in fused:
                result.final_score = min(
                    result.semantic_score or 0,
                    result.keyword_score or 0,
                    result.metadata_score or 0,
                )
        
        # Sort by final score
        fused.sort(key=lambda x: x.final_score, reverse=True)
        
        return fused

    def _reciprocal_rank_fusion(
        self, semantic_results: List[SearchResult], keyword_results: List[SearchResult], k: int = 60
    ) -> List[SearchResult]:
        """Apply Reciprocal Rank Fusion (RRF)."""
        
        scores = {}
        
        # RRF formula: score = sum(1 / (k + rank))
        for rank, result in enumerate(semantic_results, 1):
            if result.doc_id not in scores:
                scores[result.doc_id] = {"result": result, "score": 0}
            scores[result.doc_id]["score"] += 1 / (k + rank)
        
        for rank, result in enumerate(keyword_results, 1):
            if result.doc_id not in scores:
                scores[result.doc_id] = {"result": result, "score": 0}
            scores[result.doc_id]["score"] += 1 / (k + rank)
        
        # Build final results
        fused = []
        for doc_id, data in scores.items():
            result = data["result"]
            result.final_score = data["score"]
            fused.append(result)
        
        fused.sort(key=lambda x: x.final_score, reverse=True)
        return fused

    async def _rerank_results(
        self, query: str, results: List[SearchResult]
    ) -> List[SearchResult]:
        """Rerank results using LLM."""
        
        if not self.llm_client and not self.mcp_client:
            return results
        
        logger.debug("Reranking results")
        
        # Build reranking prompt
        results_text = "\n".join([
            f"{i+1}. {r.content[:100]}..." for i, r in enumerate(results)
        ])
        
        prompt = f"""Rerank these search results for the query: "{query}"

Results:
{results_text}

Provide reranked order as numbers only (e.g., "3, 1, 5, 2, 4"):"""

        try:
            if self.llm_client:
                response = await self.llm_client.generate(prompt=prompt, max_tokens=100)
                reranked_order = response.content
            elif self.mcp_client:
                response = await self.mcp_client.call_tool("llm_generate", {"prompt": prompt})
                reranked_order = response.get("content", "")
            
            # Parse order
            order = [int(x.strip()) - 1 for x in reranked_order.split(",") if x.strip().isdigit()]
            
            # Reorder results
            reranked = []
            for idx in order:
                if 0 <= idx < len(results):
                    reranked.append(results[idx])
            
            # Add any missing results
            for result in results:
                if result not in reranked:
                    reranked.append(result)
            
            return reranked
        except:
            return results
