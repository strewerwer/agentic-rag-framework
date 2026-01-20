"""
Answer Fusion Agent

Combines multiple answers from different sources or models
using ensemble techniques and consistency analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

import logging

logger = logging.getLogger(__name__)


class FusionStrategy(Enum):
    """Answer fusion strategies."""
    VOTING = "voting"  # Majority voting for similar answers
    WEIGHTED = "weighted"  # Weight by source/model reliability
    CONCATENATION = "concatenation"  # Combine all unique information
    BEST_OF_N = "best_of_n"  # Select single best answer
    HYBRID = "hybrid"  # Combine strategies adaptively


@dataclass
class Answer:
    """A single answer from a source."""
    
    content: str
    source: str
    confidence: float = 1.0
    supporting_evidence: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class AnswerFusionRequest:
    """Request for answer fusion."""
    
    answers: List[Answer]
    query: str
    strategy: FusionStrategy = FusionStrategy.HYBRID
    source_weights: Optional[Dict[str, float]] = None
    min_confidence: float = 0.5


@dataclass
class AnswerFusionResult:
    """Result from answer fusion."""
    
    fused_answer: str
    confidence: float
    sources_used: List[str]
    fusion_strategy: str
    consistency_score: float
    metadata: Dict = field(default_factory=dict)


class AnswerFusionAgent:
    """
    Universal Answer Fusion agent for RAG.
    
    Features:
    - Multiple fusion strategies (voting, weighted, concatenation, best-of-n)
    - Consistency analysis across answers
    - Source reliability weighting
    - Conflict resolution
    - Evidence aggregation
    - Adaptive strategy selection
    
    Best for: Multi-source RAG, ensemble systems, verification
    """

    def __init__(self, llm_client=None, mcp_client=None):
        """Initialize Answer Fusion agent."""
        self.llm_client = llm_client
        self.mcp_client = mcp_client
        logger.info("Answer Fusion Agent initialized")

    async def execute(self, request: AnswerFusionRequest) -> AnswerFusionResult:
        """
        Fuse multiple answers.
        
        Args:
            request: AnswerFusionRequest
            
        Returns:
            AnswerFusionResult with fused answer
        """
        logger.info(
            "Fusing answers",
            count=len(request.answers),
            strategy=request.strategy.value,
        )
        
        # Filter low-confidence answers
        filtered_answers = [
            a for a in request.answers if a.confidence >= request.min_confidence
        ]
        
        if not filtered_answers:
            logger.warning("No answers meet confidence threshold")
            return AnswerFusionResult(
                fused_answer="No confident answers available.",
                confidence=0.0,
                sources_used=[],
                fusion_strategy=request.strategy.value,
                consistency_score=0.0,
            )
        
        # Analyze consistency
        consistency_score = await self._analyze_consistency(filtered_answers)
        
        # Apply fusion strategy
        if request.strategy == FusionStrategy.VOTING:
            fused = await self._voting_fusion(filtered_answers)
        
        elif request.strategy == FusionStrategy.WEIGHTED:
            fused = await self._weighted_fusion(filtered_answers, request.source_weights or {})
        
        elif request.strategy == FusionStrategy.CONCATENATION:
            fused = await self._concatenation_fusion(filtered_answers)
        
        elif request.strategy == FusionStrategy.BEST_OF_N:
            fused = await self._best_of_n_fusion(filtered_answers)
        
        else:  # HYBRID
            fused = await self._hybrid_fusion(
                filtered_answers, consistency_score, request.query
            )
        
        # Calculate final confidence
        final_confidence = self._calculate_confidence(
            filtered_answers, consistency_score
        )
        
        # Extract sources used
        sources_used = [a.source for a in filtered_answers]
        
        logger.info(
            "Answer fusion completed",
            confidence=final_confidence,
            consistency=consistency_score,
        )
        
        return AnswerFusionResult(
            fused_answer=fused,
            confidence=final_confidence,
            sources_used=sources_used,
            fusion_strategy=request.strategy.value,
            consistency_score=consistency_score,
            metadata={"filtered_count": len(filtered_answers)},
        )

    async def _analyze_consistency(self, answers: List[Answer]) -> float:
        """Analyze consistency across answers."""
        
        if len(answers) < 2:
            return 1.0
        
        # Simple consistency: check for common keywords
        all_words = set()
        answer_words = []
        
        for answer in answers:
            words = set(answer.content.lower().split())
            answer_words.append(words)
            all_words.update(words)
        
        # Calculate pairwise overlap
        overlaps = []
        for i in range(len(answer_words)):
            for j in range(i + 1, len(answer_words)):
                overlap = len(answer_words[i] & answer_words[j])
                union = len(answer_words[i] | answer_words[j])
                if union > 0:
                    overlaps.append(overlap / union)
        
        return sum(overlaps) / len(overlaps) if overlaps else 0.0

    async def _voting_fusion(self, answers: List[Answer]) -> str:
        """Fusion by voting for most common answer."""
        
        # Group similar answers
        answer_groups = {}
        for answer in answers:
            # Simple grouping by first 50 chars
            key = answer.content[:50].lower().strip()
            if key not in answer_groups:
                answer_groups[key] = []
            answer_groups[key].append(answer)
        
        # Find most common group
        max_count = 0
        best_answer = answers[0].content
        
        for key, group in answer_groups.items():
            if len(group) > max_count:
                max_count = len(group)
                # Take highest confidence from group
                best_answer = max(group, key=lambda a: a.confidence).content
        
        return best_answer

    async def _weighted_fusion(
        self, answers: List[Answer], source_weights: Dict[str, float]
    ) -> str:
        """Fusion by weighted combination."""
        
        # Calculate weighted scores
        weighted_answers = []
        for answer in answers:
            weight = source_weights.get(answer.source, 1.0)
            score = answer.confidence * weight
            weighted_answers.append((score, answer))
        
        # Sort by score
        weighted_answers.sort(key=lambda x: x[0], reverse=True)
        
        # Return highest weighted answer
        return weighted_answers[0][1].content

    async def _concatenation_fusion(self, answers: List[Answer]) -> str:
        """Fusion by concatenating unique information."""
        
        # Collect unique sentences
        unique_sentences = []
        seen_sentences = set()
        
        for answer in answers:
            sentences = answer.content.split(". ")
            for sentence in sentences:
                sentence = sentence.strip()
                sentence_lower = sentence.lower()
                
                # Check if similar sentence already added
                is_unique = True
                for seen in seen_sentences:
                    if self._sentence_similarity(sentence_lower, seen) > 0.8:
                        is_unique = False
                        break
                
                if is_unique and sentence:
                    unique_sentences.append(sentence)
                    seen_sentences.add(sentence_lower)
        
        # Combine sentences
        combined = ". ".join(unique_sentences)
        if combined and not combined.endswith("."):
            combined += "."
        
        return combined

    def _sentence_similarity(self, s1: str, s2: str) -> float:
        """Calculate simple sentence similarity."""
        words1 = set(s1.split())
        words2 = set(s2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0

    async def _best_of_n_fusion(self, answers: List[Answer]) -> str:
        """Select single best answer."""
        
        # Sort by confidence
        best_answer = max(answers, key=lambda a: a.confidence)
        return best_answer.content

    async def _hybrid_fusion(
        self, answers: List[Answer], consistency_score: float, query: str
    ) -> str:
        """Adaptive hybrid fusion based on consistency."""
        
        # High consistency: use voting
        if consistency_score > 0.7:
            return await self._voting_fusion(answers)
        
        # Low consistency: use LLM to synthesize
        elif consistency_score < 0.3 and (self.llm_client or self.mcp_client):
            return await self._llm_synthesis(answers, query)
        
        # Medium consistency: concatenate unique info
        else:
            return await self._concatenation_fusion(answers)

    async def _llm_synthesis(self, answers: List[Answer], query: str) -> str:
        """Use LLM to synthesize conflicting answers."""
        
        answers_text = "\n\n".join([
            f"Answer {i+1} (from {a.source}, confidence: {a.confidence:.2f}):\n{a.content}"
            for i, a in enumerate(answers)
        ])
        
        prompt = f"""Synthesize a comprehensive answer from these multiple sources:

Query: {query}

Multiple Answers:
{answers_text}

Provide a single, coherent answer that:
1. Incorporates consistent information from all sources
2. Resolves conflicts by favoring higher-confidence sources
3. Notes any significant disagreements
4. Maintains accuracy and objectivity

Synthesized Answer:"""

        try:
            if self.llm_client:
                response = await self.llm_client.generate(prompt=prompt, max_tokens=400)
                return response.content.strip()
            elif self.mcp_client:
                response = await self.mcp_client.call_tool(
                    "llm_generate", {"prompt": prompt, "max_tokens": 400}
                )
                return response.get("content", "").strip()
        except Exception as e:
            logger.error("LLM synthesis failed", error=str(e))
            # Fallback to best answer
            return await self._best_of_n_fusion(answers)

    def _calculate_confidence(
        self, answers: List[Answer], consistency_score: float
    ) -> float:
        """Calculate final confidence for fused answer."""
        
        # Average confidence of inputs
        avg_confidence = sum(a.confidence for a in answers) / len(answers)
        
        # Boost by consistency
        final_confidence = (avg_confidence + consistency_score) / 2
        
        # Boost by number of agreeing sources
        agreement_boost = min(len(answers) / 5, 0.2)  # Up to 0.2 boost
        
        return min(final_confidence + agreement_boost, 1.0)
