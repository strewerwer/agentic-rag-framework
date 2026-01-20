"""
Cross-Reference Validation Agent

Validates information by cross-referencing multiple sources,
detecting inconsistencies, and assessing claim reliability.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum

import logging

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Validation status for claims."""
    VERIFIED = "verified"  # Confirmed by multiple sources
    LIKELY = "likely"  # Confirmed by some sources
    UNCERTAIN = "uncertain"  # Conflicting information
    UNVERIFIED = "unverified"  # Single source only
    CONTRADICTED = "contradicted"  # Directly contradicted


@dataclass
class Claim:
    """A factual claim to validate."""
    
    content: str
    source: str
    confidence: float = 1.0


@dataclass
class ValidationEvidence:
    """Evidence for or against a claim."""
    
    claim: str
    supporting_sources: List[str] = field(default_factory=list)
    contradicting_sources: List[str] = field(default_factory=list)
    neutral_sources: List[str] = field(default_factory=list)


@dataclass
class CrossReferenceRequest:
    """Request for cross-reference validation."""
    
    primary_content: str
    reference_sources: List[str]  # Additional source contents
    claims_to_validate: Optional[List[str]] = None
    min_sources_for_verification: int = 2


@dataclass
class ValidationResult:
    """Validation result for a claim."""
    
    claim: str
    status: ValidationStatus
    confidence: float
    supporting_count: int
    contradicting_count: int
    evidence: ValidationEvidence


@dataclass
class CrossReferenceResult:
    """Result from cross-reference validation."""
    
    validations: List[ValidationResult]
    overall_reliability: float
    inconsistencies: List[str]
    metadata: Dict = field(default_factory=dict)


class CrossReferenceValidationAgent:
    """
    Universal Cross-Reference Validation agent for RAG.
    
    Features:
    - Multi-source fact verification
    - Inconsistency detection
    - Claim extraction and validation
    - Source reliability assessment
    - Contradiction resolution
    - Evidence aggregation
    
    Best for: Fact-checking, research, critical information verification
    """

    def __init__(self, llm_client=None, mcp_client=None):
        """Initialize Cross-Reference Validation agent."""
        self.llm_client = llm_client
        self.mcp_client = mcp_client
        logger.info("Cross-Reference Validation Agent initialized")

    async def execute(self, request: CrossReferenceRequest) -> CrossReferenceResult:
        """
        Validate information through cross-referencing.
        
        Args:
            request: CrossReferenceRequest
            
        Returns:
            CrossReferenceResult with validation results
        """
        logger.info(
            "Starting cross-reference validation",
            reference_sources=len(request.reference_sources),
        )
        
        # Extract claims if not provided
        if request.claims_to_validate:
            claims = request.claims_to_validate
        else:
            claims = await self._extract_claims(request.primary_content)
        
        # Validate each claim
        validations = []
        for claim in claims:
            validation = await self._validate_claim(
                claim,
                request.primary_content,
                request.reference_sources,
                request.min_sources_for_verification,
            )
            validations.append(validation)
        
        # Calculate overall reliability
        overall_reliability = self._calculate_overall_reliability(validations)
        
        # Identify inconsistencies
        inconsistencies = self._identify_inconsistencies(validations)
        
        logger.info(
            "Cross-reference validation completed",
            claims_validated=len(validations),
            reliability=overall_reliability,
        )
        
        return CrossReferenceResult(
            validations=validations,
            overall_reliability=overall_reliability,
            inconsistencies=inconsistencies,
            metadata={
                "total_claims": len(claims),
                "reference_sources": len(request.reference_sources),
            },
        )

    async def _extract_claims(self, content: str) -> List[str]:
        """Extract factual claims from content."""
        
        if self.llm_client or self.mcp_client:
            return await self._llm_extract_claims(content)
        
        # Fallback: simple sentence splitting
        sentences = content.split(". ")
        claims = [s.strip() + "." for s in sentences if len(s.split()) > 3]
        return claims[:10]  # Limit to 10 claims

    async def _llm_extract_claims(self, content: str) -> List[str]:
        """Extract claims using LLM."""
        
        prompt = f"""Extract 5-10 key factual claims from this content:

Content: {content[:500]}

List claims (one per line):
1."""

        try:
            if self.llm_client:
                response = await self.llm_client.generate(prompt=prompt, max_tokens=300)
                result = response.content
            elif self.mcp_client:
                response = await self.mcp_client.call_tool(
                    "llm_generate", {"prompt": prompt, "max_tokens": 300}
                )
                result = response.get("content", "")
            
            # Parse claims
            claims = []
            for line in result.strip().split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    claim = line.lstrip('0123456789.-*) ').strip()
                    if claim:
                        claims.append(claim)
            
            return claims[:10]
        except:
            return []

    async def _validate_claim(
        self,
        claim: str,
        primary_content: str,
        reference_sources: List[str],
        min_sources: int,
    ) -> ValidationResult:
        """Validate a single claim against reference sources."""
        
        supporting_sources = []
        contradicting_sources = []
        neutral_sources = []
        
        # Check primary content (LLM-driven)
        if await self._claim_supported_by(claim, primary_content):
            supporting_sources.append("primary")
        
        # Check reference sources (LLM-driven)
        for i, source in enumerate(reference_sources):
            support_level = await self._check_claim_in_source(claim, source)
            
            if support_level > 0.7:
                supporting_sources.append(f"source_{i+1}")
            elif support_level < 0.3:
                contradicting_sources.append(f"source_{i+1}")
            else:
                neutral_sources.append(f"source_{i+1}")
        
        # Determine status
        status = self._determine_status(
            len(supporting_sources),
            len(contradicting_sources),
            min_sources,
        )
        
        # Calculate confidence
        total_sources = len(supporting_sources) + len(contradicting_sources) + len(neutral_sources)
        if total_sources > 0:
            confidence = len(supporting_sources) / total_sources
        else:
            confidence = 0.0
        
        # Create evidence
        evidence = ValidationEvidence(
            claim=claim,
            supporting_sources=supporting_sources,
            contradicting_sources=contradicting_sources,
            neutral_sources=neutral_sources,
        )
        
        return ValidationResult(
            claim=claim,
            status=status,
            confidence=confidence,
            supporting_count=len(supporting_sources),
            contradicting_count=len(contradicting_sources),
            evidence=evidence,
        )

    async def _claim_supported_by(self, claim: str, content: str) -> bool:
        """Check if claim is supported by content using LLM."""
        
        # Use LLM for semantic understanding
        if self.llm_client or self.mcp_client:
            prompt = f"""Does this content support the claim?

Claim: {claim}

Content: {content[:300]}

Respond with YES if the content supports the claim, NO if it contradicts or doesn't support it:"""
            
            try:
                if self.llm_client:
                    response = await self.llm_client.generate(prompt=prompt, max_tokens=10, temperature=0.2)
                    answer = response.content.strip().upper()
                elif self.mcp_client:
                    response = await self.mcp_client.call_tool(
                        "llm_generate", {"prompt": prompt, "max_tokens": 10, "temperature": 0.2}
                    )
                    answer = response.get("content", "").strip().upper()
                
                return "YES" in answer
            except:
                pass
        
        # Fallback: keyword matching
        claim_words = set(claim.lower().split())
        content_words = set(content.lower().split())
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "to", "for"}
        claim_words -= stop_words
        overlap = len(claim_words & content_words)
        return overlap >= len(claim_words) * 0.5

    async def _check_claim_in_source(self, claim: str, source: str) -> float:
        """Check claim against source using LLM, return support level (0-1)."""
        
        # Use LLM for nuanced support assessment
        if self.llm_client or self.mcp_client:
            prompt = f"""Rate how well this source supports the claim on a scale of 0-10:

Claim: {claim}

Source: {source[:400]}

Scale:
- 0-2: Contradicts or strongly opposes the claim
- 3-4: Doesn't support the claim
- 5-6: Neutral or unclear
- 7-8: Somewhat supports the claim
- 9-10: Strongly supports the claim

Provide only the number (0-10):"""
            
            try:
                if self.llm_client:
                    response = await self.llm_client.generate(prompt=prompt, max_tokens=10, temperature=0.3)
                    score_str = response.content.strip()
                elif self.mcp_client:
                    response = await self.mcp_client.call_tool(
                        "llm_generate", {"prompt": prompt, "max_tokens": 10, "temperature": 0.3}
                    )
                    score_str = response.get("content", "5").strip()
                
                # Parse score and normalize to 0-1
                score = float(score_str)
                return max(0.0, min(1.0, score / 10.0))
            except:
                pass
        
        # Fallback: keyword matching
        claim_lower = claim.lower()
        source_lower = source.lower()
        claim_words = set(claim_lower.split())
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "to", "for"}
        claim_words -= stop_words
        
        if not claim_words:
            return 0.5
        
        matches = sum(1 for word in claim_words if word in source_lower)
        support_level = matches / len(claim_words)
        
        negation_words = ["not", "no", "never", "false", "incorrect", "wrong"]
        has_negation = any(neg in source_lower for neg in negation_words)
        
        if has_negation and support_level > 0.5:
            return 0.2
        
        return support_level

    def _determine_status(
        self, supporting_count: int, contradicting_count: int, min_sources: int
    ) -> ValidationStatus:
        """Determine validation status based on counts."""
        
        if contradicting_count > supporting_count:
            return ValidationStatus.CONTRADICTED
        
        elif supporting_count >= min_sources:
            return ValidationStatus.VERIFIED
        
        elif supporting_count > 0 and contradicting_count == 0:
            if supporting_count == 1:
                return ValidationStatus.UNVERIFIED
            else:
                return ValidationStatus.LIKELY
        
        elif supporting_count > 0 and contradicting_count > 0:
            return ValidationStatus.UNCERTAIN
        
        else:
            return ValidationStatus.UNVERIFIED

    def _calculate_overall_reliability(self, validations: List[ValidationResult]) -> float:
        """Calculate overall reliability score."""
        
        if not validations:
            return 0.0
        
        # Weight by status
        status_scores = {
            ValidationStatus.VERIFIED: 1.0,
            ValidationStatus.LIKELY: 0.7,
            ValidationStatus.UNVERIFIED: 0.4,
            ValidationStatus.UNCERTAIN: 0.3,
            ValidationStatus.CONTRADICTED: 0.1,
        }
        
        total_score = sum(status_scores[v.status] for v in validations)
        return total_score / len(validations)

    def _identify_inconsistencies(self, validations: List[ValidationResult]) -> List[str]:
        """Identify key inconsistencies."""
        
        inconsistencies = []
        
        for validation in validations:
            if validation.status == ValidationStatus.CONTRADICTED:
                inconsistencies.append(
                    f"Contradicted claim: {validation.claim} "
                    f"(contradicted by {validation.contradicting_count} sources)"
                )
            
            elif validation.status == ValidationStatus.UNCERTAIN:
                inconsistencies.append(
                    f"Uncertain claim: {validation.claim} "
                    f"({validation.supporting_count} supporting, "
                    f"{validation.contradicting_count} contradicting)"
                )
        
        return inconsistencies
