"""
Source Citation Agent

Generates properly formatted citations for retrieved sources
in multiple academic and professional styles (APA, MLA, Chicago, IEEE, etc.)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import logging

logger = logging.getLogger(__name__)


class CitationStyle(Enum):
    """Supported citation styles."""
    APA = "apa"  # American Psychological Association
    MLA = "mla"  # Modern Language Association
    CHICAGO = "chicago"  # Chicago Manual of Style
    HARVARD = "harvard"  # Harvard Reference
    IEEE = "ieee"  # Institute of Electrical and Electronics Engineers
    VANCOUVER = "vancouver"  # Vancouver System
    AMA = "ama"  # American Medical Association
    BLUEBOOK = "bluebook"  # Legal citations


@dataclass
class SourceMetadata:
    """Metadata for a source."""
    
    title: str
    authors: List[str] = field(default_factory=list)
    year: Optional[str] = None
    publisher: Optional[str] = None
    url: Optional[str] = None
    doi: Optional[str] = None
    journal: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    access_date: Optional[str] = None
    source_type: str = "article"  # article, book, website, legal, patent


@dataclass
class CitationRequest:
    """Request for citation generation."""
    
    sources: List[SourceMetadata]
    style: CitationStyle = CitationStyle.APA
    include_in_text: bool = True
    alphabetize: bool = True


@dataclass
class Citation:
    """A formatted citation."""
    
    full_citation: str
    in_text_citation: str
    source_metadata: SourceMetadata


@dataclass
class CitationResult:
    """Result from citation generation."""
    
    citations: List[Citation]
    bibliography: str
    style: str
    metadata: Dict = field(default_factory=dict)


class SourceCitationAgent:
    """
    Universal Source Citation agent for RAG.
    
    Features:
    - Multiple citation styles (APA, MLA, Chicago, IEEE, etc.)
    - Full bibliography generation
    - In-text citation generation
    - Automatic alphabetization
    - URL and DOI formatting
    - Multiple source types (articles, books, websites, legal, patents)
    
    Best for: Academic writing, research, documentation, compliance
    """

    def __init__(self, llm_client=None, mcp_client=None):
        """Initialize Source Citation agent."""
        self.llm_client = llm_client
        self.mcp_client = mcp_client
        logger.info("Source Citation Agent initialized")

    async def execute(self, request: CitationRequest) -> CitationResult:
        """
        Generate citations for sources.
        
        Args:
            request: CitationRequest
            
        Returns:
            CitationResult with formatted citations
        """
        logger.info(
            "Generating citations",
            style=request.style.value,
            sources=len(request.sources),
        )
        
        # Generate citations for each source (LLM-enhanced for complex cases)
        citations = []
        for source in request.sources:
            citation = await self._generate_citation(source, request.style, request.include_in_text)
            citations.append(citation)
        
        # Alphabetize if requested
        if request.alphabetize:
            citations.sort(key=lambda c: self._get_sort_key(c.source_metadata))
        
        # Generate bibliography
        bibliography = self._generate_bibliography(citations, request.style)
        
        logger.info("Citations generated", count=len(citations))
        
        return CitationResult(
            citations=citations,
            bibliography=bibliography,
            style=request.style.value,
            metadata={"total_sources": len(citations)},
        )

    async def _generate_citation(
        self, source: SourceMetadata, style: CitationStyle, include_in_text: bool
    ) -> Citation:
        """Generate citation for a single source."""
        
        # Check if source has unusual characteristics that need LLM help
        needs_llm = (
            not source.authors or
            (not source.year and not source.url) or
            source.source_type not in ["article", "book", "website"]
        )
        
        # Use LLM for complex/unusual sources
        if needs_llm and (self.llm_client or self.mcp_client):
            return await self._llm_generate_citation(source, style, include_in_text)
        
        # Standard rule-based formatting for common cases
        if style == CitationStyle.APA:
            full = self._format_apa(source)
            in_text = self._format_apa_in_text(source) if include_in_text else ""
        
        elif style == CitationStyle.MLA:
            full = self._format_mla(source)
            in_text = self._format_mla_in_text(source) if include_in_text else ""
        
        elif style == CitationStyle.CHICAGO:
            full = self._format_chicago(source)
            in_text = self._format_chicago_in_text(source) if include_in_text else ""
        
        elif style == CitationStyle.IEEE:
            full = self._format_ieee(source)
            in_text = "" if include_in_text else ""  # IEEE uses numbers
        
        elif style == CitationStyle.HARVARD:
            full = self._format_harvard(source)
            in_text = self._format_harvard_in_text(source) if include_in_text else ""
        
        else:
            # Default to APA
            full = self._format_apa(source)
            in_text = self._format_apa_in_text(source) if include_in_text else ""
        
        return Citation(
            full_citation=full,
            in_text_citation=in_text,
            source_metadata=source,
        )

    def _format_apa(self, source: SourceMetadata) -> str:
        """Format citation in APA style."""
        
        # Author(s)
        authors_str = self._format_authors_apa(source.authors)
        
        # Year
        year_str = f"({source.year})." if source.year else "(n.d.)."
        
        # Title
        title_str = f"{source.title}."
        
        # Publication info
        pub_info = []
        if source.journal:
            pub_info.append(f"*{source.journal}*")
            if source.volume:
                pub_info.append(f"{source.volume}")
                if source.issue:
                    pub_info[-1] += f"({source.issue})"
            if source.pages:
                pub_info.append(source.pages)
        elif source.publisher:
            pub_info.append(source.publisher)
        
        pub_str = ", ".join(pub_info) + "." if pub_info else ""
        
        # DOI or URL
        link_str = ""
        if source.doi:
            link_str = f"https://doi.org/{source.doi}"
        elif source.url:
            link_str = source.url
        
        # Combine
        parts = [authors_str, year_str, title_str, pub_str, link_str]
        return " ".join([p for p in parts if p])

    def _format_authors_apa(self, authors: List[str]) -> str:
        """Format authors in APA style."""
        if not authors:
            return ""
        
        if len(authors) == 1:
            return f"{authors[0]}."
        elif len(authors) == 2:
            return f"{authors[0]}, & {authors[1]}."
        else:
            # Last name, First initial for all
            formatted = ", ".join(authors[:-1])
            return f"{formatted}, & {authors[-1]}."

    def _format_apa_in_text(self, source: SourceMetadata) -> str:
        """Format in-text citation in APA style."""
        if not source.authors:
            return f"({source.title[:20]}, {source.year or 'n.d.'})"
        
        first_author = source.authors[0].split()[-1]  # Last name
        year = source.year or "n.d."
        
        if len(source.authors) == 1:
            return f"({first_author}, {year})"
        elif len(source.authors) == 2:
            second_author = source.authors[1].split()[-1]
            return f"({first_author} & {second_author}, {year})"
        else:
            return f"({first_author} et al., {year})"

    def _format_mla(self, source: SourceMetadata) -> str:
        """Format citation in MLA style."""
        
        # Author(s)
        authors_str = self._format_authors_mla(source.authors)
        
        # Title
        title_str = f'"{source.title}."' if source.source_type == "article" else f"*{source.title}.*"
        
        # Publication info
        pub_info = []
        if source.journal:
            pub_info.append(f"*{source.journal}*")
            if source.volume:
                pub_info.append(f"vol. {source.volume}")
            if source.issue:
                pub_info.append(f"no. {source.issue}")
            if source.year:
                pub_info.append(source.year)
            if source.pages:
                pub_info.append(f"pp. {source.pages}")
        
        pub_str = ", ".join(pub_info) + "." if pub_info else ""
        
        # URL
        url_str = source.url if source.url else ""
        
        parts = [authors_str, title_str, pub_str, url_str]
        return " ".join([p for p in parts if p])

    def _format_authors_mla(self, authors: List[str]) -> str:
        """Format authors in MLA style."""
        if not authors:
            return ""
        
        if len(authors) == 1:
            return f"{authors[0]}."
        elif len(authors) == 2:
            return f"{authors[0]}, and {authors[1]}."
        else:
            return f"{authors[0]}, et al."

    def _format_mla_in_text(self, source: SourceMetadata) -> str:
        """Format in-text citation in MLA style."""
        if not source.authors:
            return f'("{source.title[:20]}")'
        
        first_author = source.authors[0].split()[-1]
        
        if len(source.authors) == 1:
            return f"({first_author})"
        elif len(source.authors) == 2:
            second_author = source.authors[1].split()[-1]
            return f"({first_author} and {second_author})"
        else:
            return f"({first_author} et al.)"

    def _format_chicago(self, source: SourceMetadata) -> str:
        """Format citation in Chicago style."""
        
        # Author(s)
        authors_str = self._format_authors_chicago(source.authors)
        
        # Title
        title_str = f'"{source.title}."' if source.source_type == "article" else f"*{source.title}.*"
        
        # Publication info
        pub_info = []
        if source.journal:
            pub_info.append(f"*{source.journal}*")
            if source.volume and source.issue:
                pub_info.append(f"{source.volume}, no. {source.issue}")
            if source.year:
                pub_info.append(f"({source.year})")
            if source.pages:
                pub_info.append(f": {source.pages}")
        
        pub_str = " ".join(pub_info) + "." if pub_info else ""
        
        parts = [authors_str, title_str, pub_str]
        return " ".join([p for p in parts if p])

    def _format_authors_chicago(self, authors: List[str]) -> str:
        """Format authors in Chicago style."""
        if not authors:
            return ""
        
        if len(authors) == 1:
            return f"{authors[0]}."
        else:
            formatted = ", ".join(authors[:-1])
            return f"{formatted}, and {authors[-1]}."

    def _format_chicago_in_text(self, source: SourceMetadata) -> str:
        """Format in-text citation in Chicago style."""
        if not source.authors:
            return f'("{source.title[:20]}" {source.year or "n.d."})'
        
        first_author = source.authors[0].split()[-1]
        return f"({first_author} {source.year or 'n.d.'})"

    def _format_ieee(self, source: SourceMetadata) -> str:
        """Format citation in IEEE style."""
        
        # Author(s)
        authors_str = ", ".join(source.authors) if source.authors else ""
        
        # Title
        title_str = f'"{source.title},"'
        
        # Journal
        journal_str = f"*{source.journal}*," if source.journal else ""
        
        # Volume, issue, pages
        vol_str = ""
        if source.volume:
            vol_str = f"vol. {source.volume}"
            if source.issue:
                vol_str += f", no. {source.issue}"
            if source.pages:
                vol_str += f", pp. {source.pages}"
            vol_str += ","
        
        # Year
        year_str = f"{source.year}." if source.year else ""
        
        parts = [authors_str, title_str, journal_str, vol_str, year_str]
        return " ".join([p for p in parts if p])

    def _format_harvard(self, source: SourceMetadata) -> str:
        """Format citation in Harvard style."""
        return self._format_apa(source)  # Similar to APA

    def _format_harvard_in_text(self, source: SourceMetadata) -> str:
        """Format in-text citation in Harvard style."""
        return self._format_apa_in_text(source)  # Similar to APA

    async def _llm_generate_citation(
        self, source: SourceMetadata, style: CitationStyle, include_in_text: bool
    ) -> Citation:
        """Use LLM to generate citation for complex or unusual sources."""
        
        # Build source description
        source_info = f"""Title: {source.title}
Authors: {', '.join(source.authors) if source.authors else 'Unknown'}
Year: {source.year or 'n.d.'}
Publisher: {source.publisher or 'N/A'}
URL: {source.url or 'N/A'}
Type: {source.source_type}"""
        
        if source.journal:
            source_info += f"\nJournal: {source.journal}"
        if source.volume:
            source_info += f"\nVolume: {source.volume}, Issue: {source.issue or 'N/A'}"
        if source.pages:
            source_info += f"\nPages: {source.pages}"
        
        prompt = f"""Generate a properly formatted citation in {style.value.upper()} style:

{source_info}

Provide:
1. Full citation (bibliography entry)
2. In-text citation (if applicable)

Follow standard {style.value.upper()} formatting guidelines exactly.

Full citation:"""
        
        try:
            if self.llm_client:
                response = await self.llm_client.generate(prompt=prompt, max_tokens=300, temperature=0.3)
                citation_text = response.content.strip()
            elif self.mcp_client:
                response = await self.mcp_client.call_tool(
                    "llm_generate", {"prompt": prompt, "max_tokens": 300, "temperature": 0.3}
                )
                citation_text = response.get("content", "").strip()
            
            # Parse response (assume first line is full, second is in-text if present)
            lines = [line.strip() for line in citation_text.split("\n") if line.strip()]
            full_citation = lines[0] if lines else self._format_apa(source)
            in_text_citation = lines[1] if len(lines) > 1 and include_in_text else ""
            
            logger.info("LLM-generated citation", style=style.value)
            
            return Citation(
                full_citation=full_citation,
                in_text_citation=in_text_citation,
                source_metadata=source,
            )
        except Exception as e:
            logger.warning(f"LLM citation generation failed: {e}, using fallback")
            # Fallback to standard formatting
            return Citation(
                full_citation=self._format_apa(source),
                in_text_citation=self._format_apa_in_text(source) if include_in_text else "",
                source_metadata=source,
            )

    def _get_sort_key(self, source: SourceMetadata) -> str:
        """Get sort key for alphabetization."""
        if source.authors:
            return source.authors[0].split()[-1].lower()  # Last name
        return source.title.lower()

    def _generate_bibliography(self, citations: List[Citation], style: CitationStyle) -> str:
        """Generate formatted bibliography."""
        
        header = {
            CitationStyle.APA: "References",
            CitationStyle.MLA: "Works Cited",
            CitationStyle.CHICAGO: "Bibliography",
            CitationStyle.IEEE: "References",
            CitationStyle.HARVARD: "Reference List",
        }.get(style, "References")
        
        bib_lines = [f"# {header}\n"]
        
        for i, citation in enumerate(citations, 1):
            if style == CitationStyle.IEEE:
                bib_lines.append(f"[{i}] {citation.full_citation}")
            else:
                bib_lines.append(citation.full_citation)
            bib_lines.append("")  # Blank line
        
        return "\n".join(bib_lines)
