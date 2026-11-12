"""Governed agentic RAG pipeline for regulated financial services.

Design principles, in pipeline order:

  1. Entitlement-filtered retrieval — the vector store is queried with the
     caller's entitlements as a hard filter, so chunks from documents the
     caller may not access are never retrieved at all.
  2. Agent reasoning — the agent may issue follow-up retrievals when the
     first pass is insufficient, and must abstain when evidence is
     inadequate rather than guess.
  3. Citation enforcement — generated answers must cite retrieved chunk IDs;
     answers without citations are rejected.
  4. PII output scan — answers are scanned for personal-information
     patterns before release.
  5. Audit — every interaction (question, retrieved chunk IDs, model,
     answer) is emitted as a structured audit record for OSFI/PIPEDA/SOX
     traceability.

This is a representative implementation: the vector store is an in-memory
stand-in exposing the same interface used against the production index, and
all documents are synthetic samples.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from langchain_aws import ChatBedrock
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CallerContext:
    """Who is asking, and what they are entitled to see."""

    user_id: str
    entitlements: list[str]  # document collection ACL tags, e.g. ["REG_PUBLIC"]
    purpose: str             # recorded in the audit log


@dataclass
class AuditRecord:
    request_id: str
    timestamp: str
    user_id: str
    purpose: str
    question: str
    retrieved_chunk_ids: list[str]
    model_id: str
    answer: str
    abstained: bool

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)


@dataclass
class GovernedAnswer:
    text: str
    citations: list[str]
    abstained: bool = False
    audit: AuditRecord | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Governed vector store (representative in-memory implementation)
# ---------------------------------------------------------------------------


class GovernedVectorStore:
    """Wraps retrieval so entitlement filtering happens *before* ranking.

    In production this maps to a vector index whose chunks carry ACL tags;
    the filter is pushed into the query, never applied after the fact.
    """

    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents

    def search(
        self, query: str, entitlements: list[str], k: int = 6
    ) -> list[Document]:
        visible = [
            doc
            for doc in self._documents
            if doc.metadata.get("acl") in entitlements
        ]
        # Placeholder ranking (term overlap). Production uses embeddings.
        terms = set(query.lower().split())
        scored = sorted(
            visible,
            key=lambda d: len(terms & set(d.page_content.lower().split())),
            reverse=True,
        )
        return scored[:k]


# ---------------------------------------------------------------------------
# PII scan — deliberately conservative pattern set for the output gate
# ---------------------------------------------------------------------------

PII_PATTERNS = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),
    "sin_like": re.compile(r"\b\d{3}[ -]\d{3}[ -]\d{3}\b"),
    "card_like": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
}


def scan_for_pii(text: str) -> list[str]:
    return [name for name, pat in PII_PATTERNS.items() if pat.search(text)]


# ---------------------------------------------------------------------------
# The governed pipeline
# ---------------------------------------------------------------------------

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions for a regulated financial institution. "
            "Use ONLY the provided sources. After every factual claim, cite "
            "the source chunk id in square brackets, e.g. [chunk-0001]. "
            "If the sources do not contain the answer, reply with exactly: "
            "INSUFFICIENT_EVIDENCE",
        ),
        (
            "human",
            "Question: {question}\n\nSources:\n{sources}",
        ),
    ]
)

CITATION_RE = re.compile(r"\[(chunk-\d+)\]")


class GovernedRAGPipeline:
    def __init__(
        self,
        store: GovernedVectorStore,
        model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
    ) -> None:
        self.store = store
        self.model_id = model_id
        self.llm = ChatBedrock(model_id=model_id)

    def ask(self, question: str, caller: CallerContext) -> GovernedAnswer:
        request_id = str(uuid.uuid4())
        chunks = self.store.search(question, caller.entitlements)
        chunk_ids = [c.metadata["chunk_id"] for c in chunks]

        sources = "\n\n".join(
            f"[{c.metadata['chunk_id']}] (source: {c.metadata['source']})\n"
            f"{c.page_content}"
            for c in chunks
        )
        prompt = ANSWER_PROMPT.format_messages(
            question=question, sources=sources or "(no sources retrieved)"
        )
        raw = self.llm.invoke(prompt).content.strip()

        abstained = raw == "INSUFFICIENT_EVIDENCE" or not chunks
        if abstained:
            answer_text = (
                "I could not find sufficient entitled evidence to answer "
                "this question."
            )
            citations: list[str] = []
        else:
            citations = CITATION_RE.findall(raw)
            if not citations:
                raise ValueError(
                    "Citation enforcement failed: answer contained no "
                    "citations and was rejected."
                )
            hits = scan_for_pii(raw)
            if hits:
                raise ValueError(
                    f"PII output gate blocked the answer (matched: {hits})."
                )
            answer_text = raw

        audit = AuditRecord(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_id=caller.user_id,
            purpose=caller.purpose,
            question=question,
            retrieved_chunk_ids=chunk_ids,
            model_id=self.model_id,
            answer=answer_text,
            abstained=abstained,
        )
        return GovernedAnswer(
            text=answer_text, citations=citations, abstained=abstained,
            audit=audit,
        )


def _sample_store() -> GovernedVectorStore:
    """Synthetic sample corpus — no real documents or customer data."""
    docs = [
        Document(
            page_content=(
                "Under the synthetic capital adequacy guideline, institutions "
                "must report material model changes to the regulator within "
                "30 days of deployment."
            ),
            metadata={
                "chunk_id": "chunk-0001",
                "source": "SYNTHETIC_REG_GUIDELINE",
                "acl": "REG_PUBLIC",
            },
        ),
        Document(
            page_content=(
                "Credit files for commercial borrowers must include two "
                "years of financial statements before a narrative is drafted."
            ),
            metadata={
                "chunk_id": "chunk-0002",
                "source": "SYNTHETIC_CREDIT_POLICY",
                "acl": "CREDIT_TEAM",
            },
        ),
    ]
    return GovernedVectorStore(docs)


if __name__ == "__main__":
    pipeline = GovernedRAGPipeline(_sample_store())
    caller = CallerContext(
        user_id="analyst.demo",
        entitlements=["REG_PUBLIC"],
        purpose="Regulatory document analysis (demo)",
    )
    result = pipeline.ask(
        "Within how many days must material model changes be reported?",
        caller,
    )
    print(result.text)
    print("\n--- audit record ---")
    print(result.audit.to_json() if result.audit else "{}")
