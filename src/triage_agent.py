"""Customer-support triage agent.

Pipeline for one incoming support ticket:

  1. Classify — category and urgency from the ticket text (LLM, structured
     output via pydantic).
  2. Retrieve — search the support knowledge base with the governed RAG
     pipeline (entitlement-filtered, citation-enforced).
  3. Draft    — a response grounded in the retrieved knowledge, with
     citations; never sent automatically.
  4. Route    — assign queue and priority from category/urgency; anything
     touching account changes, disputes or vulnerability is escalated to a
     human queue regardless of draft quality.

The agent drafts and routes; humans send. That boundary is deliberate in a
regulated support operation.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from rag_pipeline import CallerContext, GovernedRAGPipeline

# ---------------------------------------------------------------------------
# Classification model
# ---------------------------------------------------------------------------


class Category(str, Enum):
    ACCOUNT_ACCESS = "account_access"
    PAYMENTS = "payments"
    CARDS = "cards"
    STATEMENTS = "statements"
    DISPUTE = "dispute"
    OTHER = "other"


class Urgency(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class Classification(BaseModel):
    category: Category
    urgency: Urgency
    rationale: str = Field(description="One sentence explaining the class")


# Categories that always go to a human specialist queue.
ESCALATE_CATEGORIES = {Category.DISPUTE}

# Queue routing by category.
QUEUE_BY_CATEGORY = {
    Category.ACCOUNT_ACCESS: "support.access",
    Category.PAYMENTS: "support.payments",
    Category.CARDS: "support.cards",
    Category.STATEMENTS: "support.statements",
    Category.DISPUTE: "support.disputes.specialist",
    Category.OTHER: "support.general",
}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class TriageAgent:
    """Classifies, retrieves, drafts and routes — never sends."""

    def __init__(self, pipeline: GovernedRAGPipeline) -> None:
        self.pipeline = pipeline

    def classify(self, ticket_text: str) -> Classification:
        """LLM classification with structured output.

        Representative implementation: in production the Bedrock model is
        bound to the Classification schema (structured output). Here a
        deterministic keyword pass keeps the demo runnable offline.
        """
        text = ticket_text.lower()
        if "dispute" in text or "chargeback" in text:
            category = Category.DISPUTE
        elif "card" in text:
            category = Category.CARDS
        elif "payment" in text or "transfer" in text:
            category = Category.PAYMENTS
        elif "login" in text or "password" in text or "locked out" in text:
            category = Category.ACCOUNT_ACCESS
        elif "statement" in text:
            category = Category.STATEMENTS
        else:
            category = Category.OTHER

        urgency = (
            Urgency.HIGH
            if any(w in text for w in ("urgent", "fraud", "stolen", "today"))
            else Urgency.NORMAL
        )
        return Classification(
            category=category,
            urgency=urgency,
            rationale=f"Keyword classification (demo) → {category.value}.",
        )

    def triage(self, ticket_id: str, ticket_text: str) -> dict:
        classification = self.classify(ticket_text)

        caller = CallerContext(
            user_id="triage.agent",
            entitlements=["SUPPORT_KB"],
            purpose=f"Support triage for ticket {ticket_id}",
        )
        answer = self.pipeline.ask(ticket_text, caller)

        escalate = (
            classification.category in ESCALATE_CATEGORIES
            or classification.urgency is Urgency.HIGH
        )
        return {
            "ticket_id": ticket_id,
            "category": classification.category.value,
            "urgency": classification.urgency.value,
            "queue": QUEUE_BY_CATEGORY[classification.category],
            "escalate_to_human": escalate,
            "draft_response": None if answer.abstained else answer.text,
            "citations": answer.citations,
            "draft_status": (
                "NO_DRAFT_INSUFFICIENT_EVIDENCE"
                if answer.abstained
                else "DRAFT_PENDING_HUMAN_REVIEW"
            ),
        }


if __name__ == "__main__":
    from rag_pipeline import _sample_store

    agent = TriageAgent(GovernedRAGPipeline(_sample_store()))
    result = agent.triage(
        "TCK-1001",
        "My card was stolen today and I need a replacement urgently.",
    )
    for key, value in result.items():
        print(f"{key}: {value}")
