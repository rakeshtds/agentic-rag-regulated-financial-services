# Agentic RAG for Regulated Financial Services

**Retrieval-augmented generation that a risk committee can approve: agentic RAG over regulatory documents, credit-risk narratives and support knowledge — with governance controls embedded, not bolted on.**

## Overview

This project implements agentic RAG solutions for three regulated-industry use cases: regulatory document analysis, credit-risk narrative generation, and customer-support triage. Agents built with LangChain and LlamaIndex retrieve from governed document stores, reason over what they retrieve, and produce outputs with citations back to source documents. The platform side — LLMOps/MLOps on Amazon SageMaker and Amazon Bedrock — handles model routing, evaluation, guardrails and audit logging, and the controls expected in banking (OSFI, PIPEDA, SOX) are embedded in the pipeline: access-scoped retrieval, PII handling, full prompt/response audit trails and human review for high-impact outputs.

## Problem

Generic RAG demos fail in regulated financial services for reasons that have nothing to do with model quality:

- **Access control** — a retriever that ignores document-level permissions will happily surface another team's restricted filings.
- **Auditability** — under OSFI/PIPEDA/SOX expectations, "the model said so" is not an answer. Every output needs the prompt, the retrieved sources and the model version on record.
- **PII** — support tickets and credit files contain personal information that must be detected and handled before it reaches a model or a log.
- **Drift** — without evaluation harnesses and monitored retrieval quality, answer quality decays silently as document corpora change.

The use cases were valuable — analysts, credit officers and support teams all drown in documents — but only a governed version of RAG could be deployed.

## Architecture

```
 Documents (regulatory filings, credit files, support KB)
        │  ingest: chunk, classify, PII scan, permission-tag
        ▼
 Governed vector index  ◄── document ACLs + retention tags
        │
        ▼
 Agent layer (LangChain / LlamaIndex)
   ├─ Regulatory analysis agent  — retrieve, compare, cite
   ├─ Credit narrative agent     — draft narratives from file evidence
   └─ Support triage agent      — classify, route, draft response
        │
        ▼
 Bedrock model routing + guardrails ──► response with citations
        │
        ▼
 Audit log (prompt, retrieved chunks, model, output) ──► human review
                                                       for high-impact work

 LLMOps/MLOps (SageMaker): evaluation harness, prompt/version registry,
 retrieval-quality monitoring, cost and latency dashboards
```

- **Governed retrieval** — every chunk carries its source document's access-control list and retention class; retrieval filters by the caller's entitlements before ranking.
- **Agents, not chains** — each use case is an agent that can re-retrieve, compare sources and abstain when evidence is insufficient, rather than forcing an answer.
- **Citations by construction** — outputs must cite retrieved chunks; uncited claims are rejected by a post-generation check.
- **Controls embedded** — PII scanning at ingest and at output, immutable audit logging of every request/response pair, and a human-review queue for credit narratives.

## Tech Stack

- LangChain, LlamaIndex
- Amazon Bedrock (model access and guardrails), Amazon SageMaker (LLMOps/MLOps)
- Python 3.11+
- Regulatory context: OSFI, PIPEDA, SOX control expectations

## Key Components

| Component | Description |
|---|---|
| `src/rag_pipeline.py` | The governed RAG pipeline: entitlement-filtered retrieval, citation-enforced generation, PII output scan, audit record emission. |
| `src/triage_agent.py` | Customer-support triage agent: classifies an incoming ticket, retrieves the relevant knowledge, drafts a response, routes by category/urgency. |
| `infra/notes.md` | Platform notes: Bedrock/SageMaker LLMOps setup, evaluation harness, guardrails and audit design. |
| `requirements.txt` | Python dependencies. |

## How It Works

1. **Ingest** — documents are chunked and classified; each chunk inherits its source's ACL and retention tags and is scanned for PII before indexing.
2. **Retrieve** — a query is embedded and matched against the index *after* entitlement filtering: the retriever literally cannot see chunks the caller is not entitled to.
3. **Reason** — the agent decides whether the retrieved evidence answers the question, retrieves more (or from a different collection) if not, and abstains if evidence stays insufficient.
4. **Generate with citations** — the response must reference chunk IDs; a verifier drops uncited factual claims.
5. **Control** — the output passes a PII scan, the full interaction (prompt, chunks, model version, output) is written to the audit log, and high-impact outputs (credit narratives) wait for human approval.
6. **Operate** — SageMaker-hosted evaluation jobs score sampled traffic for groundedness and retrieval quality; regressions alert before users notice.

## Outcomes

- Three document-heavy workflows — regulatory analysis, credit-risk narratives and support triage — served from one governed RAG platform instead of three one-off pilots.
- Audit-ready by default: any generated answer can be traced to its prompt, its sources and its model version.
- The control set (entitlement-filtered retrieval, PII handling, citation enforcement, human review) became the reusable pattern for subsequent GenAI use cases in the same regulated environment.

## Repo Structure

```
agentic-rag-regulated-financial-services/
├── README.md
├── src/
│   ├── rag_pipeline.py     # governed retrieval + cited generation
│   └── triage_agent.py     # support triage agent
├── infra/
│   └── notes.md            # Bedrock/SageMaker LLMOps & controls notes
└── requirements.txt
```

## Note

Portfolio project based on production work in regulated financial services; code is a clean-room representative implementation with sample/synthetic metadata only.
