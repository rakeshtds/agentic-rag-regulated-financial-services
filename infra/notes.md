# Platform Notes — LLMOps/MLOps on SageMaker & Bedrock

How the agentic RAG platform is operated. Notes describe design and setup
decisions; they contain no account IDs, endpoints or credentials.

## Model access — Amazon Bedrock

- All foundation-model calls go through **Amazon Bedrock**; no model keys
  exist in application code or config.
- Model routing is by use case, not by habit:
  - high-reasoning tasks (regulatory comparison, credit narratives) route to
    the largest enabled model;
  - high-volume tasks (triage classification) route to a smaller, cheaper
    model with structured output.
- **Bedrock Guardrails** are attached at the API layer: content filters,
  denied topics (e.g. personalised financial advice in support drafts) and
  PII redaction rules act as a second line behind the pipeline's own scan.
- Every call logs model ID and version so any historical answer can be
  reproduced — model changes are treated like code changes.

## LLMOps/MLOps — Amazon SageMaker

- **Evaluation harness** (scheduled SageMaker jobs):
  - a frozen golden set per use case (question, entitled sources, expected
    answer characteristics) runs on every prompt or model change;
  - sampled production traffic is scored for *groundedness* (are claims
    supported by retrieved chunks?) and *retrieval recall* (did the entitled
    evidence reach the context?);
  - regressions beyond threshold block promotion of the new prompt/model.
- **Prompt & version registry** — prompts are versioned artefacts, deployed
  like code with review and rollback; the audit log records which prompt
  version produced each answer.
- **Monitoring** — dashboards for latency, cost per request, abstention rate
  and citation-failure rate. A rising abstention rate usually means corpus
  drift or an ACL change, and is paged as a data problem, not ignored.
- **Retrieval quality** — index rebuilds are validated offline against the
  golden set before swapping; chunking or embedding-model changes require
  the same evaluation gate as model changes.

## Controls (OSFI / PIPEDA / SOX alignment)

| Control expectation | Where it is implemented |
|---|---|
| Access control follows the user | Entitlement filter pushed into retrieval (see `src/rag_pipeline.py`) |
| PII minimisation | PII scan at ingest and at output; guardrail redaction as backstop |
| Auditability | Immutable audit record per request: prompt version, chunk IDs, model, output |
| Human accountability | Credit narratives and escalations require human approval before use |
| Change management | Prompts, models and indexes promoted through the evaluation gate only |
| Retention | Audit logs and document classes carry retention tags enforced at ingest |

## Environments

- Dev / Test / Prod are separate AWS accounts; the golden evaluation set
  must pass in Test before Prod promotion.
- Sample/synthetic corpora only in Dev; production document stores are
  mounted read-only and never copied into lower environments.
