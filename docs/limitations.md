# Known limits and non-claims

- Evidence support and objective open-world factuality are different research constructs. HaluEval's synthetic/hallucinated pairing does not imply every hallucinated response is *logically contradicted*.
- No human adjudication has been performed; `partially_supported` is available in the schema and via user-supplied annotations, but is not a HaluEval/RAGTruth five-way gold label.
- Coarse RAGTruth response label uses contradiction priority if mixed. It does not preserve complete response composition in one categorical label (spans do preserve that detail).
- The full RAGTruth files have not been run through the current adapter; unknown source schemas or span strings fail loudly and require adapter revision plus new tests.
- Large corpora (>3,000) receive exact cross-split answer/question checks but not exhaustive near-duplicate char-NN comparison. Template- or paraphrase-based leakage remains a risk.
- Claim segmentation is sentence-oriented, and the span-IoU score is an approximate localization task rather than atomic proposition extraction.
- A per-answer NLI call may truncate references; without retrieval/chunk aggregation it does not robustly entail answers grounded in long passages.
- The per-example `original` fields can include sensitive, copyrighted or restricted documents. Do not publish raw prepared JSONL or examples in a public dashboard without access review.
- Streaming works at ingestion, but normalization/export materialize records and near-duplicate checking uses non-streaming methods. Large-corpus memory/time performance has not been profiled.
- Multi-GPU processing, API bounded-concurrency orchestration, rate-limit backoff, and exact cost estimation are not implemented. Current remote judge requests are sequential.
- Error categories are regex candidates, not adjudicated causal explanations. Inter-rater reliability requires collected annotations.
- The HTML report is readable but intentionally simple; the Streamlit dashboard is optional.
- The `requirements.txt` core install is supported by pyproject; optional dependency compatibility and model downloading are not covered by the offline tests.

Research claims must be limited to upstream labels, validated split validity, evaluated models and clearly stated hardware, evidence coverage and sample selection.
