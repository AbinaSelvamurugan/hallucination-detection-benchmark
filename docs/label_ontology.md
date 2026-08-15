# Evidence-relative label ontology and annotation protocol

| Label | Rule | Allowed truth claim |
|---|---|---|
| `supported` | Supplied evidence entails the material claim(s) | **Evidence-supported only**, not universal truth |
| `contradicted` | Evidence entails an incompatible proposition | Incorrect *relative to this evidence* |
| `unsupported` | Evidence does not support the claim, but does not establish its negation | No inference of falsehood |
| `partially_supported` | At least one material atomic proposition supported, another contradicted or unsupported | Requires decomposition and claim-level review |
| `insufficient_evidence` | No reliable determination possible (missing, contradictory, irrelevant or unintelligible reference material) | Must not be coerced to true/false |

**Positive class for coarse binary study:** any `contradicted`, `unsupported`, or `partially_supported`; `supported` is negative; `insufficient_evidence` excluded from labelled denominator and reported. “Hallucination” in this coarse task includes evidence unsupported assertions, not necessarily demonstrably false ones. Pre-existing HaluEval `right`/`hallucinated` construction labels require separate interpretation; do not claim a curated five-class dataset.

**Annotation protocol for new human adjudication (not yet performed):** Two annotators independently read the prompt, evidence and whole response. Mark minimal factual claim spans with exact char offsets. For each, record evidence excerpt location; mark support, contradiction, unsupported or insufficient with rationale. Decompose compound assertions before using `partially_supported`, and distinguish missing evidence from contradictions. Disagreements retain both raw labels; third blind adjudicator resolves or marks insufficient; compute Cohen's kappa or Krippendorff alpha only when multiple annotations exist. Preserve reviewer IDs (pseudonyms), confidence, instructions version, timestamps and source restrictions. No automatic LLM judgement is treated as human ground truth.

**Boundary caveat:** RAGTruth's labelled spans are annotated error fragments, often smaller than sentences. The deterministic segmenter measures sentence-level error span location and cannot establish full atomic claim extraction quality.
