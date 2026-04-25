"""Dataset adapters: original labels retained and mappings never inferred implicitly."""
from __future__ import annotations
from hallubench.schemas import Example, Span

HALU_FIELDS = {
    "qa": ("knowledge", "question", "right_answer", "hallucinated_answer"),
    "dialogue": ("knowledge", "dialogue_history", "right_response", "hallucinated_response"),
    "summarization": ("document", None, "right_summary", "hallucinated_summary"),
}
RAG_SPAN_MAP = {"evident conflict": "contradicted", "subtle conflict": "contradicted", "baseless info": "unsupported", "baseless information": "unsupported", "evident baseless info": "unsupported", "subtle baseless info": "unsupported"}


def halu_examples(raw: dict, index: int, task: str, revision: str):
    if task not in HALU_FIELDS:
        raise ValueError(f"Unsupported HaluEval task config {task}")
    evidence_key, question_key, right_key, hall_key = HALU_FIELDS[task]
    required = [evidence_key, right_key, hall_key] + ([question_key] if question_key else [])
    for key in required:
        if key not in raw or not isinstance(raw[key], str) or not raw[key].strip():
            raise ValueError(f"Missing upstream HaluEval field {key} at row {index}")
    group = f"halu:{task}:{index}"
    question = raw[question_key] if question_key else "Summarize the evidence."
    base = dict(question=question, evidence=raw[evidence_key], reference_answer=raw[right_key],
                dataset=f"HaluEval/{task}", revision=revision, task=task, group_id=group,
                source_id=group, generator="HaluEval upstream generations (specific model attribution unavailable)",
                provenance="upstream curated QA/dialogue/summarization pairs", original=dict(raw))
    # 'hallucinated' is a COARSE construction label: mapped to unsupported, never to proven false.
    # 'right' is a reference-designation, NOT independent open-world fact checking.
    return [Example(example_id=f"{group}:right", answer=raw[right_key], original_label="right",
                    label="supported", **base),
            Example(example_id=f"{group}:hallucinated", answer=raw[hall_key],
                    original_label="hallucinated", label="unsupported", **base)]


def ragtruth_examples(raw: dict, source: dict, revision: str) -> list[Example]:
    """RAGTruth canonical JSONL join. Requires explicit task/prompt/source keys."""
    for k in ("id", "source_id", "response", "labels", "split"):
        if k not in raw:
            raise ValueError(f"RAGTruth missing {k}")
    if str(raw["source_id"]) != str(source.get("source_id", source.get("id"))):
        raise ValueError("RAGTruth source_id does not match source record")
    if raw["split"] not in ("train", "test"):
        raise ValueError("Unrecognized upstream RAGTruth split")
    answer = raw["response"]
    if not isinstance(answer, str) or not answer.strip() or not isinstance(raw["labels"], list):
        raise ValueError("Malformed RAGTruth response or labels")
    annotation_spans = []
    for label in raw["labels"]:
        if not isinstance(label, dict) or "label_type" not in label:
            raise ValueError("Malformed RAGTruth span")
        original = str(label["label_type"])
        normalized = RAG_SPAN_MAP.get(original.lower().strip())
        if normalized is None:
            raise ValueError(f"Unmapped RAGTruth annotation label: {original}")
        start, end = int(label["start"]), int(label["end"])
        meta = {k: v for k, v in label.items() if k not in ("start", "end", "text", "label_type")}
        if label.get("implicit_true") is True:
            # In the upstream data, this marks a true claim unmentioned in evidence.
            # For evidence grounding, it remains unsupported, not contradicted.
            normalized = "unsupported"
        span = Span(start=start, end=end, text=label["text"], label=normalized,
                    original_label=original, meta=meta)
        span.validate(answer)
        annotation_spans.append(span)
    if any(s.label == "contradicted" for s in annotation_spans):
        label = "contradicted"  # multi-error response: precedence only, other spans preserved
    elif annotation_spans:
        label = "unsupported"
    else:
        label = "supported"  # no *annotated* hallucinated spans, not universal truth
    split = raw["split"]
    task = str(raw.get("task_type") or source.get("task_type") or "rag").lower()
    evidence = source.get("source_info")  # source describes provenance e.g. MARCO, NOT evidence
    if not isinstance(evidence, str):
        # explicit nested source payload preserved in JSON; never silently render class labels
        import json
        evidence = json.dumps(evidence, ensure_ascii=False) if evidence is not None else None
    prompt = source.get("prompt") or raw.get("prompt") or "Use the provided evidence to answer the task."
    example = Example(example_id=f"ragtruth:{raw['id']}", question=str(prompt), evidence=evidence,
        reference_answer=None, answer=answer, original_label="|".join(sorted({s.original_label for s in annotation_spans})) or "none",
        label=label, dataset="RAGTruth", revision=revision, task=task,
        group_id=f"ragtruth:source:{raw['source_id']}", source_id=str(raw["source_id"]),
        split=split, generator=raw.get("model"), annotations=annotation_spans,
        provenance="RAGTruth upstream human span annotations", original={"response": raw, "source": source})
    example.validate()
    return [example]


def explicit_mapping_rows(rows, mapping: dict, dataset_name: str, revision: str):
    """Generic local or HuggingFace records require a fully explicit column/label mapping."""
    fields = mapping.get("columns", {})
    labels = mapping.get("labels", {})
    required = ("id", "answer", "label", "question", "evidence", "task", "group_id")
    if any(k not in fields for k in required) or not labels:
        raise ValueError("Explicit mapping must define id, answer, label, question, evidence, task, group_id and labels")
    for raw in rows:
        def get(k, default=None):
            key = fields.get(k)
            return raw.get(key, default) if key else default
        original_label = str(get("label"))
        if original_label not in labels:
            raise ValueError(f"Unknown original label: {original_label}")
        item = Example(example_id=f"{dataset_name}:{get('id')}", question=str(get("question") or ""),
            evidence=get("evidence"), reference_answer=get("reference_answer"), answer=get("answer"),
            original_label=original_label, label=labels[original_label], dataset=dataset_name,
            revision=revision, task=str(get("task") or "unknown"),
            group_id=str(get("group_id") or ""), split=str(get("split") or "unspecified"),
            source_id=str(get("source_id")) if get("source_id") is not None else None,
            generator=get("generator"), provenance=str(get("provenance") or "user-supplied"), original=raw)
        item.validate()
        yield item
