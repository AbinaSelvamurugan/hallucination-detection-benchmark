"""Optional remote judge; JSON-schema enforcement with neutral abstention."""
import json
from hallubench.detectors.base import Detector
from hallubench.schemas import LABELS

RUBRIC = """You are checking grounding in PROVIDED evidence, NOT open-world truth.
Return JSON only: {"label": one of supported, contradicted, unsupported, partially_supported,
insufficient_evidence, "flagged_claims": [{"text": str, "label": str,
"evidence_excerpt": str}], "reasoning": str, "abstained": bool}.
'Supported': evidence entails all material claims. 'Contradicted': any material claim conflicts.
'Unsupported': evidence does not establish a factual claim; this DOES NOT mean false.
'Partially supported': some claims supported and others unsupported/contradicted.
'Insufficient evidence': cannot reliably decide using the available information.
Never invent supporting quotes. Abstain when the evidence is inadequate for a conclusion.
"""

class LLMJudge(Detector):
    name = "llm_judge"
    requires_evidence = True

    def __init__(self, model: str, base_url: str | None = None, api_key=None):
        if not model:
            raise ValueError("Explicit judge model required")
        self.model_id = model
        self.base_url = base_url
        self.api_key = api_key
        self._client = None
        self.version = model

    def _predict(self, ex):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=30, max_retries=1)
        payload = {"question": ex.question, "evidence": ex.evidence or ex.reference_answer, "answer": ex.answer}
        response = self._client.chat.completions.create(model=self.model_id, temperature=0,
            response_format={"type": "json_object"}, messages=[{"role": "system", "content": RUBRIC},
                                                   {"role": "user", "content": json.dumps(payload)}])
        parsed = json.loads(response.choices[0].message.content)
        if not isinstance(parsed, dict) or not all(x in parsed for x in ("label", "flagged_claims", "reasoning", "abstained")):
            raise ValueError("Judge response missing required rubric fields")
        if parsed["label"] not in LABELS or not isinstance(parsed["abstained"], bool):
            raise ValueError("Judge response has invalid label or abstain flag")
        if not isinstance(parsed["reasoning"], str) or not isinstance(parsed["flagged_claims"], list):
            raise ValueError("Judge response malformed")
        for claim in parsed["flagged_claims"]:
            if not isinstance(claim, dict) or not isinstance(claim.get("text"), str) or claim.get("label") not in LABELS or not isinstance(claim.get("evidence_excerpt"), str):
                raise ValueError("Invalid judge flagged claim")
            if claim["evidence_excerpt"] and claim["evidence_excerpt"] not in (ex.evidence or ex.reference_answer):
                raise ValueError("Judge supplied a non-verbatim evidence excerpt")
        meta = {"model": response.model, "provider": self.base_url or "default OpenAI-compatible",
                "reasoning": parsed["reasoning"], "flagged_claims": parsed["flagged_claims"],
                "usage": (response.usage.model_dump() if response.usage else None), "rubric": RUBRIC}
        return (None if parsed["abstained"] else parsed["label"], {}, "none", meta)

    def close(self):
        if self._client:
            self._client.close()
            self._client = None
