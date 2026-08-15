# Troubleshooting

**`ModuleNotFoundError: datasets`**: Install HF extra (`python -m pip install -e '.[hf]'`) with internet access. Core tests do not require it.

**`No test examples`**: `prepare` must assign proper held-out split. RAGTruth upstream `test` is preserved. Don't silently relabel data to make a metric appear.

**`Quality check failed`**: Read `*.quality.json`; duplicates, cross-split questions and invalid offsets are hard failures. Correct source grouping or curate duplicates, then re-export. Do not just suppress validation.

**`Out of memory`, `model unavailable`**: Use `--max-examples` for an explicit smoke subset, CPU or smaller local model, and `local_files_only: true` once cached. Never treat model errors as classification outcomes.

**`Missing model version`**: Set a full immutable Hub revision and verify architecture/label mapping from model config. Save usage version before running final test.

**`API failure`**: Check `openai` package, credentials, model ID, JSON-mode support and privacy terms. Results record failed statuses. No judge credential required for other detectors.

**`No claim spans`**: RAGTruth's official `response.jsonl` has labelled spans; HaluEval QA pairs do not offer the same span ground truth. Test `hallubench claims` only with RAGTruth prepared dataset and interpret as sentence-span proxy.

**Dashboard shows no run**: Point the sidebar to an existing result directory with `summary.json`. In the dashboard environment install `pip install -e '.[dashboard]'` first.
