# Contributing

Contributions must pass `pytest -q`, retain original labels and evidence, include mapping/schema test fixtures for any new dataset, and record pinned source revisions. Do not commit raw public third-party data, provider secrets, unreviewed annotation claims or fabricated evaluation scores. Explain label semantics and licensing before adding a new corpus. Security: use trusted model/checkpoint files; do not ingest untrusted pickle/joblib checkpoints.
