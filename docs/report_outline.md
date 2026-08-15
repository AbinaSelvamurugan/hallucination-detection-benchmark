# Research report structure

1. **Abstract:** objective, datasets, exact experimental eligibility, measured uncertainty and limitations; no imaginary improvement claims.
2. **Problem and hypotheses:** open-world truth vs evidence-grounded factuality, partial support, abstention, claim boundaries, cross-generator transfer.
3. **Literature review:** HaluEval (Li et al. 2023); RAGTruth (Wu et al. 2024); NLI and semantic similarity, LLM judge validation.
4. **Ethics and data governance:** license and underlying source material, restricted text, annotation disagreement, privacy.
5. **Methods:** ontology, explicit upstream mapping, source-group split, development/test separation, detector configurations and computational hardware.
6. **Experiments:** independent QA/dialogue/summarization/RAG evaluations, source- and generator-held-out tests, common-subset comparisons.
7. **Metrics and statistics:** positive-class recall and precision, MCC, coverage, probability calibration, paired grouped bootstrap and span IoU.
8. **Results:** populate ONLY from saved `results/*/summary.json` and `findings.md` when a real held-out experiment succeeds.
9. **Qualitative errors:** review and adjudicate real false alarms and missed hallucinations, negation/entity/number/partial support.
10. **Discussion and threats to validity:** evidence incompleteness, long context, label noise, model/domain shift, imperfect spans, underlying source licenses.
11. **Conclusion and future work:** empirically supported answers to the ten research questions, exact reproducibility artifacts.
12. **Appendices:** commands, dependency/environment snapshot, source revisions, annotation codebook, model prompt, claim span match protocol, grouped resampling pseudocode.

If no benchmark run is performed, leave the results section empty rather than filling it with estimates.
