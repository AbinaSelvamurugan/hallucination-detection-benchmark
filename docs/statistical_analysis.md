# Statistical analysis pre-specification

The benchmark implements **paired source-group bootstrap** with a fixed RNG seed: sample source groups with replacement, taking all their responses, recompute positive-class F1 difference, and use the 2.5/97.5 percentiles as an exploratory 95% uncertainty interval. The sampling unit is the **original source/parent group**, not individual claims or responses. Pairwise analysis uses the intersection of valid predictions for both methods.

Limitations: very few source groups, highly imbalanced classes and shared model-generation templates can make percentile intervals unreliable. Comparisons are exploratory and uncorrected for multiple testing. Do NOT call a method universally superior or statistically significant based solely on an interval. If formal NHST is required, preregister a paired, cluster-aware permutation procedure and adjust for multiple comparisons.

Subgroup metrics by source, task and generator are recorded without pooling across non-comparable evidence coverage. ROC-AUC/PR-AUC require comparable probabilities with both classes; reliability/Brier/ECE are diagnostics for model probabilities, not claims of calibration.

For claim-level spans, run paired bootstrap over `parent_id` / `source_id`, never count multiple claims as independent draws. The current claim span CLI reports aggregate matching scores only; clustered span uncertainty is not implemented.
