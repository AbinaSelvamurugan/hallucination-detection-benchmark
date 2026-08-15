# Dataset selection, provenance and use restrictions

**Sources checked 2026-08-15.** Schema details are taken from the published dataset cards and repository files.

| Corpus | Public repository / release | Schema and upstream splits | Rights / attribution | Local intake |
|---|---|---|---|---|
| HaluEval (HF mirror) | https://huggingface.co/datasets/pminervini/HaluEval, revision `72a55d2291cf5f69581e22ac6491b251fc618291` | `qa`, `dialogue`, `summarization`, each split named `data`, paired `right_*`/`hallucinated_*`; original `general` and `*_samples` are not silently used | HF mirror card lists **Apache-2.0**; original RUCAIBox/HaluEval GitHub lists **MIT**. Licence scopes may differ, including source training materials | Explicit pair adapter (`right_*`/`hallucinated_*`) |
| RAGTruth official | https://github.com/ParticleMedia/RAGTruth, download `dataset/response.jsonl` and `dataset/source_info.jsonl` at pinned git commit | `response`: id, source_id, model, labels [{start,end,text,label_type,meta}], split train/test; `source_info`: source_id, task_type, source_info (text or object), prompt, source | GitHub repository `LICENSE` MIT; underlying CNN/DM, news, MS MARCO and Yelp content may carry separate licenses or restrictions. Cite authors and dataset; avoid third-party text redistribution | Official JSONL adapter joining `response` and `source_info` on `source_id` |
| Optional third-party HF RAGTruth mirror | https://huggingface.co/datasets/leobianco/ragtruth | Hub card lists train/test and typed span/metadata fields but includes **only a subset** of official responses; not interchangeable | Hub card says Apache-2.0; check originals | Optional; the official release is preferred |

Primary sources: HaluEval paper [Li et al., 2023](https://arxiv.org/abs/2305.11747) and [official code](https://github.com/RUCAIBox/HaluEval); [HF dataset card](https://huggingface.co/datasets/pminervini/HaluEval/blob/main/README.md). RAGTruth [Wu et al., 2024](https://arxiv.org/abs/2401.00396), [official README](https://github.com/ParticleMedia/RAGTruth/blob/main/README.md) and [license](https://github.com/ParticleMedia/RAGTruth/blob/main/LICENSE).

## Lineage procedure

1. Fetch exact upstream ref, never `main` as an unrecorded experiment dependency. Record Hub repo/config/split/revision or Git commit and raw SHA-256.
2. Check upstream metadata, redistribution/citation terms and consent where applicable; keep raw source passages out of public result exports if restricted.
3. Normalize only using the named per-corpus adapter; retain upstream raw fields in `original` (the normalized private file may contain licensed text). Preserve sample ID/source ID, task and generator.
4. Validate offsets by exact character substring comparison, reject unrecognized labels and empty records; investigate reported duplicates and source-group overlaps.
5. Save seed/grouping/split rules and SHA-256 of prepared JSONL. Each run stores `prepared_sha256` and exact selected test-ID checksum.

## Dataset-specific label caveats

HaluEval task examples are **paired** reference and machine-constructed hallucinated responses, NOT individually adjudicated five-class claims. Our *operational* `right -> supported`, `hallucinated -> unsupported` is a coarse proxy for the dataset's binary evaluation. It does NOT establish that every constructed answer is an unsupported claim, rather than contradicted, mixed or true. Report HaluEval binary results as *agreement with HaluEval's construction label*. A validated five-class evaluation requires new human annotation, which is outside the scope of this project.

RAGTruth spans are labelled `Evident/Subtle Conflict` -> contradicted and `Evident/Subtle Baseless Info` -> unsupported. `implicit_true` signals true information not provided in the reference and remains *unsupported relative to supplied evidence*; never classify as false. A response containing both kinds of errors uses **contradiction priority** as its coarse response label, but all span labels are kept and must be assessed separately. No annotated hallucination spans is provisionally marked `supported relative to this annotation`, not universally correct. `quality` flags stay in original metadata, and should be filtered/analyzed before a publication-quality study.

The official RAGTruth `source_info` field can be a dict. The adapter serializes the dict into JSON evidence without silently dropping passages. HalluEval summarization uses a common task prompt and source-specific group IDs so task instructions are not mistaken for duplicate source documents.
