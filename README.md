# ECHO: Belief-Justification Decoupling in Iterative Self-Critique

This repository contains the analysis code and derived results accompanying the manuscript:

**"Belief-Justification Decoupling in Iterative Self-Critique Without Successful Retrieval: An Empirical Characterization Using the ECHO Index"**
Samuel Stephen, R. Vignesh — Karunya Institute of Technology and Sciences

## What this is

A secondary analysis of the publicly released [ForesightFlow](#data-source) multi-agent
coordination-traces dataset. No models were trained, fine-tuned, or queried as part of this
study; all analysis is performed directly on the released, frozen reasoning traces.

## Repository structure

```
src/
  extraction.py     Evidence-term extraction (stoplist-filtered regex procedure, Section 3.2)
                     and stated-probability extraction from response text.
  echo_metric.py     The ECHO index (Section 3.3) and the two related evidence-utilization
                     measures used for sequential_pipeline and orchestrator_specialist
                     (Section 3.1, Table 2).
  analysis.py        All statistical analyses reported in Section 4: the primary
                     evidence-change x belief-change association (Table 4, 8), the
                     non-circular component ablation (Table 3), threshold sensitivity
                     (Table 9), epsilon sensitivity (Table 10), the permutation test
                     (Table 11), the outcome-conditioned comparison with effect size and
                     bootstrap CI (Table 12), and the continuous probability-extremity
                     model (Table 13).
  figures.py         Generation code for Figures 1-5.

results/
  bjri_raw_CONFIRMED.npy      Per-trace ECHO values, peer_critique_debate (n=98)
  retention_raw_CONFIRMED.npy Per-trace evidence-retention values, sequential_pipeline (n=98)
  incorp_raw_FINAL.npy        Per-trace evidence-incorporation values, orchestrator_specialist (n=98)
  summary_statistics.md       Plain-text summary of all headline numbers reported in the paper
```

## Data source

This repository does **not** redistribute the ForesightFlow dataset. The original dataset
is released under a CC-BY-4.0 licence at its original source (see manuscript Section 2.1
for the citation and access details). To reproduce the analysis end to end, download the
three trace files used in this study (`peer_critique_debate.jsonl`, `sequential_pipeline.jsonl`,
`orchestrator_specialist.jsonl`) from that source and place them in a local `data/` directory
before running the scripts in `src/`.

## Reproducing the results

```bash
pip install -r requirements.txt
python src/analysis.py --data-dir ./data --out-dir ./results
python src/figures.py --data-dir ./data --out-dir ./figures
```

## Scope note

All findings in the accompanying manuscript are conditioned on the retrieval-scope
condition documented in Section 2.3: no `searchWeb` call in the underlying dataset
returned a successful result. This is a property of the released corpus, not of the
analysis code in this repository.

## Citation

If you use this code, please cite the accompanying manuscript (full citation to be added
upon publication) and the original ForesightFlow dataset release.

## License

See `LICENSE`.
