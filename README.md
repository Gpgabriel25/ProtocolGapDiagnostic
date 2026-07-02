# Protocol-Gap Diagnostic

> Code and data for the paper **"No Free Swap: Protocol-Dependent Layer Redundancy in Transformers"**

This repository contains exactly the code and data that produce the numbers and figures in the paper. Nothing more, nothing less.


- Paper PDF: [`paper/main.pdf`](paper/main.pdf)
- LaTeX source bundle: [`paper/arxiv_submission.zip`](paper/arxiv_submission.zip)
- arXiv metadata (title, abstract, ACM class codes): [`paper/arxiv_metadata.md`](paper/arxiv_metadata.md)

## What's in the paper, and where it lives in this repo

| Paper element | Producing script(s) | Data file |
|---|---|---|
| **Fig. 1 / Def. 1** — Swap protocol schematic | (TikZ in `paper/figures/swap_protocol.tex`) | — |
| **Fig. 2** — GPT-2-Medium 24×24 swap-KL heatmap | `bisimulation_experiment.py` → `generate_figures.py` | `reports/2026-04-02T09-57-25/v14_distance_matrix_100p.csv` |
| **Fig. 3** — Pythia 160M→6.9B protocol-gap trajectory | `tpu_pythia_protocol_gap_trajectory.py` → `analyze_protocol_gap_trajectory.py` | `reports/2026-04-18T21-51-24/protocol_gap_trajectory_4scale.json` |
| **Fig. 4** — Scaling across 13 models | `scaling_bisimulation.py` → `generate_figures.py` | `reports/2026-03-31T00-18-24/{gpt2-large,gpt2-xl,opt-350m}_bisimulation.json` |
| **Fig. 5** — BLOOM-560M checkpoint trajectory | `bloom_bisimulation.py` → `generate_bloom_comparison.py` | `reports/2026-04-02T09-57-25/bloom_checkpoint.json` |
| **Fig. 6** — GPT-2-Medium compression sweep | `compression_sweep.py` → `generate_figures.py` | `reports/2026-04-01T18-45-48/compression_sweep.json` |
| **Fig. 7** — GPT-2-Medium adjacent-pair profile | `bisimulation_experiment.py` → `generate_figures.py` | `reports/2026-03-30T15-15-07/sorted_pairs.csv` |
| **Fig. 8** — Qwen3-8B compute-quality frontier | `tpu_bootstrap_8b_ci.py` → `generate_figures.py` | `qwen3_8b_ci.json` + `reports/2026-04-12T12-10-20/fair_cost_final_summary.json` |
| **Tab. 1** — Domains (narrative) | — | — |
| **Tab. 2** — Protocol taxonomy (narrative) | — | — |
| **Tab. 3** — Top-10 swap-similar pairs GPT-2-M | `bisimulation_experiment.py` | `reports/2026-04-02T09-57-25/v14_distance_matrix_100p.{json,csv}` |
| **Tab. 4** — Asymmetry sensitivity (Qwen3-8B) | `d_repl_asymmetry.py` | `kaggle/output_v32/qwen3_8b_predictor_validity.json` → `reports/2026-04-22T15-00-00/d_repl_asymmetry.json` |
| **Tab. 5** — Skip-layer PPL GPT-2-M | `skip_layer_test.py` | `reports/2026-03-30T16-10-52/skip_layer_results.json` |
| **Tab. 6** — Skip-layer PPL Llama-3.1-8B | `matched_eval_llama.py` | `llama_8b_ci.json` |
| **Tab. 7** — Comparison-contract summary (narrative) | — | — |
| **Tab. 8** — 8B benchmark under matched evaluator | `matched_eval_qwen3.py` + `matched_eval_llama.py` + `matched_eval_mistral.py` | `qwen3_8b_ci.json` + `llama_8b_ci.json` (Mistral sub-block in paper uses interchange-guided layers as a proxy replacement row; rerun `matched_eval_mistral.py` to reproduce) |
| **Tab. 9** — Pythia-1.4B full baseline suite | `pythia_full_baselines.py` | `reports/2026-04-06T17-39-31/pythia_full_baselines.json` |
| **Tab. 10** — Harmonized cross-model | `run_harmonized_slice.sh` → `matched_eval_qwen3.py` + `tpu_pythia_matched_eval.py` | `reports/2026-04-18T18-21-54/harmonized/{qwen,pythia}/` |
| **Tab. 11** — Calibration-free head-to-head (Qwen3-8B) | `tpu_qwen3_sleb_calfree.py` + `qwen3_beam_search.py` + `clean_oracle_bootstrap_ci.py` + `tpu_qwen3_bootstrap_headtohead.py` | `reports/2026-04-20T10-32-11/sleb_calfree/qwen3_8b_sleb_calfree.json` + `reports/2026-04-18T21-51-24/qwen3_8b_beam_search.json` + `reports/2026-04-30T15-49-38/qwen3_clean_oracle_ci.json` |
| **Tab. 12** — Skip-layer Qwen3-8B | `matched_eval_qwen3.py` + `tpu_replacement_pruning.py` + `tpu_qwen_taylor.py` | `qwen3_8b_ci.json` |
| **Tab. 13** — Qwen3-8B compute-vs-quality frontier | `tpu_bootstrap_8b_ci.py` | `qwen3_8b_ci.json` + `reports/2026-04-12T12-10-20/fair_cost_final_summary.json` |
| **Tab. 14** — Selection algorithm (pseudocode) | — | — |
| **Tab. 15** — Extended baselines (GPT-2-M) | `laco_sleb_baselines.py` + `bi_vs_bisim_headtohead.py` + `taylor_importance_baselines.py` | `reports/2026-03-31T12-24-40/laco_sleb_baselines.json` + `reports/2026-03-31T00-18-24/bi_vs_bisim_ppl.json` |
| **Tab. 16** — Scaling table | `scaling_bisimulation.py` + `pythia_bisimulation.py` + `bloom_bisimulation.py` + `bloom_1b1_bisimulation.py` + `tpu_llama_bisimulation.py` + `qwen3_8b_tpu.py` | Files listed in Fig. 4 row + `reports/2026-04-02T23-29-53/bloom_1b1_bisimulation.json` + `reports/2026-03-30T16-10-52/pythia_bisimulation.json` + `reports/2026-04-12T07-54-13/qwen3_8b_results.json` + `logs/2026-04-08T08-58-16/llama_bisimulation_results.json` |
| **Tab. 17** — Method requirements (narrative) | — | — |
| **Tab. 18** — BI vs interchange head-to-head | `bi_score_comparison.py` + `bi_vs_bisim_headtohead.py` | `reports/2026-03-31T00-18-24/{bi_score_comparison,bi_vs_bisim_ppl}.json` |
| **Tab. 19** — Geometry ablation | `bi_geometry_ablation.py` | `reports/2026-03-31T20-59-53/bi_geometry_ablation.json` |
| **Tab. 20** — Head-level swap-KL | `head_bisimulation.py` | `reports/2026-03-30T16-10-52/head_bisimulation.json` |
| **Tab. 21** — LoRA recovery on Qwen3-8B | `tpu_lora_recovery.py` + `tpu_lora_control.py` | `reports/lora_recovery_results.json` + `reports/lora_control_results.json` |
| **Tab. 22** — Downstream tasks GPT-2-M | `downstream_benchmark.py` | `reports/2026-03-31T00-18-24/downstream_benchmarks.json` |
| **Tab. 23** — Downstream tasks Qwen3-8B | `tpu_downstream_pytorch.py` | `reports/2026-04-03T19-27-07/qwen3_8b_downstream_v2.json` |
| **Tab. 24** — GPT-2-M compression | `compression_sweep.py` | Same as Fig. 6 |
| **Tab. 25** — Adjacent-pair profile GPT-2-M | `bisimulation_experiment.py` | Same as Fig. 7 |
| **Tab. 26** — Weight sharing on GPT-2-M | `weight_sharing_test.py` | `reports/2026-03-31T00-18-24/weight_sharing.json` |
| **Tab. 27** — Setup matrix (narrative) | — | — |
| **Tab. 28** — PPL evaluator configs (narrative) | — | — |
| **Tab. 29** — Jacobian norms GPT-2-M | `jacobian_norms.py` | `reports/2026-04-03T13-21-03/jacobian_norms.json` |
| **Tab. 30** — Jacobian norms Pythia-410M | `pythia_jacobian_norms.py` | `pythia_jacobian_norms.json` |
| **Tab. 31** — RoPE counterfactual (Qwen3-8B) | `tpu_rope_counterfactual.py` | `logs/2026-04-08T12-23-00/rope_counterfactual_results.json` |
| **Tab. 32** — Matched-budget head-to-head (Qwen3-8B, Llama-3.1-8B) | `matched_budget_qwen3.py` + `matched_budget_llama.py` | `reports/2026-04-22T15-00-00/matched_budget_{qwen3,llama}.json` |
| **App. K** — Pythia predictor validity | `pythia_predictor_validity.py` | `reports/2026-04-05T21-27-05/pythia_predictor_validity.json` + `reports/2026-04-05T21-52-56/pythia_predictor_validity_stronger.json` |
| **App. K** — Pythia checkpoint intervention | `checkpoint_intervention_pythia.py` + `pythia_checkpoint_trajectory.py` | `pythia_checkpoint_trajectory.json` |
| **App. L** — PE ablation (152M Chinchilla controls) | `pe_ablation_kaggle_p100.py` (GPU) + `pe_ablation_cpu.py` (CPU) + `pe_ablation_tpu.py` (TPU) + `pe_ablation_jax.py` + `run_pe_152m_chinchilla_pmap.sh` | `reports/2026-04-20T23-28-07/pe_ablation_*.json` |

## How the pieces fit together

```
prompt_set_100.py  -> used by every 100-prompt experiment
wikitext_ppl.py    -> sliding-window WikiText-2 evaluator used by every skip/baseline script

experiment scripts -> write JSON/CSV under reports/<cycle>/
figure scripts     -> read those files and write paper/figures/*.pdf
arxiv_submission.zip <- bundles paper/main.tex + .sty + .bbl + figures
```

## Reproducing a paper number end-to-end

Example: Tab. 32 row "Qwen3-8B, B=400, beam-bisim, n=5, +20.8%".

```bash
# Requires TPU v6e-8 + HuggingFace access. Archived output is already in-repo:
cat reports/2026-04-22T15-00-00/matched_budget_qwen3.json

# To rerun on TPU:
OUTPUT_DIR=reports/2026-04-22T15-00-00 python matched_budget_qwen3.py
```

Harmonized cross-model table (Tab. 10):

```bash
./run_harmonized_slice.sh
# writes reports/<cycle>/harmonized/{qwen,pythia}/
```

Mistral Tab. 8 sub-block (proxy replacement = interchange layers):

```bash
bash deploy_mistral_tpu.sh
# or locally: REPORT_DIR=reports/mistral_eval python matched_eval_mistral.py
```

The exact JSON we used is in `reports/2026-04-22T15-00-00/matched_budget_qwen3.json`; the table-as-figure that consumes it is `paper/figures/matched_budget.tex`.

## Hardware

- **GPT-2-Medium / Pythia / BLOOM / 152M Chinchilla** — single CPU notebook or 1× P100 (Kaggle).
- **Qwen3-8B / Llama-3.1-8B** — TPU v6e-8 (32 GB/chip) via gcloud. TPU quota via the TPU Research Cloud was used for every 8B run.
- **Mistral-7B-v0.1** — TPU v6e-8 or 4-bit on a single 24 GB GPU.

## Citation

```bibtex
@misc{garcia2026freeswapprotocoldependentlayer,
  title         = {No Free Swap: Protocol-Dependent Layer Redundancy in Transformers},
  author        = {Gabriel Garcia},
  year          = {2026},
  eprint        = {2605.16234},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2605.16234},
}
```

## License

- Code: MIT (`LICENSE`).
- Paper PDF, LaTeX source, and figures under `paper/`: CC BY 4.0 (`LICENSE`).

## Contact

Gabriel Garcia — gpgabriel25@gmail.com — Independent researcher.
