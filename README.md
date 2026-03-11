# Q-Align Quality Evaluation

This repository contains a research-oriented evaluation pipeline for comparing traditional image-quality metrics and zero-shot VLM-based quality scoring.

## Overview

The project focuses on comparing:

- **Fidelity metrics:** PSNR, SSIM
- **Perceptual metric:** LPIPS
- **Zero-shot VLM-based quality score:** Q-Align

The main stable part of the repository studies metric behavior under multiple **image degradation settings**, including:

- bicubic
- blur
- noise

In addition, the repository contains an **experimental restoration extension** exploring restoration-output evaluation with models such as SwinIR and MambaIRv2.

## Main Findings

- Q-Align behaves differently from traditional fidelity metrics such as PSNR.
- In several degradation settings, Q-Align shows stronger alignment with perceptual similarity metrics than with pixel-level fidelity measures.
- Case studies reveal representative disagreement patterns between fidelity-based and VLM-based quality scoring.
- During restoration exploration, a benchmark mismatch case was identified when integrating MambaIRv2 outputs, highlighting the importance of evaluation consistency in image restoration research.

## Repository Structure

```text
scripts/
  run_eval.py
  analyze_metrics.py
  plot_results.py
  make_case_study.py
  make_case_board_compact.py
  experimental/
    generate_degradations.py
    run_eval_restoration.py
    run_eval_restoration_official.py

results/
  raw_metrics.csv
  summary_stats.csv
  case_study.csv
  figures/
  experimental/


Example Usage

1. Degradation evaluation
python scripts/run_eval.py \
  --data_root ./data/images \
  --output_csv ./results/raw_metrics.csv \
  --degradations bicubic blur noise

2. Correlation analysis
python scripts/analyze_metrics.py \
  --input_csv ./results/raw_metrics.csv \
  --output_csv ./results/summary_stats.csv

3. Plotting
python scripts/plot_results.py \
  --raw_csv ./results/raw_metrics.csv \
  --summary_csv ./results/summary_stats.csv \
  --fig_dir ./results/figures

4. Case-study mining
python scripts/make_case_study.py \
  --input_csv ./results/raw_metrics.csv \
  --output_csv ./results/case_study.csv

Notes

This repository focuses on evaluation and analysis, rather than training new restoration models.

The restoration branch is currently kept as an experimental extension, since restoration benchmarking is highly sensitive to:

-GT/reference consistency

-output scale alignment

-crop-border protocol

-benchmark evaluation settings
