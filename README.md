# Q-Align Quality Evaluation

Research-oriented evaluation pipeline for comparing traditional image-quality
metrics with zero-shot vision-language-model (VLM) quality scoring.

The project studies when fidelity-based metrics such as PSNR/SSIM disagree with
perceptual and VLM-based assessments. It is designed as a compact research
artifact for image restoration, perceptual image quality assessment, and
trustworthy multimodal evaluation.

## Research Focus

- **Fidelity metrics:** PSNR and SSIM
- **Perceptual metric:** LPIPS
- **Zero-shot VLM quality score:** Q-Align / OneAlign
- **Image degradation settings:** bicubic downsampling, blur, and noise
- **Experimental restoration extension:** SwinIR and MambaIRv2 output evaluation

This repository focuses on evaluation and analysis rather than training new
restoration models.

## Key Findings

The current pilot benchmark contains 42 degraded images across three degradation
types and 28 restoration outputs for the restoration extension.

| Analysis | Main Observation |
| --- | --- |
| Overall PSNR vs Q-Align | Weak correlation: Pearson 0.084, Spearman 0.106 |
| Overall SSIM vs Q-Align | Weak correlation: Pearson 0.107, Spearman 0.184 |
| Overall LPIPS vs Q-Align | Moderate inverse correlation: Pearson -0.639, Spearman -0.669 |
| Bicubic / blur subsets | Q-Align aligns more with LPIPS than with PSNR |
| Restoration extension | Benchmark consistency issues can strongly affect metric interpretation |

These findings suggest that zero-shot VLM-based quality scores capture a
different evaluation signal from pixel-level fidelity metrics. The results should
be treated as a small-scale exploratory study, not as a final benchmark claim.

## Repository Structure

```text
scripts/
  run_eval.py                         # PSNR / SSIM / LPIPS / Q-Align evaluation
  analyze_metrics.py                  # Correlation analysis
  plot_results.py                     # Scatter and bar plots
  make_case_study.py                  # Disagreement-case mining
  make_case_board_compact.py          # Visual case board generation
  validate_results.py                 # Lightweight result-file validation
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

docs/
  research_summary.md
  reproducibility.md

data/
  README.md                           # Expected local data layout
```

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Q-Align / OneAlign uses Hugging Face model loading with
`trust_remote_code=True`. A CUDA GPU is recommended.

### 2. Prepare data

The image data are not redistributed in this repository. Prepare the local
folder layout described in [`data/README.md`](data/README.md).

### 3. Run degradation evaluation

```bash
python scripts/run_eval.py \
  --data_root ./data/images \
  --output_csv ./results/raw_metrics.csv \
  --degradations bicubic blur noise
```

### 4. Run correlation analysis

```bash
python scripts/analyze_metrics.py \
  --input_csv ./results/raw_metrics.csv \
  --output_csv ./results/summary_stats.csv
```

### 5. Generate plots and case studies

```bash
python scripts/plot_results.py \
  --raw_csv ./results/raw_metrics.csv \
  --summary_csv ./results/summary_stats.csv \
  --fig_dir ./results/figures

python scripts/make_case_study.py \
  --input_csv ./results/raw_metrics.csv \
  --output_csv ./results/case_study.csv

python scripts/make_case_board_compact.py \
  --case_csv ./results/case_study.csv \
  --output_path ./results/figures/case_study_board_compact.png
```

### 6. Validate stored result files

```bash
python scripts/validate_results.py
```

## Restoration Extension

The restoration evaluation scripts compare outputs from restoration methods such
as SwinIR and MambaIRv2:

```bash
python scripts/experimental/run_eval_restoration_official.py \
  --data_root ./data/images_restoration \
  --output_csv ./results/experimental/raw_metrics_restoration_official.csv \
  --methods swinir mambairv2 \
  --crop_border 4
```

This part is intentionally marked as experimental because restoration evaluation
is sensitive to reference/output pairing, output scale, crop-border protocol, and
official benchmark settings.

## Result Files

See [`results/README.md`](results/README.md) for a detailed explanation of the
stored CSV files and generated figures.

## Limitations

- The current degradation benchmark is small (`n=42`).
- Q-Align scores are zero-shot image-quality estimates and are not a substitute
  for human opinion scores.
- The restoration extension is a diagnostic study, not a leaderboard.
- Reproducing Q-Align requires downloading external model weights.

## Research Relevance

This repository is part of a broader research direction on VLM-guided image
restoration and multimodal quality assessment. The immediate goal is to build a
clean evaluation pipeline that can be extended to larger restoration benchmarks
and more robust VLM reliability analysis.
