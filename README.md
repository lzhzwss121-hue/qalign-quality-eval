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
- **SR benchmark extension:** Set5, Set14, B100, Urban100, and Manga109
- **SR model comparison:** Bicubic, SwinIR, MambaIR, and MambaIRv2
- **Experimental restoration extension:** SwinIR and MambaIRv2 output evaluation

This repository focuses on evaluation and analysis rather than training new
restoration models.

## Key Findings

The current pilot benchmark contains 42 degraded images across three degradation
types, 984 bicubic super-resolution benchmark samples, 984 neural SR outputs
from SwinIR, MambaIR, and MambaIRv2, 1,312 Q-Align/LPIPS-scored X4 SR outputs,
and 28 restoration outputs for the restoration extension.

| Analysis | Main Observation |
| --- | --- |
| Overall PSNR vs Q-Align | Weak correlation: Pearson 0.084, Spearman 0.106 |
| Overall SSIM vs Q-Align | Weak correlation: Pearson 0.107, Spearman 0.184 |
| Overall LPIPS vs Q-Align | Moderate inverse correlation: Pearson -0.639, Spearman -0.669 |
| Bicubic / blur subsets | Q-Align aligns more with LPIPS than with PSNR |
| SR bicubic benchmark | X4 bicubic is hardest on Urban100 and Manga109 |
| SR restored-output comparison | MambaIRv2 is best by Y-channel PSNR on all five X4 test sets |
| SR Q-Align extension | LPIPS has the strongest correlation with Q-Align: Pearson -0.664, Spearman -0.670 |
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
  eval_sr_bicubic.py                  # Standard SR bicubic benchmark evaluation
  plot_sr_bicubic.py                  # SR benchmark plots and failure board
  eval_sr_model_outputs.py            # Restored-output comparison evaluation
  plot_sr_model_comparison.py         # Restored-output comparison plots
  eval_sr_model_qalign.py             # Q-Align / LPIPS scoring for SR outputs
  plot_sr_model_qalign.py             # Q-Align / LPIPS plots for SR outputs
  experimental/
    generate_degradations.py
    run_eval_restoration.py
    run_eval_restoration_official.py

results/
  raw_metrics.csv
  summary_stats.csv
  case_study.csv
  figures/
  sr_bicubic/
  sr_model_comparison/
  sr_model_qalign/
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
`trust_remote_code=True`. The pinned `transformers==4.36.1` dependency is used
for compatibility with the OneAlign remote-code implementation. A CUDA GPU is
recommended.

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

## SR Bicubic Benchmark

The SR benchmark extension evaluates bicubic LR inputs on five standard
super-resolution test sets: Set5, Set14, B100, Urban100, and Manga109. The
current run covers X2, X3, and X4 scales, for 984 evaluated image-scale pairs.

```bash
python scripts/eval_sr_bicubic.py \
  --dataset_root /path/to/datasets/imageSR \
  --output_dir ./results/sr_bicubic

python scripts/plot_sr_bicubic.py \
  --summary_csv ./results/sr_bicubic/summary_by_dataset_scale.csv \
  --failure_csv ./results/sr_bicubic/failure_cases.csv \
  --fig_dir ./results/figures
```

The script first mod-crops each HR image to `LR_size * scale`, then upsamples the
LR image with bicubic interpolation. It reports both RGB full-reference metrics
and Y-channel cropped metrics. The Y-channel metric uses 255 as the data range
and a crop border equal to the SR scale, matching common SR evaluation practice.

Current X4 Y-channel PSNR results:

| Dataset | n | X4 PSNR-Y | X4 SSIM-Y |
| --- | ---: | ---: | ---: |
| Set5 | 5 | 28.429 | 0.823 |
| Set14 | 14 | 26.085 | 0.722 |
| B100 | 100 | 25.954 | 0.685 |
| Manga109 | 109 | 24.896 | 0.795 |
| Urban100 | 100 | 23.136 | 0.673 |

## SR Model Output Comparison

The restored-output comparison evaluates saved X4 outputs from SwinIR, MambaIR,
and MambaIRv2 against the same HR references and bicubic baseline. It uses the
same cropped Y-channel PSNR/SSIM protocol as the bicubic benchmark. If model
outputs are generated at `LR_size * scale`, the script crops the HR reference to
the prediction size instead of resizing predictions.

```bash
python scripts/eval_sr_model_outputs.py \
  --dataset_root /path/to/datasets/imageSR \
  --scale 4 \
  --method_dirs \
    SwinIR=/path/to/SwinIR_SR_x4/visualization \
    MambaIR=/path/to/MambaIR_SR_x4/visualization \
    MambaIRv2=/path/to/MambaIRv2_SR_x4/visualization \
  --bicubic_csv ./results/sr_bicubic/raw_metrics.csv \
  --output_dir ./results/sr_model_comparison

python scripts/plot_sr_model_comparison.py \
  --raw_csv ./results/sr_model_comparison/raw_metrics_x4.csv \
  --summary_csv ./results/sr_model_comparison/summary_by_method_dataset_x4.csv \
  --improvement_csv ./results/sr_model_comparison/improvement_over_bicubic_x4.csv \
  --case_csv ./results/sr_model_comparison/case_study_x4.csv \
  --fig_dir ./results/figures
```

Current X4 Y-channel PSNR results:

| Dataset | Bicubic | SwinIR | MambaIR | MambaIRv2 |
| --- | ---: | ---: | ---: | ---: |
| Set5 | 28.429 | 32.928 | 33.029 | 33.144 |
| Set14 | 26.085 | 29.086 | 29.202 | 29.233 |
| B100 | 25.954 | 27.925 | 27.979 | 27.995 |
| Urban100 | 23.136 | 27.455 | 27.681 | 27.895 |
| Manga109 | 24.896 | 32.036 | 32.317 | 32.567 |

MambaIRv2 gives the largest gains over SwinIR on Urban100 (+0.440 dB) and
Manga109 (+0.531 dB), which are the most structure- and texture-sensitive
benchmarks in this set.

### Optional Q-Align / LPIPS scoring

For GPU servers, the saved SR outputs can be further scored with LPIPS and
Q-Align:

```bash
python scripts/eval_sr_model_qalign.py \
  --qalign_model /path/to/one-align \
  --input_csv ./results/sr_model_comparison/raw_metrics_x4.csv \
  --output_csv ./results/sr_model_qalign/raw_metrics_x4_smoke.csv \
  --summary_csv ./results/sr_model_qalign/summary_by_method_dataset_x4_smoke.csv \
  --correlation_csv ./results/sr_model_qalign/correlations_x4_smoke.csv \
  --case_csv ./results/sr_model_qalign/disagreement_cases_x4_smoke.csv \
  --max_images 20
```

```bash
python scripts/eval_sr_model_qalign.py \
  --qalign_model /path/to/one-align \
  --input_csv ./results/sr_model_comparison/raw_metrics_x4.csv \
  --output_csv ./results/sr_model_qalign/raw_metrics_x4.csv \
  --summary_csv ./results/sr_model_qalign/summary_by_method_dataset_x4.csv \
  --correlation_csv ./results/sr_model_qalign/correlations_x4.csv \
  --case_csv ./results/sr_model_qalign/disagreement_cases_x4.csv \
  --resume

python scripts/plot_sr_model_qalign.py \
  --raw_csv ./results/sr_model_qalign/raw_metrics_x4.csv \
  --summary_csv ./results/sr_model_qalign/summary_by_method_dataset_x4.csv \
  --correlation_csv ./results/sr_model_qalign/correlations_x4.csv \
  --fig_dir ./results/figures
```

Current X4 mean Q-Align results:

| Dataset | Bicubic | SwinIR | MambaIR | MambaIRv2 |
| --- | ---: | ---: | ---: | ---: |
| Set5 | 1.653 | 3.102 | 3.045 | 3.057 |
| Set14 | 1.963 | 3.322 | 3.285 | 3.292 |
| B100 | 1.732 | 2.980 | 2.946 | 2.956 |
| Urban100 | 2.857 | 4.378 | 4.360 | 4.359 |
| Manga109 | 2.904 | 3.652 | 3.637 | 3.639 |

The full SR Q-Align run contains 1,312 valid rows with no scoring errors.

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
