# Result Files

This directory stores the current pilot results used in the project README and
research summary.

## Degradation Evaluation

`raw_metrics.csv` contains one row per degraded image:

- `image_name`
- `base_name`
- `degradation`
- `gt_path`
- `pred_path`
- `width`
- `height`
- `psnr`
- `ssim`
- `lpips`
- `qalign_score`

`summary_stats.csv` stores Pearson and Spearman correlations between metric
pairs. The current overall result is:

| Metric Pair | Pearson | Spearman | n |
| --- | ---: | ---: | ---: |
| PSNR vs Q-Align | 0.084 | 0.106 | 42 |
| SSIM vs Q-Align | 0.107 | 0.184 | 42 |
| LPIPS vs Q-Align | -0.639 | -0.669 | 42 |

`case_study.csv` stores selected disagreement cases, such as images with high
PSNR but low Q-Align or low PSNR but high Q-Align.

## Figures

`figures/` contains generated scatter plots and a compact case-study board:

- `scatter_psnr_qalign.png`
- `scatter_ssim_qalign.png`
- `scatter_lpips_qalign.png`
- `bar_correlation_with_qalign.png`
- `case_study_board_compact.png`

## Restoration Extension

`experimental/raw_metrics_restoration_official.csv` stores restoration-output
evaluation for SwinIR and MambaIRv2 under the current crop-border setting.

Current mean values:

| Method | Mean PSNR | Mean SSIM | Mean LPIPS | Mean Q-Align |
| --- | ---: | ---: | ---: | ---: |
| SwinIR | 27.278 | 0.779 | 0.265 | 3.282 |
| MambaIRv2 | 19.092 | 0.495 | 0.300 | 3.386 |

These numbers should be interpreted as diagnostic outputs. They are not intended
as a final benchmark comparison.
