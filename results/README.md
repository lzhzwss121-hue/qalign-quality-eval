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

## SR Bicubic Benchmark

`sr_bicubic/raw_metrics.csv` contains 984 image-scale pairs from Set5, Set14,
B100, Urban100, and Manga109 across X2, X3, and X4 bicubic LR inputs.

`sr_bicubic/summary_by_dataset_scale.csv` contains aggregate RGB and Y-channel
PSNR/SSIM values. The headline Y-channel cropped results are:

| Dataset | Scale | n | Mean PSNR-Y | Mean SSIM-Y |
| --- | ---: | ---: | ---: | ---: |
| Set5 | X2 | 5 | 32.963 | 0.933 |
| Set5 | X3 | 5 | 27.915 | 0.845 |
| Set5 | X4 | 5 | 27.720 | 0.815 |
| Set14 | X2 | 14 | 29.203 | 0.869 |
| Set14 | X3 | 14 | 25.097 | 0.738 |
| Set14 | X4 | 14 | 24.991 | 0.695 |
| B100 | X2 | 100 | 26.714 | 0.795 |
| B100 | X3 | 100 | 26.119 | 0.738 |
| B100 | X4 | 100 | 24.879 | 0.662 |
| Urban100 | X2 | 100 | 25.549 | 0.832 |
| Urban100 | X3 | 100 | 22.751 | 0.711 |
| Urban100 | X4 | 100 | 21.922 | 0.645 |
| Manga109 | X2 | 109 | 30.106 | 0.935 |
| Manga109 | X3 | 109 | 25.472 | 0.846 |
| Manga109 | X4 | 109 | 22.279 | 0.740 |

Generated SR figures:

- `figures/sr_bicubic_psnr_y_by_scale.png`
- `figures/sr_bicubic_ssim_y_by_scale.png`
- `figures/sr_bicubic_x4_psnr_y_by_dataset.png`
- `figures/sr_bicubic_x4_failure_board.png`
