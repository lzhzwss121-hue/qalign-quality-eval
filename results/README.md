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

`figures/` contains generated scatter plots, case-study boards, and SR
comparison figures:

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
PSNR/SSIM values. The benchmark mod-crops HR images to `LR_size * scale` before
comparison. The headline Y-channel cropped results are:

| Dataset | Scale | n | Mean PSNR-Y | Mean SSIM-Y |
| --- | ---: | ---: | ---: | ---: |
| Set5 | X2 | 5 | 33.673 | 0.937 |
| Set5 | X3 | 5 | 30.403 | 0.881 |
| Set5 | X4 | 5 | 28.429 | 0.823 |
| Set14 | X2 | 14 | 30.323 | 0.881 |
| Set14 | X3 | 14 | 27.628 | 0.794 |
| Set14 | X4 | 14 | 26.085 | 0.722 |
| B100 | X2 | 100 | 29.552 | 0.857 |
| B100 | X3 | 100 | 27.202 | 0.759 |
| B100 | X4 | 100 | 25.954 | 0.685 |
| Urban100 | X2 | 100 | 26.870 | 0.850 |
| Urban100 | X3 | 100 | 24.455 | 0.751 |
| Urban100 | X4 | 100 | 23.136 | 0.673 |
| Manga109 | X2 | 109 | 30.816 | 0.938 |
| Manga109 | X3 | 109 | 26.951 | 0.864 |
| Manga109 | X4 | 109 | 24.896 | 0.795 |

Generated SR figures:

- `figures/sr_bicubic_psnr_y_by_scale.png`
- `figures/sr_bicubic_ssim_y_by_scale.png`
- `figures/sr_bicubic_x4_psnr_y_by_dataset.png`
- `figures/sr_bicubic_x4_failure_board.png`

## SR Model Output Comparison

`sr_model_comparison/raw_metrics_x4.csv` contains per-image X4 metrics for
Bicubic, SwinIR, MambaIR, and MambaIRv2. The neural outputs are evaluated by
cropping HR references to the prediction size when the original HR dimensions are
not divisible by the scale.

`sr_model_comparison/summary_by_method_dataset_x4.csv` stores aggregate
Y-channel and RGB PSNR/SSIM by method and dataset:

| Dataset | Bicubic | SwinIR | MambaIR | MambaIRv2 |
| --- | ---: | ---: | ---: | ---: |
| Set5 | 28.429 | 32.928 | 33.029 | 33.144 |
| Set14 | 26.085 | 29.086 | 29.202 | 29.233 |
| B100 | 25.954 | 27.925 | 27.979 | 27.995 |
| Urban100 | 23.136 | 27.455 | 27.681 | 27.895 |
| Manga109 | 24.896 | 32.036 | 32.317 | 32.567 |

`sr_model_comparison/improvement_over_bicubic_x4.csv` reports average gains and
win rates over bicubic interpolation. MambaIRv2 improves over bicubic by 2.041 dB
on B100, 3.148 dB on Set14, 4.715 dB on Set5, 4.759 dB on Urban100, and 7.672 dB
on Manga109.

`sr_model_comparison/case_study_x4.csv` stores representative cases where
MambaIRv2 gains most over SwinIR or bicubic, plus low-PSNR hard cases.

Generated SR model-comparison figures:

- `figures/sr_model_x4_psnr_y_by_dataset.png`
- `figures/sr_model_x4_ssim_y_by_dataset.png`
- `figures/sr_model_x4_delta_psnr_y_over_bicubic.png`
- `figures/sr_model_x4_win_rate_over_bicubic.png`
- `figures/sr_model_x4_qualitative_board.png`
