# Research Summary

## Motivation

Image restoration papers usually report fidelity metrics such as PSNR and SSIM.
These metrics are useful but can under-represent perceptual quality, semantic
consistency, and user-facing visual preference. This project explores whether a
zero-shot VLM-based quality score, Q-Align / OneAlign, provides a complementary
evaluation signal for degraded and restored images.

## Method

The pipeline computes four metrics for each image pair:

- PSNR for pixel-level fidelity
- SSIM for structural fidelity
- LPIPS for perceptual similarity
- Q-Align score for zero-shot VLM-based image quality assessment

For the degradation study, each degraded image is matched to a high-resolution
reference image. The script then computes all metrics, aggregates correlations,
and mines representative disagreement cases.

For the restoration extension, the same evaluation logic is applied to SwinIR
and MambaIRv2 outputs under a fixed crop-border protocol.

## Current Results

The degradation benchmark contains 42 samples: 14 bicubic, 14 blur, and 14 noise
cases. The SR bicubic benchmark contains 984 image-scale pairs from Set5,
Set14, B100, Urban100, and Manga109 across X2, X3, and X4 scales. The restored
SR output comparison contains X4 outputs from SwinIR, MambaIR, and MambaIRv2 on
the same five datasets. The SR Q-Align extension contains 1,312 X4 rows after
adding Bicubic, LPIPS, and Q-Align scores.

| Metric Pair | Pearson | Spearman | n |
| --- | ---: | ---: | ---: |
| PSNR vs Q-Align | 0.084 | 0.106 | 42 |
| SSIM vs Q-Align | 0.107 | 0.184 | 42 |
| LPIPS vs Q-Align | -0.639 | -0.669 | 42 |

The negative LPIPS correlation is expected because lower LPIPS means better
perceptual similarity, while higher Q-Align means higher predicted quality.

The restoration extension currently contains 28 outputs:

| Method | Mean PSNR | Mean SSIM | Mean LPIPS | Mean Q-Align |
| --- | ---: | ---: | ---: | ---: |
| SwinIR | 27.278 | 0.779 | 0.265 | 3.282 |
| MambaIRv2 | 19.092 | 0.495 | 0.300 | 3.386 |

The MambaIRv2 restoration results are kept as diagnostic evidence rather than a
method comparison claim, because the current outputs reveal reference/output
alignment and benchmark-protocol sensitivity.

The SR bicubic benchmark first mod-crops each HR image to `LR_size * scale`, then
uses Y-channel metrics with crop border equal to scale, which is common in
super-resolution evaluation:

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

The restored-output comparison shows that learned SR models provide a large
improvement over bicubic interpolation at X4. MambaIRv2 has the highest
Y-channel PSNR on all five datasets:

| Dataset | Bicubic | SwinIR | MambaIR | MambaIRv2 |
| --- | ---: | ---: | ---: | ---: |
| Set5 | 28.429 | 32.928 | 33.029 | 33.144 |
| Set14 | 26.085 | 29.086 | 29.202 | 29.233 |
| B100 | 25.954 | 27.925 | 27.979 | 27.995 |
| Urban100 | 23.136 | 27.455 | 27.681 | 27.895 |
| Manga109 | 24.896 | 32.036 | 32.317 | 32.567 |

The largest MambaIRv2 gains over SwinIR appear on Urban100 (+0.440 dB) and
Manga109 (+0.531 dB). This matches the hypothesis that attentive state-space
restoration is more useful on datasets with long-range repetitive structures,
building facades, text-like edges, and manga line art.

The SR Q-Align extension shows that Q-Align strongly separates learned SR
outputs from bicubic interpolation, but it is not identical to the PSNR ranking
among learned methods:

| Dataset | Bicubic | SwinIR | MambaIR | MambaIRv2 |
| --- | ---: | ---: | ---: | ---: |
| Set5 | 1.653 | 3.102 | 3.045 | 3.057 |
| Set14 | 1.963 | 3.322 | 3.285 | 3.292 |
| B100 | 1.732 | 2.980 | 2.946 | 2.956 |
| Urban100 | 2.857 | 4.378 | 4.360 | 4.359 |
| Manga109 | 2.904 | 3.652 | 3.637 | 3.639 |

Across all 1,312 SR output rows, LPIPS has the strongest relationship with
Q-Align (Pearson -0.664, Spearman -0.670), followed by SSIM-Y (0.487, 0.501)
and PSNR-Y (0.253, 0.278).

## Interpretation

The pilot result supports six cautious conclusions:

1. Q-Align is not redundant with PSNR or SSIM in this setting.
2. Q-Align is closer to perceptual similarity than to pixel-level fidelity in
   the aggregate result.
3. Restoration evaluation requires strict control over scale, crop border,
   reference pairing, and official benchmark protocol before making performance
   claims.
4. Urban100 and Manga109 show a sharper quality drop at X4, which makes them
   useful stress tests for structure- and texture-sensitive restoration methods.
5. The X4 output comparison supports MambaIRv2 as the strongest model among
   SwinIR, MambaIR, and MambaIRv2 under the current protocol, especially on
   Urban100 and Manga109.
6. On SR outputs, Q-Align is closer to LPIPS than to PSNR-Y, and the remaining
   method-ranking differences are useful cases for metric-disagreement analysis.

## Next Steps

- Add X2 and X3 restored-output comparisons when full prediction folders are
  available.
- Add JPEG compression and mixed-degradation cases.
- Compare Q-Align with CLIP-IQA and MUSIQ.
- Add more restoration methods under the same official evaluation protocol.
- Convert disagreement cases into a qualitative taxonomy of metric failure
  modes.
