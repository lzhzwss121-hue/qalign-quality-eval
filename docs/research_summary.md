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
cases.

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

## Interpretation

The pilot result supports three cautious conclusions:

1. Q-Align is not redundant with PSNR or SSIM in this setting.
2. Q-Align is closer to perceptual similarity than to pixel-level fidelity in
   the aggregate result.
3. Restoration evaluation requires strict control over scale, crop border,
   reference pairing, and official benchmark protocol before making performance
   claims.

## Next Steps

- Expand the benchmark to Set5, Set14, BSD100, and Urban100 subsets.
- Add JPEG compression and mixed-degradation cases.
- Compare Q-Align with CLIP-IQA and MUSIQ.
- Add more restoration methods under the same official evaluation protocol.
- Convert disagreement cases into a qualitative taxonomy of metric failure
  modes.
