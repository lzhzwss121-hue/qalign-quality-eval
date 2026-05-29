# Reproducibility Notes

## Environment

Use Python 3.10 or later when possible.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The Q-Align model is loaded through Hugging Face:

```text
q-future/one-align
```

The scripts use `trust_remote_code=True` because the model repository defines a
custom scoring interface. A CUDA GPU is recommended for practical runtime.

## Data Layout

The repository does not include image datasets. Prepare local data according to
the layouts in [`../data/README.md`](../data/README.md).

## Core Pipeline

Run the evaluation:

```bash
python scripts/run_eval.py \
  --data_root ./data/images \
  --output_csv ./results/raw_metrics.csv \
  --degradations bicubic blur noise
```

Analyze correlations:

```bash
python scripts/analyze_metrics.py \
  --input_csv ./results/raw_metrics.csv \
  --output_csv ./results/summary_stats.csv
```

Generate visualizations:

```bash
python scripts/plot_results.py \
  --raw_csv ./results/raw_metrics.csv \
  --summary_csv ./results/summary_stats.csv \
  --fig_dir ./results/figures
```

Mine cases where metrics disagree:

```bash
python scripts/make_case_study.py \
  --input_csv ./results/raw_metrics.csv \
  --output_csv ./results/case_study.csv
```

Validate saved result files:

```bash
python scripts/validate_results.py
```

## SR Bicubic Benchmark

The SR extension expects standard test-set folders with this layout:

```text
datasets/imageSR/
  Set5/
    HR/
    LR_bicubic/X2/
    LR_bicubic/X3/
    LR_bicubic/X4/
  Set14/
  B100/
  Urban100/
  Manga109/
```

Run the benchmark:

```bash
python scripts/eval_sr_bicubic.py \
  --dataset_root /path/to/datasets/imageSR \
  --output_dir ./results/sr_bicubic
```

Generate plots and a failure-case board:

```bash
MPLCONFIGDIR=/tmp/qalign_mpl_cache python scripts/plot_sr_bicubic.py \
  --summary_csv ./results/sr_bicubic/summary_by_dataset_scale.csv \
  --failure_csv ./results/sr_bicubic/failure_cases.csv \
  --fig_dir ./results/figures
```

The script upscales each LR image to the HR size with bicubic interpolation and
computes RGB metrics plus Y-channel metrics. The default Y-channel crop border is
equal to the SR scale.

## Important Evaluation Details

- Predicted images are resized to match the reference image if needed.
- LPIPS is computed on RGB tensors normalized to `[-1, 1]`.
- Q-Align is scored on the predicted/restored image only.
- Restoration evaluation supports crop-border evaluation through
  `--crop_border`.
- SR bicubic evaluation reports both RGB and cropped Y-channel PSNR/SSIM.

## Known Sources of Error

- Mismatched file names between reference and prediction folders.
- Incorrect output scale in super-resolution outputs.
- Different benchmark crop-border protocols.
- Comparing restored images generated from different low-resolution inputs.
- Re-running Q-Align with a changed model implementation or checkpoint version.
