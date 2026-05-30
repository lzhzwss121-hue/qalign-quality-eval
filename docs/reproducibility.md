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

The script first mod-crops each HR image to `LR_size * scale`, then upscales the
LR image with bicubic interpolation and computes RGB metrics plus Y-channel
metrics. The Y-channel metrics use 255 as the data range. The default Y-channel
crop border is equal to the SR scale.

## SR Model Output Comparison

The restored-output comparison expects saved model predictions organized by
dataset:

```text
method_outputs/
  Set5/
  Set14/
  B100/
  Urban100/
  Manga109/
```

Run the X4 comparison:

```bash
python scripts/eval_sr_model_outputs.py \
  --dataset_root /path/to/datasets/imageSR \
  --scale 4 \
  --method_dirs \
    SwinIR=/path/to/SwinIR/visualization \
    MambaIR=/path/to/MambaIR/visualization \
    MambaIRv2=/path/to/MambaIRv2/visualization \
  --bicubic_csv ./results/sr_bicubic/raw_metrics.csv \
  --output_dir ./results/sr_model_comparison
```

Generate comparison plots:

```bash
MPLCONFIGDIR=/tmp/qalign_mpl_cache python scripts/plot_sr_model_comparison.py \
  --raw_csv ./results/sr_model_comparison/raw_metrics_x4.csv \
  --summary_csv ./results/sr_model_comparison/summary_by_method_dataset_x4.csv \
  --improvement_csv ./results/sr_model_comparison/improvement_over_bicubic_x4.csv \
  --case_csv ./results/sr_model_comparison/case_study_x4.csv \
  --fig_dir ./results/figures
```

If prediction dimensions differ from the original HR dimensions because the HR
image is not divisible by the SR scale, the script crops the HR image to the
prediction size. It does not resize restored predictions for metric computation.

### Q-Align / LPIPS scoring on SR outputs

Run a small smoke test first:

```bash
python scripts/eval_sr_model_qalign.py \
  --input_csv ./results/sr_model_comparison/raw_metrics_x4.csv \
  --output_csv ./results/sr_model_qalign/raw_metrics_x4.csv \
  --summary_csv ./results/sr_model_qalign/summary_by_method_dataset_x4.csv \
  --correlation_csv ./results/sr_model_qalign/correlations_x4.csv \
  --case_csv ./results/sr_model_qalign/disagreement_cases_x4.csv \
  --max_images 20
```

Then run the full evaluation with resume enabled:

```bash
python scripts/eval_sr_model_qalign.py \
  --input_csv ./results/sr_model_comparison/raw_metrics_x4.csv \
  --output_csv ./results/sr_model_qalign/raw_metrics_x4.csv \
  --summary_csv ./results/sr_model_qalign/summary_by_method_dataset_x4.csv \
  --correlation_csv ./results/sr_model_qalign/correlations_x4.csv \
  --case_csv ./results/sr_model_qalign/disagreement_cases_x4.csv \
  --resume
```

The script scores generated images with Q-Align and computes LPIPS against the
same cropped references used by the SR model-output comparison.

## Important Evaluation Details

- The degradation and diagnostic restoration scripts may resize predictions to
  match references when needed; the SR model-output comparison does not resize
  restored predictions for metric computation.
- LPIPS is computed on RGB tensors normalized to `[-1, 1]`.
- Q-Align is scored on the predicted/restored image only.
- Restoration evaluation supports crop-border evaluation through
  `--crop_border`.
- SR bicubic evaluation reports both RGB and cropped Y-channel PSNR/SSIM after
  HR mod-cropping to `LR_size * scale`.
- SR model-output comparison crops HR references to prediction dimensions when
  needed, then applies the same cropped Y-channel metric protocol.

## Known Sources of Error

- Mismatched file names between reference and prediction folders.
- Incorrect output scale in super-resolution outputs.
- Different benchmark crop-border protocols.
- Comparing restored images generated from different low-resolution inputs.
- Re-running Q-Align with a changed model implementation or checkpoint version.
