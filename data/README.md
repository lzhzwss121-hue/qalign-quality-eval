# Data Layout

Image datasets are not redistributed in this repository. Create the following
local folders before running the scripts.

## Degradation Evaluation

```text
data/images/
  hr/
    baboon.png
    lenna.png
    ...
  bicubic/
    baboonx4.png
    lennax4.png
    ...
  blur/
    baboon.png
    lenna.png
    ...
  noise/
    baboon.png
    lenna.png
    ...
```

The scripts normalize file names so suffixes such as `x4`, `_x4`, `_lr`, and
`_sr` can still match the high-resolution reference name.

## Restoration Evaluation

```text
data/images_restoration/
  hr/
    baboon.png
    lenna.png
    ...
  swinir/
    baboon.png
    lenna.png
    ...
  mambairv2/
    baboon.png
    lenna.png
    ...
```

All methods should be evaluated against the same reference images and the same
crop-border setting.

## SR Model Output Comparison

```text
datasets/imageSR/
  Set5/HR/
  Set5/LR_bicubic/X4/
  Set14/HR/
  ...

model_outputs/
  SwinIR/Set5/
  SwinIR/Set14/
  SwinIR/B100/
  SwinIR/Urban100/
  SwinIR/Manga109/
  MambaIR/...
  MambaIRv2/...
```

The output evaluator normalizes suffixes such as `x4_SwinIR`,
`x4_MambaIRv2`, and `_LRBI_x4_MambaIRv2` so predictions can be matched to HR
references.
