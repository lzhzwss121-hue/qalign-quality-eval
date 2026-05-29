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
