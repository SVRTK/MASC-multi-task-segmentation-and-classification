from monai import transforms
import torch

train_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image", "mask", "LP"], allow_missing_keys=True),
        transforms.AddChanneld(keys=["image", "mask", "LP"], allow_missing_keys=True),
        transforms.NormalizeIntensityd(keys=["image"]),
        transforms.ScaleIntensityd(keys=["image"],
                                   minv=0.0, maxv=1.0),

        transforms.RandShiftIntensityd(
            keys=["image"],
            offsets=0.1,
            prob=0.30,
        ),
        transforms.RandGaussianSmoothd(
            keys=["image"],
            prob=0.30,
        ),
        transforms.RandGaussianNoised(
            keys=["image"],
            prob=0.30,
        ),
        transforms.RandGaussianSharpend(
            keys=["image"],
            prob=0.30,
        ),
        transforms.RandHistogramShiftd(
            keys=["image"],
            prob=0.30,
        ),
        transforms.RandAdjustContrastd(
            keys=["image"],
            prob=0.30,
        ),
        transforms.RandBiasFieldd(
            keys=["image"],
            prob=0.30,
        ),

        transforms.RandAffined(
            keys=["image", "mask", "LP"],
            allow_missing_keys=True,
            rotate_range=[(-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)],
            translate_range=[(-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)],
            scale_range=[(-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)],
            shear_range=[(-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)],
            mode=("bilinear", "nearest", "nearest"),
            prob=0.30
        ),
        transforms.ToTensord(keys=["image", "mask"], allow_missing_keys=True),
        transforms.ToTensord(keys=["LP"], dtype=torch.float),
        transforms.ToTensord(keys=["label"], dtype=torch.long)

    ]
)

val_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image", "mask", "LP"], allow_missing_keys=True),
        transforms.AddChanneld(keys=["image", "mask", "LP"], allow_missing_keys=True),
        transforms.NormalizeIntensityd(keys=["image"]),
        transforms.ScaleIntensityd(keys=["image"],
                                   minv=0.0, maxv=1.0),
        transforms.ToTensord(keys=["image", "mask"], allow_missing_keys=True),
        transforms.ToTensord(keys=["LP"], dtype=torch.float, allow_missing_keys=True),
        transforms.ToTensord(keys=["label"], dtype=torch.long, allow_missing_keys=True)
    ]
)
