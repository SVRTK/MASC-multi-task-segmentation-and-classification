from monai import transforms
from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
)
import torch
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np

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


def get_dataloaders(config):
    datasets = config['data_dir'] + config['data_JSON_file']

    datalist = load_decathlon_datalist(datasets, True, "training")
    val_files = load_decathlon_datalist(datasets, True, "validation")
    test_files = load_decathlon_datalist(datasets, True, "testing")
    infer_ds = None
    infer_files = None
    if config['infer']:
        infer_files = load_decathlon_datalist(datasets, True, "inference")
        infer_ds = CacheDataset(
            data=infer_files, transform=val_transforms, cache_num=20, cache_rate=1.0, num_workers=4
        )

    train_ds = CacheDataset(
        data=datalist, transform=train_transforms, cache_num=70, cache_rate=1.0, num_workers=8,
    )

    val_ds = CacheDataset(
        data=val_files, transform=val_transforms, cache_num=20, cache_rate=1.0, num_workers=4
    )

    test_ds = CacheDataset(
        data=test_files, transform=val_transforms, cache_num=20, cache_rate=1.0, num_workers=4
    )

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  #
    #    Sampling conditions with equal probability (accounting for class imbalance) #
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>s>>>>>>>>>>>>>>>>>>>>>>> #
    diagnosis = np.arange(config['N_diagnosis'])
    count0 = sum([datalist[x]["label"] == diagnosis[0] for x in range(len(datalist))])
    count1 = sum([datalist[x]["label"] == diagnosis[1] for x in range(len(datalist))])
    count2 = sum([datalist[x]["label"] == diagnosis[2] for x in range(len(datalist))])

    class_sample_count = np.array([count0, count1, count2])
    weight = 1. / class_sample_count
    samples_weight = []

    for x in range(len(datalist)):
        samples_weight.append(weight[int(datalist[x]["label"])])
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    train_loader = DataLoader(
        train_ds,
        batch_size=config['batch_size'],
        shuffle=False,
        sampler=sampler,
        num_workers=4,
        pin_memory=False  # True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )

    return train_loader, val_loader, test_ds, test_files, infer_ds, infer_files
