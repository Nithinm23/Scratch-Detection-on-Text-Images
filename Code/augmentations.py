import albumentations as A

def get_train_aug(size=256):
    return A.Compose([
        A.Resize(size, size),
        A.RandomBrightnessContrast(p=0.4),
        A.GaussianBlur(p=0.2),
        A.GaussNoise(p=0.15),
        A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05,
                           rotate_limit=6, p=0.3),
    ])

def get_val_aug(size=256):
    return A.Compose([
        A.Resize(size, size)
    ])
