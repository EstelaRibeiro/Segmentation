import albumentations as A

def get_train_augmentation():

    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.9, 1.1),
                 translate_percent=(-0.1, 0.1),
                 rotate=(-8, 8),
                 p=0.5)
    ])

def get_val_test_augmentation():
    return A.Compose([])
