from pathlib import Path

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import LungDataset


def get_image_mask_pairs(image_dir, mask_dir):

    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    images = sorted(
        image_dir.glob("*.png")
    )

    pairs = []

    for image_path in images:

        mask_path = mask_dir / image_path.name

        if mask_path.exists():
            pairs.append(
                (image_path, mask_path)
            )

    return pairs


def create_datasets(
    image_dir,
    mask_dir,
    test_size=0.10,
    val_size=0.10,
    random_state=42,
    train_transform=None,
    val_transform=None,
    test_transform=None
):

    pairs = get_image_mask_pairs(
        image_dir,
        mask_dir
    )

    print(f"Total pairs: {len(pairs)}")

    # --------------------------------------------------------
    # First split: train + temporary
    # --------------------------------------------------------

    train_pairs, temp_pairs = train_test_split(
        pairs,
        test_size=test_size + val_size,
        random_state=random_state
    )

    # --------------------------------------------------------
    # Second split: validation + test
    # --------------------------------------------------------

    relative_test_size = test_size / (test_size + val_size)

    val_pairs, test_pairs = train_test_split(
        temp_pairs,
        test_size=relative_test_size,
        random_state=random_state
    )

    print("\nDataset split:")
    print(f"Train:      {len(train_pairs)}")
    print(f"Validation: {len(val_pairs)}")
    print(f"Test:       {len(test_pairs)}")

    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    train_dataset = LungDataset(train_pairs, transform=train_transform)
    val_dataset = LungDataset(val_pairs, transform=val_transform)
    test_dataset = LungDataset(test_pairs, transform=test_transform)

    return (
        train_dataset,
        val_dataset,
        test_dataset
    )


def create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size=8,
    num_workers=0
):

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )