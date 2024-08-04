import torch
import torchvision
import numpy as np


def load_mnist(exclude_class=9, test_set=False, transform=None):
    if transform is None:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset = torchvision.datasets.MNIST(root='./data', train=not test_set, download=True,
                                         transform=transform)
    # Get all targets
    targets = dataset.targets
    # Create target_indices
    target_indices = np.arange(len(targets))
    # Get indices to keep from train split
    idx_to_keep = targets != exclude_class
    # Only keep your desired classes
    idxs = target_indices[idx_to_keep]
    # create subset without the excluded class samples
    return torch.utils.data.Subset(dataset, idxs)


class TransformDataLoader(torch.utils.data.DataLoader):
    """
    A DataLoader wrapper that applies a transform function to each batch.

    Args:
        dataloader (DataLoader): The original DataLoader.
        transform (callable): The transformation function to apply to each batch.
    """

    def __init__(self, dataloader, transform):
        super().__init__(dataloader.dataset,
                         batch_size=dataloader.batch_size,
                         num_workers=dataloader.num_workers,
                         pin_memory=dataloader.pin_memory)
        self.transform = transform

    def __iter__(self):
        for batch in super().__iter__():
            yield self.transform(batch)
