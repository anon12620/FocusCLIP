from .coco import COCODataset
from .mpii import MPIIDataset

__all__ = ['COCODataset', 'MPIIDataset']

available_datasets = {
    'coco': COCODataset,
    'mpii': MPIIDataset,
}


def get_dataset(name, **kwargs):
    if name not in available_datasets:
        raise ValueError(f"Invalid dataset name: {name}")
    return available_datasets[name](**kwargs)
