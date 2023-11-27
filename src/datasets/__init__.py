from .evaluation import *
from .training import *

__all__ = ['get_dataset']

available_datasets = {
    # Training datasets
    'multimodal_mpii': MultimodalMPIIDataset,

    # Evaluation datasets
    'emotic': EmoticDataset,
    'fairface': FairFaceDataset,
    'fer2013': FER2013Dataset,
    'ferplus': FERPlusDataset,
    'kinetics': KineticsDataset,
    'lagenda': LAGENDADataset,
    'stanford40': Stanford40Dataset,
    'utkface': UTKFaceDataset,
}


def get_dataset(dataset, **kwargs):
    """ Factory function for creating datasets. """
    if dataset not in available_datasets:
        raise ValueError(f'Dataset {dataset} is not supported')

    return available_datasets[dataset](**kwargs)
