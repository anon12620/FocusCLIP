from .actions import get_dataset as get_actions_dataset
from .emotic import EmoticDataset
from .fer import FER
from .ferplus import FERPlus
from .keypoints import get_dataset as get_keypoints_dataset
from .multimodal_mpii import MultiModalMPII
from .face import FairFace, UTKFace, LAGENDA
from .kinetics import KineticsDataset

__all__ = ['get_dataset', 'get_actions_dataset', 'get_keypoints_dataset']

available_datasets = {
    'keypoints/mpii': get_keypoints_dataset,
    'action/stanford40': get_actions_dataset,
    'emotic': EmoticDataset,
    'fer': FER,
    'ferplus': FERPlus,
    'multimodal_mpii': MultiModalMPII,
    'fairface': FairFace,
    'utk': UTKFace,
    'lagenda': LAGENDA,
    'kinetics': KineticsDataset,
}


def get_dataset(dataset, **kwargs):
    if dataset not in available_datasets:
        raise ValueError(f'Dataset {dataset} is not supported')

    if dataset.count('/') == 1:
        dataset_name = dataset.split('/')[1]
        return available_datasets[dataset](dataset_name, **kwargs)

    return available_datasets[dataset](**kwargs)
