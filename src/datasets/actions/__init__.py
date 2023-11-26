from .stanford40 import Stanford40Dataset

__all__ = ['Stanford40Dataset']

available_datasets = {
    'stanford40': Stanford40Dataset,
}


def get_dataset(name, **kwargs):
    if name not in available_datasets:
        raise ValueError(f"Invalid dataset name: {name}")
    return available_datasets[name](**kwargs)