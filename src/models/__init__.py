import torch

from .clip import CLIP
from .focusclip import FocusCLIP

__all__ = ["CLIP", "FocusCLIP", "get_model", "build_from_cfg"]

available_models = {
    "clip": CLIP,
    "focusclip": FocusCLIP,
}


def get_model(model_name: str, weights=None, lightning_weights=None, **kwargs):
    """Returns a model instance of the specified model name."""
    if model_name not in available_models:
        raise ValueError(f"Model {model_name} not available. "
                         f"Available models: {available_models.keys()}")

    model = available_models[model_name](**kwargs)
    if weights is not None:
        model.load_state_dict(torch.load(weights))

    if lightning_weights is not None:
        weights = torch.load(lightning_weights)['state_dict']
        weights_ = {}
        for k, v in list(weights.items()):
            k = k.replace('model.', '')
            weights_[k] = v

        model.load_state_dict(weights_, strict=False)

    return model


def build_from_cfg(cfg, weights=None, lightning_weights=None):
    """Builds a model from a config file."""
    return get_model(cfg.MODEL.NAME, weights=weights,
                     lightning_weights=lightning_weights,
                     **cfg.MODEL.KWARGS)
