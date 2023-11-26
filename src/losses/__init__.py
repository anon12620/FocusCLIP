from .ntxent import NTXentLoss

__all__ = [
    "NTXentLoss",
]

available_losses = {
    "ntxent": NTXentLoss,
}


def get_loss(name, **kwargs):
    if name not in available_losses:
        raise ValueError(f"Loss {name} not available. "
                         f"Available losses: {available_losses.keys()}")

    return available_losses[name](**kwargs)
