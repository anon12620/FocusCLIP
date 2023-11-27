import yaml
from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.OUTPUT_DIR = 'logs/'
__C.GPUS = 1
__C.WORKERS = 8

# Common parameters for the network
__C.MODEL = edict()
__C.MODEL.NAME = 'focusclip'
__C.MODEL.KWARGS = {
    'visual_encoder_name': 'vit_base_patch16_224',
    'text_encoder_name': 'bert-base-uncased',
    'tokenizer_name': 'bert-base-uncased',
    'context_length': 512,
    'triple_components': True,
    'pretrained': True
}
__C.MODEL.IMAGE_SIZE = [224, 224]

# Dataset parameters
__C.DATASET = edict()
__C.DATASET.NAME = 'multimodal_mpii'
__C.DATASET.KWARGS = {
    'root': 'data/mpii',
    'config_name': ['gpt-4'] # 'gpt3.5-turbo', 'llama-2', 'gpt3.5-turbo-legacy'
}

# Loss parameters
__C.LOSS = edict()
__C.LOSS.NAME = 'ntxent'
__C.LOSS.KWARGS = {
    'temperature': 0.5,
    'learn_temperature': False
}

# Training parameters
__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = 32
__C.TRAIN.MAX_EPOCHS = 64

__C.TRAIN.OPTIMIZER = 'sgd'
__C.TRAIN.LR = 0.001
__C.TRAIN.MOMENTUM = 0.9


def update_config(cfg, args):
    """
    Update the configuration dictionary with command line arguments.

    This function allows for two methods of updating the configuration dictionary:
    1. By specifying a YAML file with the '--cfg' argument, which will load and merge 
       the contents of the file into the configuration dictionary.
    2. By providing key-value pairs with the '--opts' argument, which will update 
       specific keys in the configuration dictionary with the provided values.

    The '--opts' argument should be a list of key-value pairs, where the keys are 
    strings representing valid keys in the configuration dictionary, and the values 
    are the new values for those keys. The values will be cast to the type of the 
    existing values in the configuration dictionary.

    Args:
        cfg (dict): The configuration dictionary to be updated.
        args (argparse.Namespace): Command line arguments containing the '--cfg' 
                                   and/or '--opts' arguments.

    Raises:
        ValueError: If the type of a config variable specified in the '--opts' 
                    argument is invalid.
        KeyError: If a key specified in the '--opts' argument is not found in the 
                  configuration dictionary.
    """

    if hasattr(args, 'cfg') and args.cfg is not None:
        with open(args.cfg) as f:
            exp_cfg = edict(yaml.safe_load(f))

        for k, v in exp_cfg.items():
            cfg[k] = v

    if hasattr(args, 'opts') and args.opts is not None:
        for k, v in zip(args.opts[0::2], args.opts[1::2]):
            key_list = k.split('.')
            d = cfg
            for subkey in key_list[:-1]:
                if subkey not in d:
                    raise KeyError("Invalid key in config file!")
                d = d[subkey]
            subkey = key_list[-1]
            assert subkey in d, f"{subkey} is not in {d.keys()}"
            # Coerce the types to be the same
            old_val = d[subkey]
            if isinstance(old_val, int):
                val = int(v)
            elif isinstance(old_val, float):
                val = float(v)
            elif isinstance(old_val, str):
                val = v
            else:
                raise ValueError("Invalid type of config variable!")
            d[subkey] = val
