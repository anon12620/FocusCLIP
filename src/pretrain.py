import argparse
import logging
import os

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import get_dataset
from tasks.alignment import AlignmentTask
from utils import cfg, update_config

logger = logging.getLogger()


class PretrainingData(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.DATASET
        self.shape = cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1]
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.num_workers = cfg.WORKERS

    def setup(self, stage=None):
        # Define dataset args and transforms
        kwargs = self.cfg.KWARGS
        kwargs['transforms'] = transforms.Compose([
            transforms.Resize(self.shape),
            transforms.ToTensor(),
        ])

        # Create training and validation sets
        logger.debug(f'Dataset: {self.cfg.NAME}')
        logger.debug(f'Image size: {self.shape}')
        logger.debug(f'Num workers: {self.num_workers}')
        self.trainset = get_dataset(self.cfg.NAME, split='train', **kwargs)
        self.valset = get_dataset(self.cfg.NAME, split='validation', **kwargs)

    def create_dataloader(self, dataset, shuffle):
        # Update batch size based on number of views in the dataset
        batch_size = self.batch_size
        if hasattr(dataset, 'num_views') and dataset.num_views > 1:
            logger.debug(f'Num augmented views: {dataset.num_views}')
            batch_size = batch_size // dataset.num_views
            logger.debug(f'Effective batch size: {batch_size}')

        kwargs = {
            'batch_size': batch_size,
            'num_workers': self.num_workers,
            'shuffle': shuffle,
        }

        # If dataset a custom collate_fn, use that
        if hasattr(dataset, 'collate_fn'):
            kwargs['collate_fn'] = dataset.collate_fn
            logger.debug(f'Using custom collate function')

        return DataLoader(dataset, **kwargs)

    def train_dataloader(self):
        logger.debug(f'Creating training data loader')
        return self.create_dataloader(self.trainset, shuffle=True)

    def val_dataloader(self):
        logger.debug(f'Creating validation data loader')
        return self.create_dataloader(self.valset, shuffle=False)


def parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--cfg', type=str,
                       help='Path to the config file.')
    group.add_argument('--ckpt', type=str, default=None,
                       help='Path to the checkpoint file to resume from.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='Modify config options using the command-line')
    return parser.parse_args()


def main(config, ckpt_path=None):
    pl.seed_everything(42, workers=True)

    num_gpus = config.GPUS if torch.cuda.is_available() else 0
    accelerator = 'gpu' if num_gpus > 0 else 'cpu'

    trainer = pl.Trainer(
        default_root_dir=config.OUTPUT_DIR,
        max_epochs=config.TRAIN.MAX_EPOCHS,
        accelerator=accelerator,
        devices=num_gpus,
        gradient_clip_val=1.0,
        deterministic=True,
    )

    dm = PretrainingData(config)
    model = AlignmentTask(config)
    trainer.fit(model, dm, ckpt_path=ckpt_path)


if __name__ == '__main__':
    args = parse_args()
    if args.ckpt is not None:
        model = AlignmentTask.load_from_checkpoint(args.ckpt)
        cfg_name = args.ckpt.split('/')[-5]
        main(model.hparams.cfg, ckpt_path=args.ckpt)
    else:
        cfg_name = os.path.basename(args.cfg).split('.')[0]
        update_config(cfg, args)
        cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, cfg_name)
        main(cfg, ckpt_path=args.ckpt)
