import argparse
import yaml
from dotmap import DotMap
from os import path
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPPlugin
from models.lstur import LSTUR
from models.nrms import NRMS
from models.naml import NAML
from models.sentirec import SENTIREC
from models.robust_sentirec import ROBUST_SENTIREC
from data.dataset import BaseDataset


def cli_main():
    # ------------
    # args
    # ------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        action='store',
        dest='config',
        help='config.yaml',
        required=True)
    parser.add_argument(
        '--ckpt',
        action='store',
        dest='ckpt',
        help='checkpoint to load',
        required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
        config = DotMap(config)

    assert(config.name in ["lstur", "nrms", "naml", "sentirec", "robust_sentirec"])

    pl.seed_everything(1234)
    
    # ------------
    # logging
    # ------------
    logger = TensorBoardLogger(
        **config.logger
    )

    # ------------
    # data
    # ------------

    test_dataset = BaseDataset(
        path.join(config.test_dir, 'behaviors_parsed.tsv'),
        path.join(config.test_dir, 'news_parsed.tsv'),
        config)
    test_loader = DataLoader(
        test_dataset,
        **config.val_dataloader)
   
    #print(len(dataset), len(train_dataset), len(val_dataset))
    # ------------
    # init model
    # ------------
    try:
        pretrained_word_embedding = torch.from_numpy(
            np.load(path.join(config.train_dir, 'pretrained_word_embedding.npy'))
            ).float()
    except FileNotFoundError:
        pretrained_word_embedding = None
    
    if config.name == "lstur":
        model = LSTUR.load_from_checkpoint(
            args.ckpt, 
            config=config, 
            pretrained_word_embedding=pretrained_word_embedding)
    elif config.name == "nrms":
        model = NRMS.load_from_checkpoint(
            args.ckpt, 
            config=config, 
            pretrained_word_embedding=pretrained_word_embedding)
    elif config.name == "naml":
        model = NAML.load_from_checkpoint(
            args.ckpt, 
            config=config, 
            pretrained_word_embedding=pretrained_word_embedding)
    elif config.name == "sentirec":
        model = SENTIREC.load_from_checkpoint(
            args.ckpt, 
            config=config, 
            pretrained_word_embedding=pretrained_word_embedding)
    elif config.name == "robust_sentirec":
        model = ROBUST_SENTIREC.load_from_checkpoint(
            args.ckpt, 
            config=config, 
            pretrained_word_embedding=pretrained_word_embedding)
    # elif:
        # UPCOMING MODELS

    # ------------
    # Test
    # ------------
    trainer = Trainer(
        **config.trainer,
        logger=logger,
        plugins=DDPPlugin(find_unused_parameters=False)
    )

    trainer.test(
        model=model, 
        test_dataloaders=test_loader
    )
    # trainer.test()

if __name__ == '__main__':
    cli_main()