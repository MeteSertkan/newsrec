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
        '--resume',
        action='store',
        dest='resume',
        help='resume training form ckpt',
        required=False)
    args = parser.parse_args()
    
    with open(args.config, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
        config = DotMap(config)

    assert(config.name in ["lstur", "nrms", "naml", "sentirec", "robust_sentirec"])

    pl.seed_everything(1234)
    
    # ------------
    # init callbacks & logging
    # ------------
    checkpoint_callback = ModelCheckpoint(
        **config.checkpoint
    )
    logger = TensorBoardLogger(
        **config.logger
    )

    # ------------
    # data
    # ------------
    dataset = BaseDataset(
        path.join(config.train_dir, 'behaviors_parsed.tsv'),
        path.join(config.train_dir, 'news_parsed.tsv'), 
        config) 
    train_len = int(config.train_val_split*len(dataset))
    # train val split
    val_len = len(dataset) - train_len
    train_dataset, val_dataset= torch.utils.data.random_split(dataset, [train_len, val_len])
    
    train_loader = DataLoader(
        train_dataset,
        **config.train_dataloader)
    val_loader = DataLoader(
        val_dataset,
        **config.val_dataloader)
   
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
        model = LSTUR(config, pretrained_word_embedding)
    elif config.name == "nrms":
        model = NRMS(config, pretrained_word_embedding)
    elif config.name == "naml":
        model = NAML(config, pretrained_word_embedding)
    elif config.name == "sentirec":
        model = SENTIREC(config, pretrained_word_embedding)
    elif config.name == "robust_sentirec":
        model = ROBUST_SENTIREC(config, pretrained_word_embedding)
    # elif:
        # UPCOMING MODELS

    # ------------
    # training
    # ------------
    early_stop_callback = EarlyStopping(
       **config.early_stop
    )
    if args.resume is not None:
        model = model.load_from_checkpoint(
            args.resume, 
            config=config, 
            pretrained_word_embedding=pretrained_word_embedding)
        trainer = Trainer(
            **config.trainer,
            callbacks=[early_stop_callback, checkpoint_callback],
            logger=logger,
            plugins=DDPPlugin(find_unused_parameters=False), 
            resume_from_checkpoint=args.resume
        )
    else:
        trainer = Trainer(
            **config.trainer,
            callbacks=[early_stop_callback, checkpoint_callback],
            logger=logger,
            plugins=DDPPlugin(find_unused_parameters=False)
        )
    trainer.fit(
        model=model, 
        train_dataloader=train_loader, 
        val_dataloaders=val_loader)

if __name__ == '__main__':
    cli_main()