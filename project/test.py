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
from models.naml_simple import NAML_Simple
from models.sentirec import SENTIREC
from models.robust_sentirec import ROBUST_SENTIREC
from data.dataset import BaseDataset
from tqdm import tqdm


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

    assert(config.name in ["lstur", "nrms", "naml", "naml_simple", "sentirec", "robust_sentirec"])

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
        path.join(config.test_behavior),
        path.join(config.test_news), 
        config)
    test_loader = DataLoader(
        test_dataset,
        **config.test_dataloader)
   
    #print(len(dataset), len(train_dataset), len(val_dataset))
    # ------------
    # init model
    # ------------
    # ------------
    # init model
    # ------------
    # load embedding pre-trained embedding weights
    embedding_weights=[]
    with open(config.embedding_weights, 'r') as file: 
        lines = file.readlines()
        for line in tqdm(lines):
            weights = [float(w) for w in line.split(" ")]
            embedding_weights.append(weights)
    pretrained_word_embedding = torch.from_numpy(
        np.array(embedding_weights, dtype=np.float32)
        )

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
    elif config.name == "naml_simple":
        model = NAML_Simple.load_from_checkpoint(
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