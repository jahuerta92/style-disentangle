import argparse
import os
import logging

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from model import SyntheticHardNegativesLossNonZeroSelfModel as Model
from data import make_data_loaders
from utils import set_global_logging_level

import datetime

set_global_logging_level(logging.ERROR, ["transformers", "torch"])

# Scripting the parameters
parser = argparse.ArgumentParser(description='Author style-semantics detangled model')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--minibatch_size', type=int, default=8, help='batch size')
parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
parser.add_argument('--num_epochs', type=int, default=1, help='number of epochs')
parser.add_argument('--warmup_steps', type=float, default=0.1, help='warmup steps')
parser.add_argument('--max_length', type=int, default=512, help='max length')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--name', type=str, default='debug-run', help='name of the run')
parser.add_argument('--unfrozen_layers', type=int, default=24, help='unfrozen layers')
parser.add_argument('--reset_layers', type=int, default=0, help='reset layers')
parser.add_argument('--eps', type=float, default=1, help='eps')
parser.add_argument('--gamma', type=float, default=1, help='gamma')
parser.add_argument('--with_weights', action='store_true', help='with weights')
parser.add_argument('--gpu', type=int, default=0, help='with weights')
parser.add_argument('--dataset', type=str, default='blog', help='dataset', choices=['blog', 'fanfic'])
parser.add_argument('--log_steps', type=str, default='max', help='dataset')

args = parser.parse_args()
PATH_DATA = {'blog':'data',
             'fanfic':'data/fanfic'}

def train(args, remove=False):
    # Define constant hparams
    BATCH_SIZE = args.batch_size
    MINIBATCH_SIZE = args.minibatch_size
    LEARNING_RATE = args.lr
    NUM_EPOCHS = args.num_epochs
    WARMUP_STEPS = args.warmup_steps
    DROPOUT = args.dropout
    MAX_LENGTH = args.max_length
    UNFROZEN_LAYERS = args.unfrozen_layers
    RESET_LAYERS = args.reset_layers
    EPS = args.eps
    GAMMA = args.gamma
    WITH_WEIGHTS = args.with_weights
    GPU = args.gpu
    DATASET = args.dataset
    
    # get time and date for model logging
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
    #filename = f'ckpt-{date}_{time}'
    type_train = 'hardnegs' if WITH_WEIGHTS else 'vanilla'
    filename = f'{type_train}-{DATASET}-{date}_{time}'

    # Create your dataset and dataloader
    train_dataset, valid_dataset, test_dataset = make_data_loaders(PATH_DATA[DATASET], 
                                                                   ['WhereIsAI/UAE-Large-V1', 
                                                                    'roberta-large'], 
                                                                   MAX_LENGTH, 
                                                                   BATCH_SIZE, 
                                                                   dataset=DATASET)


    # Create your model
    model = Model(training_steps=len(train_dataset)*NUM_EPOCHS,
                  eps=EPS,
                  gamma=GAMMA,
                  minibatch_size=MINIBATCH_SIZE,
                  warmup_steps=WARMUP_STEPS,
                  lr=LEARNING_RATE,
                  dropout=DROPOUT,
                  unfrozen_layers=UNFROZEN_LAYERS,
                  reset_layers=RESET_LAYERS,
                  with_weights=WITH_WEIGHTS,
                  )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(monitor='val/accuracy',
                                          dirpath='checkpoints/',
                                          filename=filename,
                                          mode='max',
                                          )
    wandb_logger = WandbLogger(project='detangle-style-content')
    wandb_logger.log_hyperparams(vars(args))

    # Define trainer
    log_steps = 1. if args.log_steps == 'max' else int(args.log_steps)
    trainer = Trainer(accelerator="gpu", devices=[GPU],
                    max_epochs=NUM_EPOCHS,
                    precision='16-mixed',
                    callbacks=[checkpoint_callback],
                    logger=wandb_logger,
                    val_check_interval=log_steps,
                    )
    
    trainer.fit(model, train_dataset, valid_dataset)
    
    #UNTESTED
    model =  Model.load_from_checkpoint(checkpoint_callback.best_model_path)
    test_metrics = trainer.test(model, test_dataset)[0]
    
    wandb_logger.log_metrics(test_metrics)
    wandb_logger.experiment.finish()
    if remove:
        os.remove(f'checkpoints/{filename}.ckpt')

    return checkpoint_callback.best_model_score

def main():
    train(args, remove=False)

if __name__ == '__main__':
    main()