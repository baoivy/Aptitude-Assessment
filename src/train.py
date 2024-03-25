import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import callbacks
import argparse
from pytorch_lightning.strategies import DDPStrategy
from model import ScoreReviewModel
from preprocessing import LoadDataset
import math


def get_parse():
    parser = argparse.ArgumentParser(description="Run training.")
    parser.add_argument("--input_file", type=str, default='data/train_dataset.json')
    parser.add_argument("--model", type=str, default="xlm-roberta-base")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--epoch", type=int, default=2)
    parser.add_argument("--frac_warmup", type=float, default=0.1,
                            help="The fraction of training to use for warmup.")
    parser.add_argument("--scheduler_total_epochs", default=None, type=int,
                            help="If given, pass as total # epochs to LR scheduler.")
    parser.add_argument("--gradient_accumulations", default=8, type=int)
    parser.add_argument("--accelerator", type=str, default="ddp")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--precision", default='16-mixed', type=str)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--starting_checkpoint", type=str, default=None)
    parser.add_argument("--monitor", type=str, default="valid_accuracy")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--num_label", default=5, type=int)
    parser.add_argument("--debug_mode", type=bool, default=True)
    parser.add_argument("--predict", default=False, type=bool)
    parser.add_argument("--has_loss_scl", default=True, type=bool)
    args = parser.parse_args()
    return args


def main():
    pl.seed_everything(76)
    args = get_parse()

    lr_callback = callbacks.LearningRateMonitor(logging_interval="step")
    gpu_callback = callbacks.DeviceStatsMonitor()
    checkpoint_callback = callbacks.ModelCheckpoint(
        monitor=args.monitor, 
        mode="max", 
        save_top_k=1, 
        save_last=False,
    )

    #Load data
    dataset = LoadDataset(args)
    dataset.setup()
    
    if args.starting_checkpoint is None:
        model = ScoreReviewModel(
            args=args, 
            loss_fct=dataset.weight_label(),
            steps_per_epoch=dataset.len_trainset()
        )
    else:
        model = ScoreReviewModel.load_from_checkpoint(checkpoint_path=args.starting_checkpoint)
    
    model.train()
    trainer = pl.Trainer(
        callbacks=[lr_callback, checkpoint_callback, gpu_callback], 
        precision=args.precision,
        max_epochs=args.epoch,
        accumulate_grad_batches=args.gradient_accumulations,
    )
    
    trainer.fit(model, datamodule=dataset)


if __name__ == '__main__':
    main()