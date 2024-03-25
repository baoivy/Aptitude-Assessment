from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import numpy as np
import torch
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from utils import preprocess_text
import json

class ReviewDataset(Dataset):
    def __init__(self, args) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.path = args.input_file
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.predict = args.predict
        self.data = self.load_data()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        return item
    
    def load_data(self):
        with open(self.path, encoding='utf-8') as f:
            loader = json.load(f)

        data_loader = []
        for element in loader:
            tokenized = self.tokenize(preprocess_text(element['title']), preprocess_text(element['review']))
            if self.predict == False:
                entry = {
                    "tokenized": tokenized,
                    "label" : int(float(element['score'])) - 1
                }
            else:
                entry = {
                    "tokenized": tokenized
                }
            data_loader.append(entry)
            
        return data_loader
    
    def tokenize(self, sent1, sent2):
        tokenizer = self.tokenizer.encode_plus(
            sent1, sent2,
			add_special_tokens=True,
			truncation=True,
            max_length=256,
			#return_tensors='pt',
			padding='max_length'
        )
        tokenizer = {key: torch.tensor(tensor).to(self.device) for key, tensor in tokenizer.items()}
        return tokenizer
    
class LoadDataset(pl.LightningDataModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.pin_memory = True if self.num_workers > 0 else False
        self.args = args
        self.all_data = ReviewDataset(args=self.args)

    def setup(self, stage=None) -> None :
        self.train_ds, self.valid_ds = train_test_split(self.all_data, test_size=0.1, random_state=42) 
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_ds,
			#num_workers=self.num_workers,
			batch_size=self.batch_size,
			shuffle=True
			#pin_memory=self.pin_memory
        )
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader (
            self.valid_ds,
            #num_workers=self.num_workers,
			batch_size=1,
			shuffle=False
			#pin_memory=self.pin_memory
        )
    
    def len_trainset(self):
        return int(len(self.all_data) * 0.9)
    
    def weight_label(self):
        labels = [x["label"] for x in self.train_ds]
        classes = np.unique(labels)
        class_weights = compute_class_weight(
            "balanced", classes=np.array(classes), y=labels
        )
        return torch.tensor(class_weights, dtype=torch.float).to(device=self.device)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass