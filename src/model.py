from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig, AutoModel
import pytorch_lightning as pl
from loss import SupConLoss
import math

class ScoreReviewModel(pl.LightningModule):
    def __init__(self, args, steps_per_epoch=0 ,loss_fct=None) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.config = AutoConfig.from_pretrained(args.model, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(args.model, config=self.config)
        #Loss initialize
        self.loss_ce = torch.nn.CrossEntropyLoss(weight=loss_fct)
        self.loss_scl = SupConLoss() if args.has_loss_scl else 0
        #GRU initialize
        self.gru1 = torch.nn.GRU(
            input_size=self.config.hidden_size,
            hidden_size=args.hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        #Parameter initialize
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.alpha = args.alpha
        self.epochs = args.epoch
        self.batch_size = args.batch_size
        self.gra_accum = args.gradient_accumulations
        #self.device = device #if torch.cuda.is_available() else "cpu"
        self.steps_per_epoch =  math.ceil(
            (steps_per_epoch / self.batch_size) / self.gra_accum
        )
        #Feedforward
        self.dropout = torch.nn.Dropout(p=0.1)
        self.classifier = torch.nn.Linear(args.hidden_size*2, 5)
        self.softmax = torch.nn.Softmax(dim=1)

    def configure_optimizers(self) -> torch.Any:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=self.lr,
            pct_start=0.05,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.epochs,
            anneal_strategy="linear",
        )

        scheduler_dict = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler_dict] 
    

    def forward(self, tokenized):
        bert_output = self.model(**tokenized)
        hidden_state = bert_output.last_hidden_state
        out_gru1, _ = self.gru1(hidden_state)
        out_gru1 = self.dropout(out_gru1)
        output = self.classifier(out_gru1[:, -1, :])
        return output, out_gru1[:, -1, :]

    def training_step(self, train_batch, batch_idx) -> torch.Tensor | torch.Dict[str, torch.Any]:
        input, verdict = train_batch['tokenized'], train_batch['label']
        output, output_head = self.forward(input)
        loss_cea = self.loss_ce(output, verdict)
        loss_scl = 0
        #self.loss_scl(output_head, verdict)
        train_loss = loss_cea + self.alpha*loss_scl 
        self.log("train_loss", train_loss, on_epoch=True, prog_bar=True, logger=True)
        return train_loss
    
    def validation_step(self, valid_batch, batch_idx) -> None:
        input, verdict = valid_batch['tokenized'], valid_batch['label']
        output, output_head = self.forward(input)
        output = self.softmax(output)
        loss_cea = self.loss_ce(output, verdict)
        loss_scl = 0
        #self.loss_scl(output_head, verdict)
        valid_loss = loss_cea + self.alpha*loss_scl 
        accuracy = torch.sum(torch.argmax(output) == verdict).item() / (len(verdict) * 1.0)
        self.log("valid_loss", valid_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_accuracy", accuracy, on_epoch=True, prog_bar=True, logger=True)
    
    def unbatch_result(self, test_batch):
        result = []
        for output in test_batch:
            result.append(output)
        return result
    
    def predict_step(self,  test_batch, batch_idx: int, dataloader_idx: int = 0) -> torch.Any:
        input = test_batch['tokenized']
        with torch.no_grad():
            output, _ = self.forward(input)
            res = self.softmax(output)
            res = torch.argmax(res, dim=-1)
        return self.unbatch_result(res)