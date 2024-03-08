import torch
import logging
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from src.preprocessing.make_dataset import SplitWordsDataset
from typing import Dict
from torch.nn.functional import cross_entropy
from transformers import AdamW
from torcheval.metrics.functional import multiclass_f1_score, multiclass_confusion_matrix, binary_f1_score
from copy import deepcopy
from transformers import AutoModelForSequenceClassification

class SequenceClassificationTrainer:
    def __init__(
        self, 
        model : AutoModelForSequenceClassification,
        num_classes : int,
        device : str = 'cuda',
    ) -> None:
        
        self.model = model
        self.num_classes = num_classes

        self.device = device

    def compile(
        self,
        train_dataloader : DataLoader,
        val_dataloader : DataLoader,
        n_epochs : int = 10,
        lr : float = 1e-5,
        loss_fn : torch.nn.Module = None,
        evaluation_fn : callable = multiclass_f1_score, 
    ):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr = lr
        )

        self.history = {
            'val_f1' : [],
            'conf_matrix': None
        }

        best_f1 = 0
        self.best_model = None

        logging.info(f"Training model with {len(train_dataloader.dataset)//4} shots")

        for epoch in range(n_epochs):
            for batch in tqdm(train_dataloader):

                for k,v in batch.items():
                    batch[k] = v.to(self.device)

                optimizer.zero_grad()

                output = self.model(**batch)

                if loss_fn is None:
                    loss = output['loss']

                else:
                    loss = loss_fn(
                        output['logits'],
                        batch['labels']
                    )

                loss.backward()
                optimizer.step()


            y_true_val = torch.tensor([],device=self.device)
            y_pred_val = torch.tensor([],device=self.device)
            
            for batch_val in val_dataloader:

                for k,v in batch_val.items():
                    batch_val[k] = v.to(self.device)

                with torch.no_grad():
                    out = self.model(**batch_val)

                    y_pred = out.logits.softmax(dim = 1)
                    y_true = batch_val['labels']

                    y_pred_val = torch.cat([
                        y_pred_val, 
                        torch.tensor(y_pred).to(self.device)
                    ])

                    y_true_val = torch.cat([
                        y_true_val, 
                        torch.tensor(y_true).to(self.device)
                    ])

            if evaluation_fn == multiclass_f1_score:
                
                f1 = evaluation_fn(
                    y_pred_val.argmax(dim = 1).to(torch.int64),
                    y_true_val.to(torch.int64),
                    num_classes= self.num_classes
                )

            else:
                f1 = evaluation_fn(
                    y_pred_val.argmax(dim = 1).to(torch.int64),
                    y_true_val.to(torch.int64),
                )

                print(binary_f1_score(
                    y_pred_val.argmax(dim = 1).to(torch.int64),
                    y_true_val.to(torch.int64),
                ))
                print(f1)

            conf_matrix = multiclass_confusion_matrix(
                y_pred_val.argmax(dim = 1).to(torch.int64), 
                y_true_val.to(torch.int64),
                num_classes=self.num_classes
            )

            logging.info(f"f1: {f1}")
            logging.info(conf_matrix)

            self.history['val_f1'].append(f1.item())

            if f1 > best_f1:
                best_f1 = f1
                self.best_model = deepcopy(self.model)
                self.history['val_conf_matrix'] = conf_matrix
                self.history['y_pred'] = y_pred_val.argmax(dim = 1).to(torch.int64)
                self.history['y_true'] = y_true_val.to(torch.int64)
        
        return self.history

        