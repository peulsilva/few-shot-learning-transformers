import logging
import torch
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoModelForTokenClassification
from copy import deepcopy
from torcheval.metrics.functional import multiclass_f1_score, multiclass_confusion_matrix

class BioTrainer:

    def __init__(
        self,
        model : AutoModelForTokenClassification,
        optimizer : torch.optim,
        device : str = "cuda"
    ) -> None:
        self.model = model
        self.device = device
        self.optimizer = optimizer 


    def compile(
        self,
        train_dataloader : Dataset,
        validation_dataloader : Dataset,
        n_epochs : int = 20
    ):
        self.history = []
        self.model.train()
        best_f1 = 0
        self.best_model = None

        for epoch in tqdm(range(n_epochs)):
            for batch in (train_dataloader):
                for k,v in batch.items():
                    batch[k] = v.to(self.device)

                    if k == "bbox":
                        continue
                    batch[k] = batch[k].reshape(1,512)
                batch.pop('bbox')

                self.optimizer.zero_grad()

                out = self.model(**batch)
                loss = out.loss

                loss.backward()
                self.optimizer.step()

            y_pred_val = torch.tensor([],device=self.device)
            y_true_val = torch.tensor([],device=self.device)

            for batch in validation_dataloader:
                for k,v in batch.items():
                    batch[k] = v.to(self.device)

                    if k == "bbox":
                        continue
                    batch[k] = batch[k].reshape(4,512)
                batch.pop('bbox')

                y_true = batch['labels']
                mask = y_true!= -100

                with torch.no_grad():
                    y_pred = self.model(**batch).logits[:,:,1]
                y_pred = y_pred[mask]\
                    .reshape(4,-1)[:,1:]\
                    .argmax(dim = 0)

                y_true = y_true[mask]\
                    .reshape(4,-1)[:, 1:]\
                    .argmax(dim = 0)
                
                y_pred_val = torch.cat([y_pred, y_pred_val])
                y_true_val = torch.cat([y_true, y_true_val])

            f1 = multiclass_f1_score(
                y_pred_val,
                y_true_val,
                num_classes=4
            )

            self.history.append(f1.item())

            conf_matrix = multiclass_confusion_matrix(
                y_true_val.to(torch.int64),
                y_pred_val.to(torch.int64),
                num_classes= 4
            )

            if f1 > best_f1:
                best_f1 = f1
                self.best_model = deepcopy(self.model)

            logging.info(f"f1 score: {f1}")
            logging.info(conf_matrix)

        return self.history