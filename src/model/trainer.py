import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
import logging
import numpy as np
from torcheval.metrics.functional import multiclass_f1_score
from abc import ABC, abstractmethod
from copy import deepcopy

class BaseTrainer(ABC):
    def __init__(self,
                 model,
                 optimizer = AdamW) -> None:
        self.model = model
        self.optimizer = optimizer
        self.best_model = None
        self.best_f1 = 0

    @abstractmethod
    def compile(
        self,
        train_dataloader : DataLoader,
        validation_dataloader : DataLoader,
        n_classes : int,
        device :str = 'cpu', 
        num_epochs : int = 10,
        lr : float = 5e-5
    ) -> None:
        """Trains a model on train dataloader and evaluates it 
        on validation dataloader.

        Args:
            train_dataloader (DataLoader): train data
            validation_dataloader (DataLoader): validation data
            n_classes (int): number of classes
            device (str, optional): Device to train ("cuda" or "cpu"). Defaults to 'cpu'.
            num_epochs (int, optional): Number of epochs. Defaults to 10.
            lr (float, optional): Learning rate. Defaults to 5e-5.
        """        
        ...

    def save_best_model(
        self,
        y_pred : torch.Tensor,
        y_true : torch.Tensor,
        n_classes : int,
    ):
        """Evaluates model based on f1-score and saves best model

        Args:
            y_pred (torch.Tensor): validation predictions
            y_true (torch.Tensor): validation targets
            n_classes (int): number of classes
        """        
        f1 = multiclass_f1_score(
            y_pred,
            y_true,
            num_classes=n_classes
        )

        if f1 > self.best_f1:
            self.best_f1 = f1
            self.best_model = deepcopy(self.model)

class LayoutLMTrainer(BaseTrainer):
    def __init__(self, model, optimizer=AdamW) -> None:
        super().__init__(model, optimizer)
        self.model_name = "layoutlm v1"

    def compile(self, 
                train_dataloader: DataLoader, 
                validation_dataloader: DataLoader, 
                n_classes: int, 
                device: str = 'cpu', 
                num_epochs: int = 10, 
                lr: float = 0.00005
            ) -> None:
        optimizer = self.optimizer(self.model.parameters(), lr=lr)
        self.model.to(device)

        logging.info('''
            Starting model training
            -----------------------       
        ''')
        self.model.train()
        self.history = {
            "train_accuracy": [],
            "train-f1": [],
            "validation-accuracy": [],
            "validation-f1": []
        }

        for epoch in tqdm(range(num_epochs)):
            logging.info(f"Epoch: {epoch}")
            
            y_true_train = torch.tensor([],device=device)
            y_pred_train = torch.tensor([],device=device)
            y_pred_val = torch.tensor([],device=device)
            y_true_val = torch.tensor([],device=device)
            for X in (train_dataloader):
                input_ids = X["input_ids"]\
                    .to(device)\
                    .squeeze()
                
                bbox = X["bbox"]\
                    .to(device)\
                    .squeeze()

                attention_mask = X["attention_mask"]\
                    .to(device)\
                    .squeeze()
                token_type_ids = X["token_type_ids"]\
                    .to(device)\
                    .squeeze()
                
                labels = X["labels"]\
                    .to(device)\
                    .squeeze()
                
                if input_ids.shape[0] == 512:
                    continue

               
                outputs = self.model(
                    input_ids=input_ids, 
                    bbox=bbox, 
                    attention_mask=attention_mask, 
                    token_type_ids=token_type_ids,
                    labels=labels
                )
                
                loss = outputs.loss
                predictions = outputs.logits.argmax(-1)

                valid_outputs_mask = labels != -100

                y_pred = predictions[valid_outputs_mask].to(device)
                y_true = labels[valid_outputs_mask].to(device)

                y_pred_train = torch.cat([y_pred, y_pred_train])
                y_true_train = torch.cat([y_true, y_true_train])

                # backward pass to get the gradients 
                loss.backward()

                # update
                optimizer.step()
                optimizer.zero_grad()

            with torch.no_grad():
                for X_validation in validation_dataloader:
                    input_ids = X_validation["input_ids"]\
                        .to(device)\
                        .squeeze()
                
                    bbox = X_validation["bbox"]\
                        .to(device)\
                        .squeeze()

                    attention_mask = X_validation["attention_mask"]\
                        .to(device)\
                        .squeeze()
                    token_type_ids = X_validation["token_type_ids"]\
                        .to(device)\
                        .squeeze()
                    
                    labels = X_validation["labels"]\
                        .to(device)\
                        .squeeze()

                    
                    outputs = self.model(
                        input_ids=input_ids.reshape(1,-1), 
                        bbox= bbox.reshape([1, 512, 4]),
                        attention_mask=attention_mask.reshape(1,-1), 
                        token_type_ids=token_type_ids.reshape(1,-1),
                        labels=labels.reshape(1,-1)
                    )

                    loss = outputs.loss
                    predictions = outputs\
                        .logits\
                        .argmax(-1)\
                        .squeeze()

                    valid_outputs_mask = labels != -100


                    y_pred = predictions[valid_outputs_mask].to(device)
                    y_true = labels[valid_outputs_mask].to(device)

                    y_pred_val = torch.cat([y_pred, y_pred_val])
                    y_true_val = torch.cat([y_true, y_true_val])



            train_f1 = multiclass_f1_score(
                y_pred_train,
                y_true_train,
                num_classes=n_classes
            )

            val_f1 = multiclass_f1_score(
                y_pred_val, 
                y_true_val,
                num_classes=n_classes
            )

            self.save_best_model(
                y_pred_val, 
                y_true_val, 
                n_classes
            )

            self.history['train-f1'].append(train_f1.item())
            self.history['validation-f1'].append(val_f1.item())


            logging.info(
                f'''
                End of epoch {epoch}
                ---------------------
                
                Tranining f1-score : {train_f1}
                Validation f1-score : {val_f1}
                ''', 
            )


        
class BertTrainer(BaseTrainer):
    def __init__(self, model, optimizer=AdamW) -> None:
        super().__init__(model, optimizer)
        self.model_name = "bert"
    
    def compile(self, 
                train_dataloader: DataLoader, 
                validation_dataloader: DataLoader, 
                n_classes: int, 
                device: str = 'cpu', 
                num_epochs: int = 10, 
                lr: float = 0.00005
            ) -> None:
        optimizer = self.optimizer(self.model.parameters(), lr=lr)
        self.model.to(device)

        logging.info('''
            Starting model training
            -----------------------       
        ''')
        self.model.train()
        self.history = {
            "train_accuracy": [],
            "train-f1": [],
            "validation-accuracy": [],
            "validation-f1": []
        }

        for epoch in tqdm(range(num_epochs)):
            logging.info(f"Epoch: {epoch}")
            
            y_true_train = torch.tensor([],device=device)
            y_pred_train = torch.tensor([],device=device)
            y_pred_val = torch.tensor([],device=device)
            y_true_val = torch.tensor([],device=device)
            for X in (train_dataloader):
                input_ids = X["input_ids"]\
                    .to(device)\
                    .squeeze()
                
                bbox = X["bbox"]\
                    .to(device)\
                    .squeeze()

                attention_mask = X["attention_mask"]\
                    .to(device)\
                    .squeeze()
                token_type_ids = X["token_type_ids"]\
                    .to(device)\
                    .squeeze()
                
                labels = X["labels"]\
                    .to(device)\
                    .squeeze()
                
                if input_ids.shape[0] == 512:
                    continue


                outputs = self.model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    token_type_ids=token_type_ids,
                    labels=labels
                )
                
                loss = outputs.loss
                predictions = outputs.logits.argmax(-1)

                valid_outputs_mask = labels != -100

                y_pred = predictions[valid_outputs_mask].to(device)
                y_true = labels[valid_outputs_mask].to(device)

                y_pred_train = torch.cat([y_pred, y_pred_train])
                y_true_train = torch.cat([y_true, y_true_train])

                # backward pass to get the gradients 
                loss.backward()

                # update
                optimizer.step()
                optimizer.zero_grad()

            with torch.no_grad():
                for X_validation in validation_dataloader:
                    input_ids = X_validation["input_ids"]\
                        .to(device)\
                        .squeeze()
                
                    bbox = X_validation["bbox"]\
                        .to(device)\
                        .squeeze()

                    attention_mask = X_validation["attention_mask"]\
                        .to(device)\
                        .squeeze()
                    token_type_ids = X_validation["token_type_ids"]\
                        .to(device)\
                        .squeeze()
                    
                    labels = X_validation["labels"]\
                        .to(device)\
                        .squeeze()

                    outputs = self.model(
                        input_ids=input_ids.reshape(1,-1), 
                        attention_mask=attention_mask.reshape(1,-1), 
                        token_type_ids=token_type_ids.reshape(1,-1),
                        labels=labels.reshape(1,-1)
                    )

                    loss = outputs.loss
                    predictions = outputs\
                        .logits\
                        .argmax(-1)\
                        .squeeze()

                    valid_outputs_mask = labels != -100


                    y_pred = predictions[valid_outputs_mask].to(device)
                    y_true = labels[valid_outputs_mask].to(device)

                    y_pred_val = torch.cat([y_pred, y_pred_val])
                    y_true_val = torch.cat([y_true, y_true_val])



            train_f1 = multiclass_f1_score(
                y_pred_train,
                y_true_train,
                num_classes=n_classes
            )

            val_f1 = multiclass_f1_score(
                y_pred_val, 
                y_true_val,
                num_classes=n_classes
            )

            self.save_best_model(
                y_pred_val, 
                y_true_val, 
                n_classes
            )

            self.history['train-f1'].append(train_f1.item())
            self.history['validation-f1'].append(val_f1.item())


            logging.info(
                f'''
                End of epoch {epoch}
                ---------------------
                
                Tranining f1-score : {train_f1}
                Validation f1-score : {val_f1}
                ''', 
            )

