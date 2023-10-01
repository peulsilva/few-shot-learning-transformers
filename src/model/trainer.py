import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
import logging
from torcheval.metrics.functional import multiclass_f1_score

class Trainer():
    def __init__(self,
                 model,
                 optimizer = AdamW,
                 model_type : str = "bert"):
       
       assert model_type in ["bert", "layoutlm"]
       self.optimizer = optimizer
       self.model = model
       self.model_type = model_type


    def compile(
            self, 
            train_dataloader : DataLoader,
            validation_dataloader : DataLoader,
            n_classes : int,
            device :str = 'cpu', 
            num_epochs : int = 10,
            lr : float = 5e-5
        ):
        optimizer = AdamW(self.model.parameters(), lr=lr)
        self.model.to(device)
        early_stopping = False

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

                if self.model_type == "layoutlm":
                    outputs = self.model(
                        input_ids=input_ids, 
                        bbox=bbox, 
                        attention_mask=attention_mask, 
                        token_type_ids=token_type_ids,
                        labels=labels
                    )

                elif self.model_type == "bert":
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

            if len(self.history['validation-f1']) > 0 and self.history['validation-f1'][-1] > val_f1 and epoch >= 3:
                early_stopping = True

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

            if early_stopping:
                print(f"Early stopping on epoch {epoch}")
                return

