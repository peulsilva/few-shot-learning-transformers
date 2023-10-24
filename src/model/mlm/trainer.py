import torch
import logging
import tqdm
from torch.utils.data import Dataset
from src.preprocessing.make_dataset import SplitWordsDataset
from typing import Dict
from torch.nn.functional import cross_entropy
from transformers import AdamW
from torcheval.metrics.functional import multiclass_f1_score, multiclass_confusion_matrix
from copy import deepcopy

class MLMTrainer:

    def __init__(self,
                 model,
                 tokenizer,
                 labels_idx_keymap : Dict, 
                 optimizer = AdamW,
                 alpha : float = 1e-4,
                 device : str = "cuda",) -> None:
        
        self.device = device
        self.tokenizer = tokenizer
        self.model = model
        self.alpha = alpha
        self.optimizer = optimizer
        self.labels_idx_keymap = labels_idx_keymap
        
    def compile(self,
                train_data : SplitWordsDataset,
                n_shots : int,
                n_epochs : int = 10):
        
        self.history_train = []
        self.history_val = []

        for epoch in range(n_epochs):
            self.y_true_train = torch.tensor([],device=self.device)
            self.y_pred_train = torch.tensor([],device=self.device)

            for i in tqdm(range(n_shots)):

                for processed_data in (train_data[i]):
                    phrase = processed_data['pattern']
                    label = processed_data["label"]

                    tokens = self.tokenizer(
                        phrase,
                        truncation= True,
                        padding= "max_length",
                        return_tensors= "pt",
                        max_length=256
                    )

                    correct_word = self.tokenizer.tokenize(label)[0]

                    label = self.tokenizer(
                        correct_word,
                    )

                    input_ids = tokens['input_ids'].to(self.device)
                    attention_mask = tokens['attention_mask'].to(self.device)

                    y = input_ids.clone()

                    y[input_ids== self.tokenizer.mask_token_id] = self.tokenizer.vocab[correct_word]
                    y[input_ids != self.tokenizer.mask_token_id] = -100

                    outputs = self.model(
                        input_ids =input_ids,
                        attention_mask = attention_mask,
                        labels = y
                    )

                    mask_token_index = torch.where(tokens["input_ids"] == self.tokenizer.mask_token_id)[1]
                    mask_token_logits = outputs.logits[0, mask_token_index, :]

                    question_logits = mask_token_logits[0, self.tokenizer.vocab["question"]].item()
                    answer_logits = mask_token_logits[0, self.tokenizer.vocab["answer"]].item()
                    header_logits = mask_token_logits[0, self.tokenizer.vocab["header"]].item()
                    none_logits = mask_token_logits[0, self.tokenizer.vocab["nothing"]].item()

                    logits_list = torch.tensor([
                        none_logits,
                        question_logits, 
                        answer_logits,
                        header_logits
                    ])

                    real_value = correct_word.lower()
                    real_value = self.labels_idx_keymap[real_value]

                    predicted_value = logits_list.argmax().item()

                    self.y_pred_train = torch.cat([
                        self.y_pred_train, 
                        torch.tensor([predicted_value]).to(self.device)
                    ])
                    
                    self.y_true_train = torch.cat([
                        self.y_true_train,
                        torch.tensor([real_value]).to(self.device)
                    ])

                    real_list = [0,0,0,0]
                    real_list[real_value] =1

                    real_list = torch.tensor(
                        real_list
                    )

                    ce_loss = cross_entropy(
                        logits_list.softmax(dim = 0).to(torch.float64),
                        real_list.to(torch.float64)
                    )

                    mlm_loss = outputs.loss
                    loss = (1-self.alpha)*ce_loss + mlm_loss*self.alpha

                    loss.backward()

                    self.optmizer.step()
                    self.optmizer.zero_grad()

            f1 = multiclass_f1_score(
                self.y_pred_train,
                self.y_true_train,
                num_classes=4
            )
            self.history.append(f1)

            logging.info(f'''
                        
            -------------------------
                End of epoch {epoch} 
                F1 score : {f1}     
            ''')

            logging.info(f'''
                {multiclass_confusion_matrix(
                    self.y_pred_train.to(torch.int64),
                    self.y_true_train.to(torch.int64),
                    num_classes=4
                )}
            ''')

            y_true_val, y_pred_val = self.evaluate(num_shots=5,
                                                   model = self.model)

            f1_val = multiclass_f1_score(
                y_pred_val,
                y_true_val,
                num_classes=4
            )

            self.history_val.append(f1_val)
            
            logging.info(f'''
                        
            -------------------------
                Validation results
                F1 score (validation): {f1_val}     
            ''')

            if f1_val > best_f1:
                self.best_model = deepcopy(self.model)
                best_f1 = f1_val
            


            logging.info(f'''
                {multiclass_confusion_matrix(
                    y_pred_val.to(torch.int64),
                    y_true_val.to(torch.int64),
                    num_classes=4
                )}
            ''')

    def evaluate(
        self,
        validation_dataset : SplitWordsDataset,
        model = None,
        num_shots : int = 30
    ):
        if model is None:
            model = self.best_model
        self.y_true_val = torch.tensor([],device=self.device)
        self.y_pred_val = torch.tensor([],device=self.device)


        with torch.no_grad():
            for i in tqdm(range(num_shots)):

                for processed_data in (validation_dataset[i]):

                    phrase = processed_data['pattern']
                    label = processed_data["label"]

                    tokens = self.tokenizer(
                        phrase,
                        truncation= True,
                        padding= "max_length",
                        return_tensors= "pt",
                        max_length=256
                    )

                    correct_word = self.tokenizer.tokenize(label)[0]

                    label = self.tokenizer(
                        correct_word,
                    )

                    input_ids = tokens['input_ids'].to(self.device)
                    attention_mask = tokens['attention_mask'].to(self.device)

                    y = input_ids.clone()

                    y[input_ids== self.tokenizer.mask_token_id] = self.tokenizer.vocab[correct_word]
                    y[input_ids!= self.tokenizer.mask_token_id] = -100

                    outputs = model(
                        input_ids =input_ids,
                        attention_mask = attention_mask,
                    )

                    mask_token_index = torch.where(tokens["input_ids"] == self.tokenizer.mask_token_id)[1]
                    mask_token_logits = outputs.logits[0, mask_token_index, :]

                    question_logits = mask_token_logits[0,self.tokenizer.vocab["question"]].item()
                    answer_logits = mask_token_logits[0, self.tokenizer.vocab["answer"]].item()
                    header_logits = mask_token_logits[0, self.tokenizer.vocab["header"]].item()
                    none_logits = mask_token_logits[0, self.tokenizer.vocab["none"]].item()

                    logits = {
                        "question": question_logits,
                        "answer": answer_logits,
                        "header": header_logits,
                        "none": none_logits
                    }

                    predicted_value = sorted(
                        logits.items(), 
                        reverse= True, 
                        key = lambda x: x[1]
                    )[0][0]

                    predicted_value = self.labels_idx_keymap[predicted_value]

                    # logging.info(predicted_value)

                    real_value = correct_word.lower()
                    real_value = self.labels_idx_keymap[real_value]

                    self.y_pred_val = torch.cat([
                        self.y_pred_val, 
                        torch.tensor([predicted_value]).to(self.device)
                    ])
                    
                    self.y_true_val = torch.cat([
                        self.y_true_val,
                        torch.tensor([real_value]).to(self.device)
                    ])

        return self.y_true_val, self.y_pred_val