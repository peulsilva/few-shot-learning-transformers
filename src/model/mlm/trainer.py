import torch
import logging
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from src.preprocessing.make_dataset import SplitWordsDataset
from typing import Dict
from torch.nn.functional import cross_entropy
from transformers import AdamW
from torcheval.metrics.functional import multiclass_f1_score, multiclass_confusion_matrix
from copy import deepcopy
from IPython.display import clear_output

class MLMTrainer:

    def __init__(self,
                 model,
                 tokenizer,
                 verbalizer : Dict, 
                 optimizer = AdamW,
                 alpha : float = 1e-4,
                 device : str = "cuda",) -> None:
        
        self.device = device
        self.tokenizer = tokenizer
        self.model = model
        self.alpha = alpha
        self.optimizer = optimizer(self.model.parameters(), lr=1e-5)
        self.verbalizer = verbalizer
        self.inverse_verbalizer = {v:k for k, v in self.verbalizer.items()}
        self.best_model = None
        self.n_classes = len(verbalizer)

    def get_y_true(
        self,
        input: Dict[str, torch.Tensor],
        inverse_verbalizer : Dict,
        device : str = "cuda",
    )-> torch.tensor:
        
        y_true = input['labels']
        y_true = y_true[y_true!= -100].item()
        y_true = inverse_verbalizer[y_true]
        
        return torch.tensor(y_true, device= device)
        
    def compile(self,
                train_dataloader : DataLoader,
                n_shots : int,
                n_epochs : int = 10,
                n_validation : int = 10
            ):
        
        self.history_train = []
        self.history_val = []
        best_f1 = 0

        if self.best_model is None:
            self.best_model = deepcopy(self.model)

        logging.info(f"Starting model training with {n_shots} shots")

        for epoch in range(n_epochs):
            self.y_true_train = torch.tensor([],device=self.device)
            self.y_pred_train = torch.tensor([],device=self.device)

            for input in train_dataloader:
                out = self.model(**input)
                    
                tokens['labels'] = torch.tensor(label)

                for (k,v) in tokens.items():
                    tokens[k] = v.to(self.device)

                y_true = self.get_y_true(
                    tokens,
                    self.inverse_verbalizer
                )

                


                predictions = torch.Tensor(
                    [mask_token_logits[0,x] for x in self.verbalizer.values()]
                )

                real_value = correct_word.lower()
                real_value = self.verbalizer[real_value]

                predicted_value = predictions.argmax().item()

                self.y_pred_train = torch.cat([
                    self.y_pred_train, 
                    torch.tensor([predicted_value]).to(self.device)
                ])
                
                self.y_true_train = torch.cat([
                    self.y_true_train,
                    torch.tensor([real_value]).to(self.device)
                ])

                real_list = [0 for i in range(self.n_classes)]
                real_list[real_value] =1

                real_list = torch.tensor(
                    real_list
                )

                ce_loss = cross_entropy(
                    predictions.softmax(dim = 0).to(torch.float64),
                    real_list.to(torch.float64)
                )

                mlm_loss = outputs.loss
                loss = (1-self.alpha)*ce_loss + mlm_loss*self.alpha

                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

        f1 = multiclass_f1_score(
            self.y_pred_train,
            self.y_true_train,
            num_classes=self.n_classes
        )
        self.history_train.append(f1)

        logging.info(f'''
                    
        -------------------------
            End of epoch {epoch} 
            F1 score : {f1}     
        ''')

        logging.info(f'''
            {multiclass_confusion_matrix(
                self.y_pred_train.to(torch.int64),
                self.y_true_train.to(torch.int64),
                num_classes=self.n_classes
            )}
        ''')

        y_true_val, y_pred_val = self.evaluate(
            train_data[100:],
            n_shots=n_validation,
            model = self.model
        )

        f1_val = multiclass_f1_score(
            y_pred_val,
            y_true_val,
            num_classes=self.n_classes
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

        conf_matrix = multiclass_confusion_matrix(
            y_pred_val.to(torch.int64),
            y_true_val.to(torch.int64),
            num_classes=self.n_classes
        )

        clear_output(True)
        print(f'f1-score : {f1_val.item()}')
        print(conf_matrix)
        
        

        logging.info(f'''
            {conf_matrix}
        ''')

    def evaluate(
        self,
        validation_dataset : SplitWordsDataset,
        model = None,
        n_shots : int = 50,
        return_generated_dataset : bool = False
    ):
        if model is None:
            model = self.best_model
        self.y_true_val = torch.tensor([],device=self.device)
        self.y_pred_val = torch.tensor([],device=self.device)
        generated_labels = []


        with torch.no_grad():
            for i in tqdm(range(n_shots)):
                generated_labels_i=[]
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

                    generated_labels_i.append(predicted_value.upper())
                    predicted_value = self.verbalizer[predicted_value]

                    # logging.info(predicted_value)

                    real_value = correct_word.lower()
                    real_value = self.verbalizer[real_value]

                    self.y_pred_val = torch.cat([
                        self.y_pred_val, 
                        torch.tensor([predicted_value]).to(self.device)
                    ])
                    
                    self.y_true_val = torch.cat([
                        self.y_true_val,
                        torch.tensor([real_value]).to(self.device)
                    ])
                generated_labels.append(generated_labels_i)

        if return_generated_dataset:
            return generated_labels, self.y_true_val, self.y_pred_val
        
        return self.y_true_val, self.y_pred_val