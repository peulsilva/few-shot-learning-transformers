import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer
from typing import Dict
from tqdm import tqdm
import logging
from torcheval.metrics.functional import multiclass_f1_score, multiclass_confusion_matrix, binary_f1_score
from torch.nn.functional import cross_entropy
from copy import deepcopy
from IPython.display import clear_output


def get_y_true(
    input: Dict[str, torch.Tensor],
    inverse_verbalizer : Dict,
    device : str = "cuda",
)-> torch.tensor:
    
    y_true = input['labels']
    y_true = y_true[y_true!= -100].item()
    y_true = inverse_verbalizer[y_true]
    
    return torch.tensor(y_true, device= device)

def train(
    train_dataloader : DataLoader,
    val_dataloader : DataLoader,
    num_classes: int,
    model : AutoModelForMaskedLM,
    verbalizer : Dict,
    tokenizer : AutoTokenizer,
    alpha : float,
    evaluation_fn : callable = multiclass_f1_score,
    loss_fn : torch.nn.Module = None,
    device : str = 'cuda',
    lr : float = 1e-5,
    n_epochs : int =10,
):
    inverse_verbalizer = {v:k for k, v in verbalizer.items()}

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = lr
    )

    best_f1=  0
    best_model = None
    confusion_matrix= None

    history = []

    for epoch in range(n_epochs):

        for input in train_dataloader:
            out = model(**input)

            loss_mlm = out['loss']

            y_true = get_y_true(
                input,
                inverse_verbalizer
            )

            try:
                mask_token_index = torch.where(input["input_ids"] == tokenizer.mask_token_id)[1]

            except:
                mask_token_index = torch.where(input["input_ids"] == tokenizer.mask_token_id)[0]

            mask_token_logits = out.logits[0, mask_token_index, :]
            
            predictions = torch.Tensor(
                [mask_token_logits[0,x] for x in verbalizer.values()]
            )

            probabilities = predictions.softmax(dim = 0).to(device)
                
            loss_ce = cross_entropy(
                probabilities,
                y_true
            )

            if loss_fn is not None:
                loss_ce = loss_fn(
                    probabilities, 
                    y_true
                )
            loss_ce.requires_grad = True

            loss = alpha*loss_mlm + (1-alpha)* loss_ce
            

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():    
            y_true_val = torch.tensor([],device=device)
            y_pred_val = torch.tensor([],device=device)

            for input in tqdm(val_dataloader):
                out = model(**input)

                y_true = get_y_true(
                    input,
                    inverse_verbalizer
                )

                try:
                    mask_token_index = torch.where(input["input_ids"] == tokenizer.mask_token_id)[1]

                except:
                    mask_token_index = torch.where(input["input_ids"] == tokenizer.mask_token_id)[0]
                mask_token_logits = out.logits[0, mask_token_index, :]
                
                predictions = torch.Tensor(
                    [mask_token_logits[0,x] for x in verbalizer.values()]
                )

                y_pred = predictions.argmax().to(device)

                y_pred_val = torch.cat([
                    y_pred_val, 
                    torch.tensor([y_pred]).to(device)
                ])

                y_true_val = torch.cat([
                    y_true_val,
                    torch.tensor([y_true]).to(device)
                ])
            
            if evaluation_fn == multiclass_f1_score:
                f1 = multiclass_f1_score(
                    y_pred_val,
                    y_true_val,
                    num_classes= num_classes
                )

            else:
                f1 = evaluation_fn(
                    y_pred_val,
                    y_true_val,
                )


            conf_matrix = multiclass_confusion_matrix(
                y_pred_val.to(torch.int64),
                y_true_val.to(torch.int64),
                num_classes= num_classes
            )
            
            clear_output(True)
            print(f'f1-score : {f1.item()}')
            print(conf_matrix)
            


            if f1 > best_f1:
                best_f1 = f1
                confusion_matrix = conf_matrix
                best_model = deepcopy(model)

            history.append(f1.item())

            logging.info("-------------------------")
            logging.info(f"End of epoch {epoch}: f1 = {f1.item()}")
            logging.info(conf_matrix)
    
    return history, confusion_matrix, best_model