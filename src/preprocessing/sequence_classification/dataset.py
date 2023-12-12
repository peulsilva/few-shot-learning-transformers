import torch 
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from transformers import AutoTokenizer

class TextDataset(Dataset):
    def __init__(self, 
                 input_ids : torch.Tensor, 
                 attention_mask : torch.Tensor,
                 labels : torch.tensor, 
                ) -> None:
        super().__init__()

        self.input_ids = input_ids
        self.labels = labels 
        self.attention_mask = attention_mask

    def __len__(self):
        return self.input_ids.shape[0]
    
    def __getitem__(self, index):
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "labels": self.labels[index]
        }
    
    
def get_n_shots_per_class(
    text : List[str],
    labels : List[str],
    n_shots : int,
    num_classes : int
):
    data_per_label = {}

    for i in range(num_classes):
        data_per_label[i] = []

    for i in range(len(text)):
        data_per_label[labels[i]].append(text[i])

    text = []
    labels = []
    num_classes = len(data_per_label.keys())
    
    for label in range(num_classes):
        for j in range(n_shots):
            text.append(data_per_label[label][j])
            labels.append(label)
    
    return text, labels

def get_dataloader(
    text : List[str],
    labels : List[str],
    tokenizer : AutoTokenizer,
    n_shots : int,
    num_classes : int,
    equalize_class : bool = True,
    **kwargs
):
    """Returns the dataloader with the correct number of shots per label

    Args:
        data_per_label (Dict): _description_
        tokenizer (AutoTokenizer): _description_
        n_shots (int): _description_

    Returns:
        _type_: _description_
    """    
    if equalize_class:
        text, labels = get_n_shots_per_class(
            text,
            labels, 
            n_shots, 
            num_classes
        )

    else:
        text = text[0:n_shots*num_classes]
        labels = labels[0:n_shots*num_classes]

    tokens = tokenizer(
        text,
        truncation= True,
        padding= "max_length",
        return_tensors= "pt",
        max_length=256,
    )

    dataset = TextDataset(
        input_ids=tokens['input_ids'],
        attention_mask=tokens['attention_mask'],
        labels = torch.tensor(labels)
    )

    dataloader = DataLoader(
        dataset,
        **kwargs
    )

    return dataloader