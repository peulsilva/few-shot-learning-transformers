from transformers import AutoTokenizer
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
from typing import List
import torch
from abc import ABC, abstractmethod




class ImageLayoutDataset(Dataset):
    def __init__(self, 
                 data,
                 tokenizer,
                 encode : bool = True) -> None:
        super().__init__()

        self.tokenizer = tokenizer

        if encode:
            self.X = []
            for example in tqdm(data):
                X= self.encode(example)
                self.X.append(X)

        else:
            self.X = data
    
    def tokenize_labels(
        self,
        ner_tags : List,
        tokens 
    )-> torch.Tensor:
        """Aligns and tokenize labels

        Args:
            ner_tags (List): labels
            tokens (_type_): tokens

        Returns:
            torch.Tensor: tokenized labels
        """        
        
        labels = []

        word_ids = tokens.word_ids()  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(ner_tags[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

        return torch.Tensor(labels).to(torch.int64)
    
    
    def tokenize_boxes(
        self,
        words : List,
        boxes : List,
    ):
        
        token_boxes = []
        max_seq_length = 512
        pad_token_box = [0,0,0,0]
        
        for word, box in zip(words, boxes):
            word_tokens = self.tokenizer.tokenize(word)
            token_boxes.extend([box] * len(word_tokens))

        # Truncation of token_boxes
        special_tokens_count = 2 
        if len(token_boxes) > max_seq_length - special_tokens_count:
            token_boxes = token_boxes[: (max_seq_length - special_tokens_count)]

        # add bounding boxes of cls + sep tokens
        token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

        # Padding of token_boxes up the bounding boxes to the sequence length.
        input_ids = self.tokenizer(' '.join(words), truncation=True)["input_ids"]
        padding_length = max_seq_length - len(input_ids)
        token_boxes += [pad_token_box] * padding_length

        return torch.tensor(token_boxes)

    def encode(
        self,
        example, 
    ):
        words = example['words']
        boxes = example['bboxes']
        # image = Image.open(example['image_path'])s
        word_labels = example['ner_tags']

        
        tokens = self.tokenizer(
            words, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt",
            is_split_into_words= True
        )

        labels = self.tokenize_labels(word_labels,tokens)
        bbox = self.tokenize_boxes(words, boxes)

        tokens = {
            **tokens,
            "labels": labels,
            "bbox": bbox
        }
    
        return tokens

        
    
    def __getitem__(self, index: int):
        return self.X[index]

    def __len__(self):
        return len(self.X)
    
