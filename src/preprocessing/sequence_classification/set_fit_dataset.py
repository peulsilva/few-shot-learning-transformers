import torch 
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import InputExample
from typing import List, Dict
import numpy as np

class SetFitDataset(Dataset):
    def __init__(
        self, 
        text : List[str],
        labels : List[int],
        R : int = -1,
        input_example_format : bool = True
    ) -> None:
        
        self.data = SetFitDataset.expand_data(text, labels, R, input_example_format)

    @staticmethod
    def expand_data(
        X : List[str],
        y : List[int], 
        R : int = -1,
        input_example_format : bool = True
    ):
        expanded_data = []
        for i in range(len(X)):
            if R < 0:
                upper_bound = len(X)
            
            else:
                upper_bound = R
            for j in range(i+1, min(i+1+upper_bound, len(X))):
                label_i = y[i]
                if R > 0:
                    idx_j = np.random.randint(i+1, len(X))

                else: 
                    idx_j = j
                label_j = y[idx_j]

                y_ = int(label_i == label_j)

                if input_example_format:
                
                    expanded_data.append(
                        InputExample(texts = [X[i], X[idx_j]], label =  float(y_) )
                    )
                
                else:
                    expanded_data.append(
                        [X[i], X[idx_j], y_] 
                    )

        return expanded_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]