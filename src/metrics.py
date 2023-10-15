import pandas as pd
import numpy as np
from typing import Dict
import torch

def confusion_matrix(
    y_pred: torch.Tensor,
    y_true : torch.Tensor,
    n_classes : int,
    keymap : Dict
):
    confusion_matrix = pd.DataFrame(
        np.zeros((n_classes, n_classes))
    )

    confusion_matrix.index.name = "prediction"
    confusion_matrix.columns.name = "true value"

    for idx in range(len(y_pred)):
        # if y_true[idx] == y_pred[idx]:
        #     continue
        confusion_matrix.loc[y_pred[idx].item(), y_true[idx].item()] += 1


    confusion_matrix.columns = confusion_matrix\
        .columns\
        .map(keymap)

    confusion_matrix.index = confusion_matrix\
        .index\
        .map(keymap)
    
    return confusion_matrix