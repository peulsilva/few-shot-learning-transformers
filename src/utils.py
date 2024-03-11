
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.utils.data import DataLoader, Dataset
from typing import Union


# class FocalLoss(nn.Module):
#     def __init__(self, gamma=0, alpha=None, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
#         if isinstance(alpha,(list, np.ndarray)): self.alpha = torch.Tensor(alpha)
#         self.size_average = size_average

#     def forward(self, input, target):
#         if input.dim()>2:
#             input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
#         target = target.view(-1,1)

#         logpt = F.log_softmax(input)
#         logpt = logpt.gather(1,target)
#         logpt = logpt.view(-1)
#         pt = Variable(logpt.data.exp())

#         if self.alpha is not None:
#             if self.alpha.type()!=input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#             at = self.alpha.gather(0,target.data.view(-1))
#             logpt = logpt * Variable(at)

#         loss = -1 * (1-pt)**self.gamma * logpt
#         if self.size_average: 
#             return loss.mean()
        
#         else: 
#             return loss.sum()

class FocalLoss(torch.nn.Module):
    def __init__(
        self, 
        alpha : Union[list, float, int]=None, 
        gamma=2,
        device : str = 'cuda'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.device = device

        if isinstance(alpha,(float,int)): 
            self.alpha = torch.Tensor([alpha,1-alpha]).to(device)

        if isinstance(alpha,(list, np.ndarray)): 
            self.alpha = torch.Tensor(alpha).to(device)


    def forward(self, inputs, targets):
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none').to(self.device)
        pt = torch.exp(-ce_loss).to(self.device)
        loss = (self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss).mean()
        return loss


def stratified_train_test_split(
    dataset : Dataset,
    classes : np.ndarray,
    train_size : float
):
    """Performs train test split keeping class distributions

    Args:
        dataset (Dataset): _description_
        classes (np.ndarray): _description_
        train_size (float): _description_

    Returns:
        _type_: _description_
    """    

    indexes_dict = {}
    for label in classes[0]:
        indexes_dict[label] = []

    for i in range(len(dataset['labels'])):
        label = dataset['labels'][i]
        text = dataset['Sentence'][i]
        indexes_dict[label].append(text)


    train_data = {
        'labels': [],
        'text': []
    }

    validation_data = {
        "labels" : [],
        "text": []
    }

    # generating train data
    for label in classes[0]:
        n = len(indexes_dict[label])
        size = int(train_size * n)

        train_data['text'] += indexes_dict[label][:size]
        train_data['labels'] += [label]*size
        
        validation_data['text'] +=indexes_dict[label][size:]
        validation_data['labels'] += [label]* (n-size)

    return train_data, validation_data

