from sentence_transformers import SentenceTransformer, losses
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torcheval.metrics.functional import binary_f1_score, multiclass_confusion_matrix, multiclass_f1_score
from copy import deepcopy
from typing import List
from IPython.display import clear_output

class SetFitTrainer():
    def __init__(
        self,
        embedding_model ,
        classifier_model : torch.nn.Module,
        dataset_name : str,
        model_name : str,
        num_classes : int,
        device : str ='cuda' ,
    ) -> None:
        self.embedding_model = embedding_model.to(device)
        self.device = device
        self.dataset_name = dataset_name
        self.classifier_model = classifier_model.to(device)
        self.num_classes = num_classes
        self.model_name= model_name

    def train_embedding(
            self,
            train_dataloader : DataLoader,
            val_dataloader : DataLoader,
            n_shots : int,
            n_epochs : int = 10,
            save : bool = True,
            **kwargs
        ):
        loss_fn = losses.CosineSimilarityLoss(self.embedding_model)
        cos_sim = torch.nn.CosineSimilarity(dim = 1)

        best_f1 = 0
        self.best_model = None

        for epoch in range(n_epochs):
            self.embedding_model.fit(
                train_objectives=[ (train_dataloader, loss_fn)],
                epochs = 1,
                **kwargs
            )

            y_true_val = torch.tensor([],device=self.device)
            y_pred_val = torch.tensor([],device=self.device)

            print(f"Running validation after {epoch} epochs")

            for [x1, x2, y] in tqdm(val_dataloader):
                with torch.no_grad():
                    v1 = self.embedding_model.encode(x1, convert_to_tensor= True)
                    v2 = self.embedding_model.encode(x2, convert_to_tensor= True)

                    cos = cos_sim(v1, v2)

                    y_pred = round(cos.item())
                    y_true = y

                    y_pred_val = torch.cat([
                        y_pred_val, 
                        torch.tensor([y_pred]).to(self.device)
                    ])

                    y_true_val = torch.cat([
                        y_true_val, 
                        torch.tensor([y_true]).to(self.device)
                    ])
                    
            f1 = binary_f1_score(
                y_pred_val,
                y_true_val,
            )
            if f1 > best_f1:
                best_f1 = f1
                self.best_model = deepcopy(self.embedding_model)

            conf_matrix= multiclass_confusion_matrix(
                y_pred_val.to(torch.int64),
                y_true_val.to(torch.int64),
                num_classes=2
            )

            clear_output()
            print(f'f1 score: {f1.item()}')
            print(conf_matrix)

        if save:
            self.best_model.save_to_hub(f"peulsilva/{self.model_name}-setfit-{n_shots}shots-{self.dataset_name}")

    def train_classifier(
        self,
        X_train : List[str],
        y_train : List[int],
        X_val : List[str],
        y_val : List[int],
        n_epochs : int =100,
        loss_fn : torch.nn.Module = torch.nn.CrossEntropyLoss(),
        embedding_model = None,
        clf: torch.nn.Module = None,
        lr : float = 1e-5
    ):
        if embedding_model is None:
            self.embedding_model = self.best_model\
                .to(self.device)
            
        else:
            self.embedding_model = embedding_model

        if clf is not None:
            self.clf = clf

        optimizer = torch.optim.Adam(
            self.classifier_model.parameters(),
            lr = lr
        )

        self.best_clf = None
        best_f1 = 0

        self.history = []

        for epoch in (range(n_epochs)):
            for i in tqdm(range(len(X_train))):
                text = X_train[i]
                label = torch.tensor(y_train[i])\
                    .to(self.device)

                with torch.no_grad():
                    embedding = self.\
                        embedding_model\
                        .encode(text, convert_to_tensor=True)\
                        .to(self.device)

                optimizer.zero_grad()
                output = self.classifier_model(embedding)
                loss = loss_fn(output, label)


                loss.backward()
                optimizer.step()

            y_true_val = torch.tensor([],device=self.device)
            y_pred_val = torch.tensor([],device=self.device)

            for i in range(len(X_val)):
                text = X_val[i]
                label = torch.tensor(y_val[i])\
                    .to(self.device)

                with torch.no_grad():
                    embedding = self\
                        .embedding_model\
                        .encode(text, convert_to_tensor=True)

                    y_pred = self.classifier_model(embedding)\
                        .argmax()
                    
                    y_pred_val = torch.cat([
                        y_pred_val, 
                        torch.tensor([y_pred]).to(self.device)
                    ])

                    y_true_val = torch.cat([
                        y_true_val, 
                        torch.tensor([y_val[i]]).to(self.device)
                    ])

            if self.num_classes == 2:
                f1 = binary_f1_score(
                    y_pred_val,
                    y_true_val
                )

            else:
                    
                f1 = multiclass_f1_score(
                    y_pred_val,
                    y_true_val,
                    num_classes=self.num_classes
                )
            
            self.history.append(f1.item())
            if f1 > best_f1:
                best_f1 = f1
                self.best_clf = deepcopy(self.classifier_model)

            conf_matrix= multiclass_confusion_matrix(
                y_pred_val.to(torch.int64),
                y_true_val.to(torch.int64),
                num_classes=self.num_classes
            )

            clear_output()
            print(f"---------Epoch: {epoch}-----------")
            print(f'f1 score: {f1.item()}')
            print(conf_matrix)

        return self.history, self.embedding_model, self.best_clf
