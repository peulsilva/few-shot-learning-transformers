from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from src.model.container.auxiliary_functions import F_mean, F_cov
from transformers import AutoModel
from IPython.display import clear_output
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from copy import deepcopy
from sklearn.metrics import f1_score, confusion_matrix


class ContainerTrainer:
    def __init__(
        self,
        model : AutoModel,
        f_mean : F_mean,
        f_cov : F_cov,
        optimizer : torch.optim = None,
        checkpoint_llm : str = None,
        checkpoint_f_mean : str = None,
        checkpoint_f_cov : str = None,
        device : str = 'cuda'
    ) -> None:
        
        self.f_mean = f_mean
        self.f_cov = f_cov
        self.model = model

        params = list(f_mean.parameters()) + list(f_cov.parameters()) + list(model.parameters())

        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                params,
                lr = 1e-4
            )

        else :
            self.optimizer = optimizer

        self.checkpoint_llm = checkpoint_llm
        self.checkpoint_f_mean = checkpoint_f_mean
        self.checkpoint_f_cov = checkpoint_f_cov
        self.device = device

    @staticmethod
    def KL_div(
        mu_1 : torch.Tensor,
        mu_2 : torch.Tensor,
        cov_1: torch.Tensor,
        cov_2: torch.Tensor
    ):
        l = cov_1.shape[0]
        inv_1 = cov_1.inverse()
        # inv_2 = cov_2.inverse()
        kl = 1/2 * (mu_1 - mu_2).T @ inv_1 @ (mu_1 - mu_2)
        kl+= 1/2 * torch.trace(inv_1 @ cov_2)
        kl+= -l/2
        kl+= torch.log(
            torch.det(cov_1)/torch.det(cov_2)
        )

        return kl
    
    @staticmethod
    def generate_minibatches(
        source_df : Dataset, 
        minibatch_size : int = 5
    ):
        total_samples = len(source_df)
        total_batches = total_samples // minibatch_size

        # Initialize a list to store minibatches
        minibatches = []

        # Loop over the total number of batches
        for _ in range(total_batches):
            # Generate random indices for the minibatch
            random_indices = np.random.choice(total_samples, minibatch_size, replace=False)
            # Append the minibatch to the list
            minibatches.append(random_indices)

        return minibatches

    def train_on_source_set(
        self,
        source_df : Dataset,
        save : bool = False,
        num_batches : int = 20,
        **kwargs
    ):
        minibatches = ContainerTrainer.generate_minibatches(source_df, **kwargs)

        for batch_idx in range(num_batches):
            loss = 0
            total_points = 0
            for document_idx in tqdm(minibatches[batch_idx]):
                means = {}
                covs = {}
                X_p = 0
                s_X = 0
                n_p = 0

                size = source_df[document_idx]['labels'].shape[1]
                input = source_df[document_idx]

                out = self.model(
                    input_ids=input['input_ids'].reshape(1,-1), 
                    bbox= input['bbox'].reshape([1, 512, 4]),
                    attention_mask=input['attention_mask'].reshape(1,-1), 
                    token_type_ids=input['token_type_ids'].reshape(1,-1),
                )

                for i in range(size):
                    if source_df[document_idx]['labels'][:,i] == -100:
                        continue
                    mu_i = self.f_mean(out.last_hidden_state[:,i,:])
                    cov_i = self.f_cov(out.last_hidden_state[:,i, :])

                    means[i] = mu_i
                    covs[i] = cov_i

                del out

                for i in range(size):
                    if source_df[document_idx]['labels'][:,i] == -100:
                        continue


                    total_points+= 1
                    for j in range(i+1, size):
                        if source_df[document_idx]['labels'][:,j] == -100:
                            continue

                        # mu_i = f_mean(out.last_hidden_state[:,i,:])
                        # cov_i = f_cov(out.last_hidden_state[:,i, :])
                        # mu_j = f_mean(out.last_hidden_state[:,j,:])
                        # cov_j = f_cov(out.last_hidden_state[:,j, :])
                        mu_i = means[i]
                        cov_i = covs[i]
                        mu_j = means[j]
                        cov_j = covs[j]
                        d_ij = 1/2 * (ContainerTrainer.KL_div(mu_i, mu_j, cov_i, cov_j) + ContainerTrainer.KL_div(mu_j, mu_i, cov_j, cov_i))

                        if source_df[document_idx]['labels'][:,i] == source_df[document_idx]['labels'][:,j]:
                            X_p += torch.exp(-d_ij)
                            n_p += 1

                        s_X += torch.exp(-d_ij)

                del means
                del covs

                loss -= torch.log(X_p / n_p / s_X)
                

            loss = loss/total_points
            clear_output()
            print(f"batch {batch_idx}")
            print(f"loss: {loss.item()}")

            loss.backward()

            self.optimizer.step()

            self.optimizer.zero_grad()

        if save and self.checkpoint_f_cov and self.checkpoint_f_mean and self.checkpoint_llm:
            self.model.push_to_hub(self.checkpoint_llm)
            torch.save(self.f_mean.state_dict(), self.checkpoint_f_mean)
            torch.save(self.f_cov.state_dict(), self.checkpoint_f_cov)

    def train_on_support_set(
        self,
        support_df : Dataset,
        val_df : Dataset,
        n_shots : int, 
        n_epochs : int = 10,
        lr : float = 1e-4,
        checkpoint_f_mean : str = None, 
        checkpoint_f_cov : str = None,
        checkpoint_llm : str = None
    ):
        loss_prev = torch.inf
        loss_ft = 1e100

        self.knn = KNeighborsClassifier(n_neighbors=5)

        best_f1 = -1
        self.best_model = None
        history = []


        self.f_mean.load_state_dict(torch.load(self.checkpoint_f_mean))
        self.f_cov.load_state_dict(torch.load(self.checkpoint_f_cov))

        self.model = AutoModel.from_pretrained(
            self.checkpoint_llm,
            cache_dir = "/Data/pedro.silva/"
        ).to(self.device)

        params = list(self.f_mean.parameters()) + list(self.f_cov.parameters()) + list(self.model.parameters())
        optimizer_support = torch.optim.Adam(
            params,
            lr = lr
        )

        for epoch in range(n_epochs):
            

            loss_prev = loss_ft
            loss_ft = 0
            total_points = 0
            for document_idx in tqdm(range(n_shots)):
                means = {}
                covs = {}
                X_p = 0
                s_X = 0
                n_p = 0

                size = support_df[document_idx]['labels'].shape[1]
                input = support_df[document_idx]

                out = self.model(
                    input_ids=input['input_ids'].reshape(1,-1), 
                    bbox= input['bbox'].reshape([1, 512, 4]),
                    attention_mask=input['attention_mask'].reshape(1,-1), 
                    token_type_ids=input['token_type_ids'].reshape(1,-1),
                )

                for i in range(size):
                    if support_df[document_idx]['labels'][:,i] == -100:
                        continue
                    mu_i = self.f_mean(out.last_hidden_state[:,i,:])
                    cov_i = self.f_cov(out.last_hidden_state[:,i, :])

                    means[i] = mu_i
                    covs[i] = cov_i

                del out

                for i in range(size):
                    if support_df[document_idx]['labels'][:,i] == -100:
                        continue

                    total_points+= 1
                    for j in range(i+1, size):
                        if support_df[document_idx]['labels'][:,j] == -100:
                            continue

                        # mu_i = f_mean(out.last_hidden_state[:,i,:])
                        # cov_i = f_cov(out.last_hidden_state[:,i, :])
                        # mu_j = f_mean(out.last_hidden_state[:,j,:])
                        # cov_j = f_cov(out.last_hidden_state[:,j, :])
                        mu_i = means[i]
                        cov_i = covs[i]
                        mu_j = means[j]
                        cov_j = covs[j]
                        d_ij = 1/2 * (ContainerTrainer.KL_div(mu_i, mu_j, cov_i, cov_j) + ContainerTrainer.KL_div(mu_j, mu_i, cov_j, cov_i))

                        if support_df[document_idx]['labels'][:,i] == support_df[document_idx]['labels'][:,j]:
                            X_p += torch.exp(-d_ij)
                            n_p += 1

                        s_X += torch.exp(-d_ij)

                del means
                del covs

                loss_ft -= torch.log(X_p / n_p / s_X)
                

            loss_ft = loss_ft/total_points
            clear_output()
            print(f"epoch {epoch}")
            print(f"loss: {loss_ft.item()}")

            loss_ft.backward()

            optimizer_support.step()

            optimizer_support.zero_grad()

            # validation
            print("running validation")
            
            with torch.no_grad():
                h = []
                y = []
                for document_idx in tqdm(range(n_shots)):
                    

                    X_p = 0
                    s_X = 0
                    n_p = 0

                    size = support_df[document_idx]['labels'].shape[1]
                    input = support_df[document_idx]

                    out = self.model(
                        input_ids=input['input_ids'].reshape(1,-1), 
                        bbox= input['bbox'].reshape([1, 512, 4]),
                        attention_mask=input['attention_mask'].reshape(1,-1), 
                        token_type_ids=input['token_type_ids'].reshape(1,-1),
                    )

                    for i in range(size):
                        if support_df[document_idx]['labels'][:,i] == -100:
                            continue
                        v = out\
                            .last_hidden_state[:,i,:]\
                            .cpu()\
                            .numpy()\
                            .reshape(-1)
                        
                        label = support_df[document_idx]['labels'][:,i].item()
                        
                        h.append(v)
                        y.append(label)

                self.knn.fit(h,y)
                h = []
                y = []
                for document_idx in tqdm(range(n_shots, n_shots+100)):
                    

                    
                    size = val_df[document_idx]['labels'].shape[1]
                    input = val_df[document_idx]

                    out = self.model(
                        input_ids=input['input_ids'].reshape(1,-1), 
                        bbox= input['bbox'].reshape([1, 512, 4]),
                        attention_mask=input['attention_mask'].reshape(1,-1), 
                        token_type_ids=input['token_type_ids'].reshape(1,-1),
                    )

                    for i in range(size):
                        if val_df[document_idx]['labels'][:,i] == -100:
                            continue
                        v = out\
                            .last_hidden_state[:,i,:]\
                            .cpu()\
                            .numpy()\
                            .reshape(-1)
                        
                        label = val_df[document_idx]['labels'][:,i].item()
                        
                        h.append(v)
                        y.append(label)

                pred = self.knn.predict(h)
                f1= f1_score(y, pred, average='micro')
                conf_matrix = confusion_matrix(y, pred)

                print(f"f1: {f1}")
                print(conf_matrix)

                history.append(f1)
                if f1 > best_f1:
                    self.best_model = deepcopy(self.model)
                    best_f1 = f1

        if checkpoint_f_cov and checkpoint_f_mean and checkpoint_llm:
            self.best_model.push_to_hub(checkpoint_llm)
            torch.save(self.f_mean.state_dict(), checkpoint_f_mean)
            torch.save(self.f_cov.state_dict(),checkpoint_f_cov)

        return history


    def test(
        self,
        support_df : Dataset,
        test_df : Dataset,
        n_shots : int 
    ):
        with torch.no_grad():
            h = []
            y = []
            for document_idx in tqdm(range(n_shots)):

                size = support_df[document_idx]['labels'].shape[1]
                input = support_df[document_idx]

                out = self.best_model(
                    input_ids=input['input_ids'].reshape(1,-1), 
                    bbox= input['bbox'].reshape([1, 512, 4]),
                    attention_mask=input['attention_mask'].reshape(1,-1), 
                    token_type_ids=input['token_type_ids'].reshape(1,-1),
                )

                for i in range(size):
                    if support_df[document_idx]['labels'][:,i] == -100:
                        continue
                    v = out\
                        .last_hidden_state[:,i,:]\
                        .cpu()\
                        .numpy()\
                        .reshape(-1)
                    
                    label = support_df[document_idx]['labels'][:,i].item()
                    
                    h.append(v)
                    y.append(label)

            self.knn.fit(h,y)

            h = []
            y = []
            for document_idx in tqdm((range(len(test_df)))):
            

                size = test_df[document_idx]['labels'].shape[1]
                input = test_df[document_idx]

                out = self.best_model(
                    input_ids=input['input_ids'].reshape(1,-1), 
                    bbox= input['bbox'].reshape([1, 512, 4]),
                    attention_mask=input['attention_mask'].reshape(1,-1), 
                    token_type_ids=input['token_type_ids'].reshape(1,-1),
                )

                for i in range(size):
                    if test_df[document_idx]['labels'][:,i] == -100:
                        continue
                    v = out\
                        .last_hidden_state[:,i,:]\
                        .cpu()\
                        .numpy()\
                        .reshape(-1)
                    
                    label = test_df[document_idx]['labels'][:,i].item()
                    
                    h.append(v)
                    y.append(label)

            pred = self.knn.predict(h)
            
            return y, pred
