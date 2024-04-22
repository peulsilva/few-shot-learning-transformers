import torch
class F_mean(torch.nn.Module):
    def __init__(self, 
                 input_dim:int, 
                 output_dim: int, 
                 hidden_size : int = 128,
                 device : str = 'cuda',
                 *args, 
                 **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.layer1 = torch.nn.Linear(input_dim, hidden_size).to(device)
        self.layer2 = torch.nn.Linear(hidden_size, output_dim).to(device)
        self.elu = torch.nn.ELU().to(device)
        self.relu = torch.nn.ReLU().to(device)

    def forward(self, x: torch.Tensor):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return self.elu(x).squeeze()

class F_cov(torch.nn.Module):
    def __init__(self, 
                 input_dim:int, 
                 output_dim: int, 
                 hidden_size : int = 128,
                 device : str = 'cuda',
                 epsilon : float = 1e-14,
                 *args, 
                 **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.epsilon = epsilon

        self.layer1 = torch.nn.Linear(input_dim, hidden_size).to(device)
        self.layer2 = torch.nn.Linear(hidden_size, output_dim).to(device)
        self.elu = torch.nn.ELU().to(device)
        self.relu = torch.nn.ReLU().to(device)

    def forward(self, x: torch.Tensor):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.elu(x)
        x += 1 + self.epsilon

        # return torch.diag(x)
        return torch.diag(x.squeeze())