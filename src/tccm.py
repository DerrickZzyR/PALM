import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):

    def __init__(self, num_features: int, eps: float=1e-05, affine: bool=False, subtract_last: bool=False):
        super().__init__()
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str='norm'):
        if mode == 'denorm':
            return self._denormalize(x)
        if self.subtract_last:
            self.center = x[:, -1:, :]
        else:
            self.center = x.mean(dim=1, keepdim=True).detach()
        self.scale = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        normalized = (x - self.center) / self.scale
        if self.affine:
            normalized = normalized * self.affine_weight + self.affine_bias
        return normalized, self.center

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        return x * self.scale + self.center


class CausalMLP(nn.Module):
    def __init__(self, num_series: int, lag: int, hidden_dims: tuple[int, ...] = (256, 128, 64), activation: str = "relu"):
        super().__init__()
        layers = [nn.Conv1d(num_series, hidden_dims[0], lag)]
        layers.extend(nn.Conv1d(input_dim, output_dim, 1) for input_dim, output_dim in zip(hidden_dims, (*hidden_dims[1:], 1)))
        self.layers = nn.ModuleList(layers)
        activations = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh, "leakyrelu": nn.LeakyReLU}
        self.activation = activations[activation]()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        for index, layer in enumerate(self.layers):
            if index:
                x = self.activation(x)
            x = layer(x)
        return x.transpose(1, 2)


class TCCM(nn.Module):
    def __init__(self, num_series: int, lag: int, affine: bool = False, subtract_last: bool = False, hidden: tuple[int, ...] = (256, 128, 64), activation: str = "relu"):
        super().__init__()
        self.p = num_series
        self.lag = lag
        self.revin_layer = RevIN(num_features=num_series, affine=affine, subtract_last=subtract_last)
        self.networks = nn.ModuleList(CausalMLP(num_series, lag, hidden, activation) for _ in range(num_series))

    def _parallel_network_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        first_layers = [network.layers[0] for network in self.networks]
        x = F.conv1d(x, torch.cat([layer.weight for layer in first_layers]), torch.cat([layer.bias for layer in first_layers]))

        for layer_index in range(1, len(self.networks[0].layers)):
            x = self.networks[0].activation(x)
            layers = [network.layers[layer_index] for network in self.networks]
            x = F.conv1d(x, torch.cat([layer.weight for layer in layers]), torch.cat([layer.bias for layer in layers]), groups=self.p)
        return x.transpose(1, 2)

    def forward(self, x: torch.Tensor):
        normalized, statistics = self.revin_layer(x, "norm")
        model_input = normalized[:, :-1]
        prediction = self._parallel_network_forward(model_input)
        raw_prediction = self.revin_layer(prediction, "denorm")
        target = normalized[:, self.lag:]
        residual = prediction - target
        return prediction, raw_prediction, statistics, residual, target

    def GC(self, threshold: bool = True, ignore_lag: bool = True) -> torch.Tensor:
        if ignore_lag:
            weights = [torch.norm(network.layers[0].weight, dim=(0, 2)) for network in self.networks]
        else:
            weights = [torch.norm(network.layers[0].weight, dim=0) for network in self.networks]
        causality = torch.stack(weights)
        return (causality > 0).int() if threshold else causality
