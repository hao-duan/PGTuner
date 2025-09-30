import torch
import torch.nn as nn

class Direct_Predict_MLP(nn.Module):
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            layer_list.append(nn.Linear(input_size, curr_size))

            if i < final_relu_layer:
                layer_list.append(nn.BatchNorm1d(curr_size))
                layer_list.append(nn.ReLU(inplace=False))

        layer_list.append(nn.ReLU(inplace=False))
        self.net = nn.Sequential(*layer_list)

        self._init_weights()

        self.weights = torch.tensor([[1, 5, 20]]).cuda()

        self.loss = nn.MSELoss(reduction='none')

    def _init_weights(self):
        for m in self.net:
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, mean=0.0, std=1e-2)
                nn.init.uniform_(m.bias, -0.1, 0.1)

    def forward(self, x):
        o = self.net(x)
        return o

    def get_feature_vectors(self, x, feature_layer=3):
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i == feature_layer * 3 - 1:
                break
        return x

    def calculate_loss(self, inp, label):
        weighted_loss = self.loss(inp, label) * self.weights
        loss = torch.mean(weighted_loss)

        return loss

class Direct_Predict_MLP_nsg(nn.Module):
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]  # [17, 128, 256, 64, 2]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            layer_list.append(nn.Linear(input_size, curr_size))

            if i < final_relu_layer:
                layer_list.append(nn.BatchNorm1d(curr_size))
                layer_list.append(nn.ReLU(inplace=False))

        layer_list.append(nn.ReLU(inplace=False))
        self.net = nn.Sequential(*layer_list)

        self._init_weights()

        self.weights = torch.tensor([[1, 20]]).cuda()

        self.loss = nn.MSELoss(reduction='none')

    def _init_weights(self):
        for m in self.net:
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, mean=0.0, std=1e-2)
                nn.init.uniform_(m.bias, -0.1, 0.1)

    def forward(self, x):
        o = self.net(x)
        return o

    def get_feature_vectors(self, x, feature_layer=3):
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i == feature_layer * 3 - 1:
                break

        return x

    def calculate_loss(self, inp, label):
        weighted_loss = self.loss(inp, label) * self.weights
        loss = torch.mean(weighted_loss)

        return loss

