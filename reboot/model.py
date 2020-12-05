import pandas as pd
import torch
from collections import OrderedDict


class Model_builder(object):
    @classmethod
    def dnn_relu(cls):
        return DNN_relu

    @classmethod
    def dnn_tanh(cls):
        return DNN_tanh

    @classmethod
    def dnn_relu_tanh(cls):
        return DNN_relu_tanh

    @classmethod
    def rnn(cls):
        return RNN

    @classmethod
    def transformer(cls):
        return Transformer


def get_model(model_name):
    return getattr(Model_builder, model_name)()


class DNN_relu(torch.nn.Module):
    def __init__(self, feature_size, hidden_layer, **kwargs):
        super().__init__()
        if type(hidden_layer) == int:
            hidden_layer = [hidden_layer]
        dropout = kwargs.get('dropout', [])
        use_batchnorm = kwargs.get('batchnorm', [])
        if type(dropout) == int:
            dropout = [dropout]
        if type(use_batchnorm) == bool:
            use_batchnorm = [use_batchnorm]
        if dropout and len(dropout) != len(hidden_layer) - 1:
            raise ValueError("Dropout size and hidden layer size mismatched")
        if use_batchnorm and len(use_batchnorm) != len(hidden_layer):
            raise ValueError("use_batchnorm and hidden layer size mismatched")
        self.layers = []
        if use_batchnorm and use_batchnorm[0]:
            self.layers.append(torch.nn.BatchNorm1d(feature_size))
        self.layers.append(torch.nn.Linear(feature_size, hidden_layer[0]))
        for i in range(1, len(hidden_layer)):
            self.layers.append(torch.nn.ReLU())
            if use_batchnorm and use_batchnorm[i]:
                self.layers.append(torch.nn.BatchNorm1d(hidden_layer[i - 1]))
            if dropout and dropout[i - 1] > 0:
                self.layers.append(torch.nn.Dropout(dropout[i - 1]))
            self.layers.append(torch.nn.Linear(hidden_layer[i - 1], hidden_layer[i]))
        self.model = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.model(x)
        return x


class DNN_tanh(torch.nn.Module):
    def __init__(self, feature_size, hidden_layer, **kwargs):
        super().__init__()
        if type(hidden_layer) == int:
            hidden_layer = [hidden_layer]
        dropout = kwargs.get('dropout', [])
        use_batchnorm = kwargs.get('batchnorm', [])
        if type(dropout) == int:
            dropout = [dropout]
        if type(use_batchnorm) == bool:
            use_batchnorm = [use_batchnorm]
        if dropout and len(dropout) != len(hidden_layer) - 1:
            raise ValueError("Dropout size and hidden layer size mismatched")
        if use_batchnorm and len(use_batchnorm) != len(hidden_layer):
            raise ValueError("use_batchnorm and hidden layer size mismatched")
        self.layers = []
        if not use_batchnorm and use_batchnorm[0]:
            self.layers.append(torch.nn.BatchNorm1d(feature_size))
        self.layers.append(torch.nn.Linear(feature_size, hidden_layer[0]))
        for i in range(1, len(hidden_layer)):
            self.layers.append(torch.nn.Tanh())
            if use_batchnorm and use_batchnorm[i-1]:
                self.layers.append(torch.nn.BatchNorm1d(hidden_layer[i - 1]))
            if dropout and dropout[i - 1] > 0:
                self.layers.append(torch.nn.Dropout(dropout[i - 1]))
            self.layers.append(torch.nn.Linear(hidden_layer[i - 1], hidden_layer[i]))
        self.model = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.model(x)
        return x


class DNN_relu_tanh(torch.nn.Module):
    def __init__(self, feature_size, hidden_layer, **kwargs):
        super().__init__()
        if type(hidden_layer) == int:
            hidden_layer = [hidden_layer]
        dropout = kwargs.get('dropout', [])
        if type(kwargs['dropout']) == int:
            dropout = [dropout]
        if len(dropout) != len(hidden_layer) - 1:
            raise ValueError("Dropout size and hidden layer size mismatched")
        self.layers = [torch.nn.Linear(feature_size, hidden_layer[0])]
        for i in range(1, len(hidden_layer)):
            self.layers.append(torch.nn.ReLU())
            if dropout[i - 1] > 0:
                self.layers.append(torch.nn.Dropout(dropout[i - 1]))
            self.layers.append(torch.nn.Linear(hidden_layer[i - 1], hidden_layer[i]))
        self.layers.append(torch.nn.Tanh())
        self.model = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.model(x)
        return x


class RNN(torch.nn.Module):
    def __init__(self, feature_size, hidden_layer, config=None):
        super().__init__()
        if type(hidden_layer) == int:
            hidden_layer = [hidden_layer, 1]
        if config is None:
            config = ('tanh',)
        self.rnn = torch.nn.RNN(input_size=feature_size, hidden_size=hidden_layer[0], num_layers=hidden_layer[1],
                                nonlinearity=config[0])

    def forward(self, x):
        pass


class RNN_bidrectional(torch.nn.Module):
    def __init__(self, feature_size, hidden_layer, config=None):
        super().__init__()
        if type(hidden_layer) == int:
            hidden_layer = [hidden_layer, 1]
        if config is None:
            config = ('tanh',)
        self.rnn = torch.nn.RNN(input_size=feature_size, hidden_size=hidden_layer[0], num_layers=hidden_layer[1],
                                nonlinearity=config[0])

    def forward(self, x):
        pass


class Transformer(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, nonlinearity='tanh'):
        super().__init__()
        pass

    def forward(self, x):
        pass
