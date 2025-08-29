import torch
import torch.nn as nn

class BatchNorm1dLayer(nn.Module):

    def __init__(self, feature_num):

        super(BatchNorm1dLayer, self).__init__()
        self.normlayer = nn.BatchNorm1d(feature_num)

    def forward(self, input):

        return self.normlayer(input)

class BatchNorm2dLayer(nn.Module):

    def __init__(self, feature_num):

        super(BatchNorm2dLayer, self).__init__()
        self.normlayer = nn.BatchNorm2d(feature_num)

    def forward(self, input):

        return self.normlayer(input)

class LayerNormLayer(nn.Module):

    def __init__(self, layer_shape):

        super(LayerNormLayer, self).__init__()
        self.normlayer = nn.LayerNorm(layer_shape)

    def forward(self, input):

        return self.normlayer(input)
