import torch
from torch import nn
from tradingbot.constants import match_price_buckets, getMatchPriceBucket, volume_buckets, getVolumeBucket, prediction_intervals, order_book_price_buckets




class PredictionNetwork(nn.Module):
    def __init__(self):
        super(PredictionNetwork, self).__init__()

        self.inputSize = 1 + len(match_price_buckets) + len(order_book_price_buckets) - 2
        self.outputSize = len(match_price_buckets) * len(volume_buckets) * len(prediction_intervals)
        self.lstmHiddenSize = 128
        self.denseHiddenSize = 128

        self.layer1 = nn.LSTM(self.inputSize, self.lstmHiddenSize)
        self.layer2 = nn.LSTM(self.lstmHiddenSize, self.lstmHiddenSize)

        self.dense = nn.Sequential(
            nn.Linear(self.lstmHiddenSize, self.denseHiddenSize),
            nn.ELU(),
            nn.Linear(self.denseHiddenSize, self.denseHiddenSize),
            nn.ELU(),
            nn.Linear(self.denseHiddenSize, self.outputSize),
        )

        self.softmax = torch.nn.Softmax(dim=4)

    def forward(self, input):
        batchSize = input.shape[0]
        sequenceLength = input.shape[1]

        layer1_output, (hidden1, cell1) = self.layer1(input)
        layer2_output, (hidden2, cell2) = self.layer2(layer1_output)

        sequence_reshaped = torch.reshape(layer2_output, [batchSize * sequenceLength, self.lstmHiddenSize])

        dense_output = self.dense(sequence_reshaped)

        dense_output_reshaped = torch.reshape(dense_output, [batchSize, sequenceLength, len(prediction_intervals), len(match_price_buckets), len(volume_buckets)])

        output_softmax = self.softmax(dense_output_reshaped)

        return output_softmax
