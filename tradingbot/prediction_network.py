import torch
from torch import nn
from tradingbot.constants import match_price_buckets, getMatchPriceBucket, volume_buckets, getVolumeBucket, prediction_intervals, order_book_price_buckets


class RNNNoHidden(nn.GRU):
    def forward(self, input, hx=None):
        output = super(RNNNoHidden, self).forward(input, hx)
        return output[0]



class PredictionNetwork(nn.Module):
    def __init__(self):
        super(PredictionNetwork, self).__init__()

        # self.inputSize = 2 + len(match_price_buckets) * len(volume_buckets) + len(order_book_price_buckets) - 2
        self.inputSize = 2 + len(match_price_buckets) + len(order_book_price_buckets) - 2
        # self.outputSize = len(match_price_buckets) * len(volume_buckets) * len(prediction_intervals)
        self.outputSize = len(match_price_buckets) * len(prediction_intervals)
        self.rnnHiddenSize = 128
        self.denseHiddenSize = 128

        self.sequenceLayers = nn.Sequential(
            RNNNoHidden(self.inputSize, self.rnnHiddenSize, bidirectional=True, batch_first=True),
            # nn.LayerNorm(self.rnnHiddenSize * 2),
            nn.Dropout(0.6),
            RNNNoHidden(self.rnnHiddenSize * 2, self.rnnHiddenSize, batch_first=True),
            # nn.LayerNorm(self.rnnHiddenSize)
        )

        self.dense = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(self.rnnHiddenSize, self.denseHiddenSize),
            nn.ELU(),
            # nn.BatchNorm1d(self.denseHiddenSize),
            nn.Dropout(0.6),
            nn.Linear(self.denseHiddenSize, self.denseHiddenSize),
            nn.ELU(),
            # nn.BatchNorm1d(self.denseHiddenSize),
            nn.Linear(self.denseHiddenSize, self.outputSize),
        )

        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, input):
        batchSize = input.shape[0]
        sequenceLength = input.shape[1]

        sequence_output = self.sequenceLayers(input)

        last_sequence_entry = sequence_output[:, -1, :]

        dense_output = self.dense(last_sequence_entry)
        dense_output_reshaped = torch.reshape(dense_output, [batchSize, len(prediction_intervals), len(match_price_buckets)])

        output_softmax = self.softmax(dense_output_reshaped)

        return output_softmax
