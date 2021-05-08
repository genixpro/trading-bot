import torch
from datetime import timedelta, datetime
from constants import relative_price_buckets, getPriceBucket, volume_buckets, getVolumeBucket, prediction_intervals, order_book_price_buckets
import math
from prediction_network import PredictionNetwork
import pymongo
import random
import numpy


class NoFutureForAggregationFoundError(ValueError):
    pass

class Predictor:
    def __init__(self, mongo_client):
        self.mongo_client = mongo_client
        self.aggregated_matches_collection = self.mongo_client.trader.aggregated_matches
        self.aggregated_order_book_collection = self.mongo_client.trader.aggregated_order_book

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using {} device'.format(self.device))

        self.model = PredictionNetwork().to(self.device)

        self.optimizer = torch.optim.Adamax(
            self.model.parameters(),
            lr=0.0001
        )

        self.maxIterations = 50000
        self.sequenceLength = 15
        self.batchSize = 16

        self.inputs = []
        self.outputs = []


    def prepareMatchAggregation(self, aggregation):
        time = aggregation["time"]

        inputVector = [
            math.log(aggregation['basePrice'])
        ]

        for priceBucket in relative_price_buckets:
            volume = aggregation['volumes'][str(priceBucket)]
            if volume == 0:
                inputVector.append(-1)
            else:
                inputVector.append(math.log(volume)*0.1)

        outputVector = []

        for interval in prediction_intervals:
            nextAggregationTime = time + timedelta(minutes=interval)
            nextAggregation = self.aggregated_matches_collection.find_one({"_id": nextAggregationTime})

            if nextAggregation is None:
                raise NoFutureForAggregationFoundError(f"There is no data {interval} minutes in the future for aggregation {aggregation['time']} in order to prepare a prediction")

            volumesRelativeToOriginalAggregation = {
                priceRange: 0 for priceRange in relative_price_buckets
            }

            for priceBucket in relative_price_buckets:
                # TODO: Improve how this is done so that we don't lose fidelity as a result of this bucketing we are doing
                nextPriceMidpoint = nextAggregation['basePrice'] * (priceBucket[1] + priceBucket[0]) / 2.0

                nextPriceRelative = nextPriceMidpoint / aggregation['basePrice']

                nextPriceBucketRelativeToOriginal = getPriceBucket(nextPriceRelative)

                volume = nextAggregation['volumes'][str(priceBucket)]

                # if volume > 0:
                #     print(nextPriceBucketRelativeToOriginal)

                volumesRelativeToOriginalAggregation[nextPriceBucketRelativeToOriginal] += volume


            intervalOutputVector = []

            for priceRange in relative_price_buckets:
                volume = volumesRelativeToOriginalAggregation[priceRange]
                volumeBucket = getVolumeBucket(volume)

                volumeOutputVector = [0] * len(volume_buckets)

                volumeOutputVector[volume_buckets.index(volumeBucket)] = 1

                intervalOutputVector.append(volumeOutputVector)

            outputVector.append(intervalOutputVector)

        return inputVector, outputVector

    def prepareOrderBookAggregation(self, aggregation):
        vector = []

        for priceBucket in order_book_price_buckets[1:-1]:
            vector.append(math.log(aggregation['histogram'][str(priceBucket)]) * 0.1)

        return vector


    def prepareAllAggregations(self):
        inputs = []
        outputs = []

        for aggregation in self.aggregated_matches_collection.find({}, sort=[("_id", pymongo.ASCENDING)]):
            try:
                orderBookAggregation = self.aggregated_order_book_collection.find_one({"_id": aggregation['_id']})

                if orderBookAggregation is not None:
                    inputVector, outputVectors = self.prepareMatchAggregation(aggregation)

                    inputVector.extend(self.prepareOrderBookAggregation(orderBookAggregation))

                    inputs.append(inputVector)
                    outputs.append(outputVectors)

            except NoFutureForAggregationFoundError:
                pass

        print(f"Finished preparing {len(inputs)} samples.")
        self.inputs = inputs
        self.outputs = outputs


    def prepareBatch(self):
        input = []
        output = []

        for n in range(self.batchSize):
            start = random.randint(0, len(self.inputs) - self.sequenceLength - 1)

            input.append(self.inputs[start:start + self.sequenceLength])
            output.append(self.outputs[start:start + self.sequenceLength])

        return torch.tensor(numpy.array(input), device=self.device, dtype=torch.float32), \
               torch.tensor(numpy.array(output), device=self.device, dtype=torch.float32)


    def runSingleTrainingIterations(self, iteration):
        start = datetime.now()

        self.optimizer.zero_grad(set_to_none=True)

        batchInput, batchExpectedOutput = self.prepareBatch()

        modelOutput = self.model.forward(batchInput)

        loss = torch.nn.functional.binary_cross_entropy(modelOutput, batchExpectedOutput)

        loss.backward()

        self.optimizer.step()

        end = datetime.now()

        if iteration % 100 == 0:
            print(f"Iteration {iteration}. Loss {loss.cpu().item():.4f} Time {(end - start).total_seconds():.3f}")

    def train(self):
        self.prepareAllAggregations()

        for iteration in range(self.maxIterations):
            self.runSingleTrainingIterations(iteration)
