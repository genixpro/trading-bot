import torch
from datetime import timedelta, datetime
from tradingbot.constants import match_price_buckets, getMatchPriceBucket, volume_buckets, getVolumeBucket, prediction_intervals, order_book_price_buckets
import math
from tradingbot.prediction_network import PredictionNetwork
import pymongo
import random
import numpy
from concurrent.futures import ProcessPoolExecutor
from pprint import pprint


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

        self.maxIterations = 25000
        self.sequenceLength = 15
        self.batchSize = 16

        self.inputs = torch.tensor([])
        self.outputs = torch.tensor([])

    def saveModel(self, fileName="model.bin"):
        f = open(fileName, 'wb')
        torch.save(self.model.state_dict(), f)
        f.close()


    def loadModel(self, fileName="model.bin"):
        f = open(fileName, 'rb')

        stateDict = torch.load(f, map_location=self.device)

        f.close()

        # Load the state dictionary into the model itself.
        self.model.load_state_dict(stateDict)

    def computeRelativeHistogram(self, aggregation, basePrice):
        histogram = {
            priceBucket: 0 for priceBucket in match_price_buckets
        }
        for price, volume in aggregation['matches']:
            relativePrice = price / basePrice

            priceBucket = getMatchPriceBucket(relativePrice)

            histogram[priceBucket] += volume

        return histogram

    def prepareMatchAggregationInput(self, aggregation):
        inputVector = [
            math.log(aggregation['averagePrice'])
        ]

        histogram = self.computeRelativeHistogram(aggregation, aggregation['averagePrice'])

        for priceBucket in match_price_buckets:
            volume = histogram[priceBucket]
            if volume == 0:
                inputVector.append(-1)
            else:
                inputVector.append(math.log(volume)*0.1)

        return inputVector

    def prepareMatchAggregationExpectedOutput(self, aggregation):
        time = aggregation["time"]

        outputVector = []

        for intervalStart, intervalEnd in prediction_intervals:
            histogram = {
                priceBucket: 0 for priceBucket in match_price_buckets
            }
            for relativeMinute in range(intervalStart, intervalEnd + 1):
                nextAggregationTime = time + timedelta(minutes=relativeMinute)
                nextAggregation = self.aggregated_matches_collection.find_one({"_id": nextAggregationTime})

                if nextAggregation is None:
                    raise NoFutureForAggregationFoundError(f"There is no data {relativeMinute} minutes in the future for aggregation {aggregation['time']} in order to prepare a prediction")

                minuteHistogram = self.computeRelativeHistogram(nextAggregation, aggregation['averagePrice'])
                for priceBucket, volume in minuteHistogram.items():
                    histogram[priceBucket] += volume

            intervalOutputVector = []

            for priceRange in match_price_buckets:
                volume = histogram[priceRange]
                volumeBucket = getVolumeBucket(volume)

                volumeOutputVector = [0] * len(volume_buckets)

                volumeOutputVector[volume_buckets.index(volumeBucket)] = 1

                intervalOutputVector.append(volumeOutputVector)

            outputVector.append(intervalOutputVector)

        return outputVector

    def prepareOrderBookAggregation(self, aggregation):
        vector = []

        for priceBucket in order_book_price_buckets[1:-1]:
            value = aggregation['histogram'][str(priceBucket)]

            if value == 0:
                vector.append(-1)
            else:
                vector.append(math.log(value) * 0.1)

        return vector


    def prepareAllAggregations(self):
        inputs = []
        outputs = []

        for aggregation in self.aggregated_matches_collection.find({}, sort=[("_id", pymongo.ASCENDING)]):
            try:
                orderBookAggregation = self.aggregated_order_book_collection.find_one({"_id": aggregation['_id']})

                if orderBookAggregation is not None:
                    inputVector = self.prepareMatchAggregationInput(aggregation)
                    outputVectors = self.prepareMatchAggregationExpectedOutput(aggregation)

                    inputVector.extend(self.prepareOrderBookAggregation(orderBookAggregation))

                    inputs.append(inputVector)
                    outputs.append(outputVectors)

            except NoFutureForAggregationFoundError:
                pass

        print(f"Finished preparing {len(inputs)} samples.")
        self.inputs = torch.tensor(numpy.array([inputs]), device=self.device, dtype=torch.float32)
        self.outputs = torch.tensor(numpy.array([outputs]), device=self.device, dtype=torch.float32)


    def prepareBatch(self):
        input = []
        output = []

        for n in range(self.batchSize):
            start = random.randint(0, self.inputs.shape[1] - self.sequenceLength - 1)

            input.append(self.inputs[:, start:start + self.sequenceLength])
            output.append(self.outputs[:, start:start + self.sequenceLength])

        return torch.cat(input, dim=0), torch.cat(output, dim=0)


    def runSingleTrainingIterations(self, iteration):
        start = datetime.now()

        self.optimizer.zero_grad(set_to_none=True)

        batchInput, batchExpectedOutput = self.prepareBatch()

        modelOutput = self.model.forward(batchInput)

        loss = torch.nn.functional.binary_cross_entropy(modelOutput, batchExpectedOutput)

        loss.backward()

        self.optimizer.step()

        end = datetime.now()

        if iteration % 1000 == 0:
            print(f"Iteration {iteration}. Loss {loss.cpu().item():.4f} Time per batch {(end - start).total_seconds():.3f}")

    def train(self):
        self.prepareAllAggregations()

        for iteration in range(self.maxIterations):
            self.runSingleTrainingIterations(iteration)


    def predict(self, recentMatchAggregations, recentOrderBookAggregations):
        if len(recentMatchAggregations) != len(recentOrderBookAggregations):
            raise ValueError(f"The length of the recent match aggregations does not match the length of the recent order book aggregations")

        recentMatchAggregations = sorted(recentMatchAggregations, key=lambda match: match['_id'])
        recentOrderBookAggregations = sorted(recentOrderBookAggregations, key=lambda orderBookAgg: orderBookAgg['_id'])

        # print([match['time'] for match in recentMatchAggregations])
        # print(recentOrderBookAggregations)

        input = []
        for matchAggregation, orderBookAggregation in zip(recentMatchAggregations, recentOrderBookAggregations):
            inputVector = self.prepareMatchAggregationInput(matchAggregation)
            inputVector.extend(self.prepareOrderBookAggregation(orderBookAggregation))

            input.append(inputVector)

        inputTensor = torch.tensor(numpy.array([input]), device=self.device, dtype=torch.float32)

        predictions = self.model(inputTensor).cpu()

        predictionObjects = []

        for intervalIndex, (intervalStart, intervalEnd) in enumerate(prediction_intervals):
            predictionData = {
                "intervalStart": intervalStart,
                "intervalEnd": intervalEnd,
                "start": recentMatchAggregations[-1]['time'] + timedelta(minutes=intervalStart),
                "end": recentMatchAggregations[-1]['time'] + timedelta(minutes=intervalEnd),
                "prices": []
            }

            for priceRangeIndex, priceBucket in enumerate(match_price_buckets):
                priceData = {
                    "relativePriceStart": priceBucket[0],
                    "relativePriceEnd": priceBucket[1],
                    "priceStart": priceBucket[0] * recentMatchAggregations[-1]['averagePrice'],
                    "priceEnd": priceBucket[1] * recentMatchAggregations[-1]['averagePrice'],
                    "volumes": []
                }
                for volumeIndex, volumeBucket in enumerate(volume_buckets):
                    priceData["volumes"].append({
                        "volumeStart": volumeBucket[0],
                        "volumeEnd": volumeBucket[1],
                        "probability": predictions[0][-1][intervalIndex][priceRangeIndex][volumeIndex].item()
                    })

                predictionData["prices"].append(priceData)

            predictionObjects.append(predictionData)

        return predictionObjects

