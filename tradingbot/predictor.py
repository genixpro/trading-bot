import torch
from datetime import timedelta, datetime
from tradingbot.constants import match_price_buckets, getMatchPriceBucket, volume_buckets, getVolumeBucket, prediction_intervals, order_book_price_buckets, prediction_sequence_input_length, aggregated_data_database_name
import math
from tradingbot.prediction_network import PredictionNetwork
import pymongo
import random
import numpy
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pprint import pprint
import sklearn.isotonic
import sklearn.calibration
import matplotlib.pyplot as plt

class NoFutureForAggregationFoundError(ValueError):
    pass

class Predictor:
    def __init__(self, mongo_client):
        self.mongo_client = mongo_client
        self.aggregated_matches_collection = self.mongo_client[aggregated_data_database_name].aggregated_matches
        self.aggregated_order_book_collection = self.mongo_client[aggregated_data_database_name].aggregated_order_book

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using {} device'.format(self.device))

        self.model = PredictionNetwork().to(self.device)

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=0.01,
            momentum=0.99,
            dampening=0.1,
            weight_decay=0.001,
            nesterov=False
        )

        self.lrScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10, factor=0.5, verbose=True)

        self.maxIterations = 50000
        self.sequenceLength = prediction_sequence_input_length
        self.batchSize = 32
        self.testingDataProportion = 0.7

        self.threadExecutor = ThreadPoolExecutor(max_workers=64)


        self.trainingInputs = torch.tensor([])
        self.trainingOutputs = torch.tensor([])

        self.testingInputs = torch.tensor([])
        self.testingOutputs = torch.tensor([])

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

    def computeMatchPriceDistribution(self, histogram):
        totalVolume = numpy.sum(list(histogram.values()))

        distribution = [
            histogram[priceBucket] / totalVolume
            for priceBucket in match_price_buckets
        ]

        return distribution

    def computeRelativeHistogram(self, aggregation, basePrice):
        histogram = {
            priceBucket: 0 for priceBucket in match_price_buckets
        }
        for price, volume in aggregation['matches']:
            relativePrice = price / basePrice

            priceBucket = getMatchPriceBucket(relativePrice)

            histogram[priceBucket] += volume

        return histogram

    def computeMergedRelativeHistogram(self, aggregations, basePrice):
        histogram = {
            priceBucket: 0 for priceBucket in match_price_buckets
        }
        for aggregation in aggregations:
            minuteHistogram = self.computeRelativeHistogram(aggregation, basePrice)
            for priceBucket, volume in minuteHistogram.items():
                histogram[priceBucket] += volume
        return histogram


    def prepareAbsoluteVolumeVectorForMatchPriceHistogram(self, histogram):
        vector = []

        for priceBucket in match_price_buckets:
            volume = histogram[priceBucket]

            # vector.append(0)

            if volume == 0:
                vector.append(0)
            else:
                vector.append(math.log(volume) * 0.1)

        return vector

    def prepareAbsoluteVolumeVectorForOrderBookHistogram(self, histogram):
        vector = []

        for priceBucket in order_book_price_buckets[1:-1]:
            value = histogram[str(priceBucket)]

            # vector.append(0)

            if value == 0:
                vector.append(0)
            else:
                vector.append(math.log(value) * 0.1)

        return vector

    def prepareCategoricalVolumeVectorForMatchPriceHistogram(self, histogram):
        vector = []

        for priceRange in match_price_buckets:
            volume = histogram[priceRange]
            volumeBucket = getVolumeBucket(volume)

            volumeOutputVector = [0] * len(volume_buckets)

            volumeOutputVector[volume_buckets.index(volumeBucket)] = 1

            vector.append(volumeOutputVector)

        return numpy.array(vector)

    def prepareMatchAggregationInput(self, aggregation, basePrice=None):
        vector = [
            math.log(aggregation['averagePrice']) * 0.1
        ]

        if basePrice is None:
            basePrice = aggregation['averagePrice']

        histogram = self.computeRelativeHistogram(aggregation, basePrice)

        # vector.extend(self.prepareAbsoluteVolumeVectorForMatchPriceHistogram(histogram))
        vector.extend(self.computeMatchPriceDistribution(histogram))
        # vector.extend(self.prepareCategoricalVolumeVectorForMatchPriceHistogram(histogram).flatten())

        return vector

    def prepareMatchAggregationExpectedOutput(self, aggregation, basePrice=None):
        time = aggregation["time"]

        if basePrice is None:
            basePrice = aggregation['averagePrice']

        outputVector = []

        for intervalStart, intervalEnd in prediction_intervals:
            nextAggregationTimes = [
                time + timedelta(minutes=relativeMinute)
                for relativeMinute in range(intervalStart, intervalEnd + 1)
            ]

            nextAggregations = list(self.aggregated_matches_collection.find({"_id": {"$in": nextAggregationTimes}}))

            if len(nextAggregations) != len(nextAggregationTimes):
                raise NoFutureForAggregationFoundError(f"We couldn't find all of the aggregations for interval range {intervalStart},{intervalEnd} in the future for aggregation {aggregation['time']} in order to prepare a prediction")

            histogram = self.computeMergedRelativeHistogram(nextAggregations, basePrice)

            # Temporary - have it predict the same output as it has input. This is just to test to ensure the
            # network is processing data correctly.
            # histogram = self.computeRelativeHistogram(aggregation, aggregation['averagePrice'])

            # intervalOutputVector = self.prepareCategoricalVolumeVectorForMatchPriceHistogram(histogram)

            intervalOutputVector = self.computeMatchPriceDistribution(histogram)

            outputVector.append(intervalOutputVector)

        return outputVector

    def prepareOrderBookAggregation(self, aggregation):
        vector = self.prepareAbsoluteVolumeVectorForOrderBookHistogram(aggregation['histogram'])

        # plt.bar(range(len(aggregation['histogram']) - 2), vector, tick_label=[str(v[0]) for v in order_book_price_buckets[1:-1]] )
        # plt.xticks(rotation=90)
        # plt.show()

        return vector


    def prepareAllAggregations(self):
        inputs = []
        outputs = []
        self.aggregations = []

        for aggregation in self.aggregated_matches_collection.find({}, sort=[("_id", pymongo.ASCENDING)]):
            try:
                orderBookAggregation = self.aggregated_order_book_collection.find_one({"_id": aggregation['_id']})

                if orderBookAggregation is not None:
                    inputVector = self.prepareMatchAggregationInput(aggregation)
                    outputVectors = self.prepareMatchAggregationExpectedOutput(aggregation)

                    inputVector.extend(self.prepareOrderBookAggregation(orderBookAggregation))

                    inputs.append(inputVector)
                    outputs.append(outputVectors)

                    self.aggregations.append((aggregation, orderBookAggregation))

            except NoFutureForAggregationFoundError:
                pass

        print(f"Finished preparing {len(inputs)} samples.")

        testingCutoffIndex = int(len(inputs) * self.testingDataProportion)

        self.trainingAggregations = self.aggregations[:testingCutoffIndex]
        self.testingAggregations = self.aggregations[testingCutoffIndex:]

        self.trainingInputs = torch.tensor(numpy.array([inputs[:testingCutoffIndex]]), device=self.device, dtype=torch.float32)
        self.trainingOutputs = torch.tensor(numpy.array([outputs[:testingCutoffIndex]]), device=self.device, dtype=torch.float32)

        self.testingInputs = torch.tensor(numpy.array([inputs[testingCutoffIndex:]]), device=self.device, dtype=torch.float32)
        self.testingOutputs = torch.tensor(numpy.array([outputs[testingCutoffIndex:]]), device=self.device, dtype=torch.float32)

    def prepareSequentialPricesForVectorSequence(self, vectors):
        absolutePrices = torch.exp(vectors[0, :, 0] * 10)
        relativePrices = (absolutePrices[1:] - absolutePrices[:-1]) * 0.1
        relativePrices = torch.cat([torch.zeros(1, device=self.device), relativePrices])
        relativePricesWithExtraDims = torch.unsqueeze(torch.unsqueeze(relativePrices, dim=1), dim=0)
        return relativePricesWithExtraDims

    def prepareNormalizedBatch(self, inputAggregations):
        lastPrice = inputAggregations[-1][0]['averagePrice']
        inputSequence = []
        for aggregation in inputAggregations:
            matchAggregation, orderBookAggregation = aggregation
            inputVector = self.prepareMatchAggregationInput(matchAggregation, lastPrice)
            inputVector.extend(self.prepareOrderBookAggregation(orderBookAggregation))

            inputSequence.append(inputVector)

        outputVector = self.prepareMatchAggregationExpectedOutput(inputAggregations[-1][0], lastPrice)
        inputSequence = torch.tensor(numpy.array([inputSequence]), device=self.device, dtype=torch.float32)
        outputVector = torch.tensor(numpy.array([outputVector]), device=self.device, dtype=torch.float32)

        # print(inputSequence.shape, outputVector.shape)

        return inputSequence, outputVector

    def prepareSingleTrainingBatchItem(self):
        start = random.randint(0, self.trainingInputs.shape[1] - self.sequenceLength - 1)
        inputSequence = self.trainingInputs[:, start:start + self.sequenceLength]
        outputVector = self.trainingOutputs[:, start + self.sequenceLength - 1]

        # inputAggregations = self.trainingAggregations[start:start + self.sequenceLength]
        # inputSequence, outputVector = self.prepareNormalizedBatch(inputAggregations)

        inputSequence = torch.cat([inputSequence, self.prepareSequentialPricesForVectorSequence(inputSequence)], dim=2)
        return inputSequence, outputVector

    def prepareTrainingBatch(self):
        input = []
        output = []

        for n in range(self.batchSize):
            inputSequence, outputVector = self.prepareSingleTrainingBatchItem()
            input.append(inputSequence)
            output.append(outputVector)

        return torch.cat(input, dim=0), torch.cat(output, dim=0)

    def prepareSingleTestingBatchItem(self):
        start = random.randint(0, self.testingInputs.shape[1] - self.sequenceLength - 1)

        inputSequence = self.testingInputs[:, start:start + self.sequenceLength]
        outputVector = self.testingOutputs[:, start + self.sequenceLength - 1]

        # inputAggregations = self.trainingAggregations[start:start + self.sequenceLength]
        # inputSequence, outputVector = self.prepareNormalizedBatch(inputAggregations)

        inputSequence = torch.cat([inputSequence, self.prepareSequentialPricesForVectorSequence(inputSequence)], dim=2)
        return inputSequence, outputVector

    def prepareTestingBatch(self):
        input = []
        output = []

        for n in range(self.batchSize):
            inputSequence, outputVector = self.prepareSingleTestingBatchItem()
            input.append(inputSequence)
            output.append(outputVector)

        return torch.cat(input, dim=0), torch.cat(output, dim=0)


    def runSingleTrainingIterations(self, iteration):
        self.model.train()

        batchInput, batchExpectedOutput = self.prepareTrainingBatch()

        loss = torch.zeros(1)

        def closure():
            nonlocal loss

            self.optimizer.zero_grad(set_to_none=True)

            modelOutput = self.model.forward(batchInput)

            # loss = torch.nn.functional.binary_cross_entropy(modelOutput, batchExpectedOutput)

            # loss = torch.nn.functional.kl_div(torch.log(modelOutput), batchExpectedOutput)
            # print(batchExpectedOutput[0])
            loss = torch.nn.functional.l1_loss(modelOutput, batchExpectedOutput)

            loss.backward()

            return loss

        self.optimizer.step(closure)

        return loss.cpu().item()


    def runSingleTestingIterations(self, iteration):
        batchInput, batchExpectedOutput = self.prepareTestingBatch()

        self.model.eval()

        modelOutput = self.model.forward(batchInput)

        # loss = torch.nn.functional.binary_cross_entropy(modelOutput, batchExpectedOutput)
        # loss = torch.nn.functional.kl_div(torch.log(modelOutput), batchExpectedOutput)
        loss = torch.nn.functional.l1_loss(modelOutput, batchExpectedOutput)

        if iteration % 1000 == 0:
            print("modelOutput", modelOutput[0])
            print("batchExpectedOutput", batchExpectedOutput[0])

        return loss.cpu().item()

    def train(self):
        self.prepareAllAggregations()

        bestTestingLoss = None

        testingLosses = []
        lastAverageTestingLoss = None

        epochsWithIncreasingTestingLoss = 0

        for iteration in range(self.maxIterations):
            start = datetime.now()

            trainingLoss = self.runSingleTrainingIterations(iteration)

            testingLoss = self.runSingleTestingIterations(iteration)
            testingLosses.append(testingLoss)

            if iteration % 100 == 0:
                testingLoss = numpy.mean(testingLosses[-100:])
                self.lrScheduler.step(testingLoss)

            if iteration % 1000 == 0:
                testingLoss = numpy.mean(testingLosses)
                if lastAverageTestingLoss is not None:
                    if testingLoss >= lastAverageTestingLoss:
                        epochsWithIncreasingTestingLoss += 1

                    if bestTestingLoss is None or testingLoss < bestTestingLoss:
                        bestTestingLoss = testingLoss

                print(f"Testing Iteration {iteration}. Loss {testingLoss:.4f}")
                testingLosses = []
                lastAverageTestingLoss = testingLoss

            end = datetime.now()

            if iteration % 1000 == 0:
                print(f"Iteration {iteration}. Loss {trainingLoss:.4f} Time per iteration {(end - start).total_seconds():.3f}")

            if epochsWithIncreasingTestingLoss > 2:
                break

        print(f"Best Testing Loss {bestTestingLoss:.4f}")

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

        inputTensor = torch.cat([inputTensor, self.prepareSequentialPricesForVectorSequence(inputTensor)], dim=2)

        self.model.eval()

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
                    "estimatedProportion": predictions[0][intervalIndex][priceRangeIndex].item()
                    # "volumes": []
                }
                # for volumeIndex, volumeBucket in enumerate(volume_buckets):
                #     priceData["volumes"].append({
                #         "volumeStart": volumeBucket[0],
                #         "volumeEnd": volumeBucket[1],
                #         "probability": predictions[0][intervalIndex][priceRangeIndex][volumeIndex].item()
                #     })

                predictionData["prices"].append(priceData)

            predictionObjects.append(predictionData)

        return predictionObjects

    def measureCalibration(self):
        plt.figure(figsize=(10, 10))
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax2 = plt.subplot2grid((3, 1), (2, 0))

        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

        input = []
        output = []

        for start in range(self.testingInputs.shape[1] - self.sequenceLength - 1):
            input.append(self.testingInputs[:, start:start + self.sequenceLength])
            output.append(self.testingOutputs[:, start:start + self.sequenceLength])

        inputTensor = torch.cat(input, dim=0)
        outputTensor = torch.cat(output, dim=0)

        predictions = self.model(inputTensor).cpu()

        for intervalIndex, interval in enumerate(prediction_intervals):
            outputsFlat = outputTensor[:, :, intervalIndex].reshape([-1]).cpu().detach().numpy()
            predictionsFlat = predictions[:, :, intervalIndex].reshape([-1]).cpu().detach().numpy()

            # calibrationModel = sklearn.isotonic.IsotonicRegression()
            # predictionsFlat = calibrationModel.fit_transform(predictionsFlat, outputsFlat)

            fraction_of_positives, mean_predicted_value = \
                sklearn.calibration.calibration_curve(outputsFlat, predictionsFlat, n_bins=30)

            name = f"Interval {interval[0]} - {interval[1]}"

            ax1.plot(mean_predicted_value, fraction_of_positives, "s-",label=name)

            ax2.hist(predictionsFlat, range=(0, 1), bins=30, label=name, histtype="step", lw=2)

        ax1.set_ylabel("Fraction of positives")
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend(loc="lower right")
        ax1.set_title('Calibration plots  (reliability curve)')

        ax2.set_xlabel("Mean predicted value")
        ax2.set_ylabel("Count")
        ax2.legend(loc="upper center", ncol=2)

        plt.tight_layout()
        plt.show()
