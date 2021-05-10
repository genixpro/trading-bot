from tradingbot.match_stream_aggregator import MatchStreamAggregator
from tradingbot.order_book_stream_aggregator import OrderBookStreamAggregator
from tradingbot.predictor import Predictor
import pymongo
from tradingbot.data_gathering import TraderDataGatherer
from datetime import datetime, timedelta
from pprint import pprint
import time
import tradingbot.constants

class TradingBot:
    def __init__(self):
        self.mongoClient = pymongo.MongoClient('mongodb://localhost:27017/')

        self.matchCollection = self.mongoClient[tradingbot.constants.raw_data_database_name].matches
        self.matchCollection.create_index("time")

        self.orderBookCollection = self.mongoClient[tradingbot.constants.raw_data_database_name].order_book
        self.orderBookCollection.create_index("time")

        self.aggregatedMatchesCollection = self.mongoClient[tradingbot.constants.aggregated_data_database_name].aggregated_matches
        self.aggregatedOrderBookCollection = self.mongoClient[tradingbot.constants.aggregated_data_database_name].aggregated_order_book

        self.myOrdersCollection = self.mongoClient[tradingbot.constants.my_orders_database_name].my_orders
        self.myOrdersCollection.create_index("end")

        self.matchAggregator = MatchStreamAggregator(self.mongoClient, saveToDisc=True)
        self.orderBookAggregator = OrderBookStreamAggregator(self.mongoClient, saveToDisc=True)

        self.predictor = None
        self.dataGatherer = None

    def clearHistoricMatchCollections(self):
        self.aggregatedMatchesCollection.delete_many({})
        self.aggregatedOrderBookCollection.delete_many({})


    def processHistoricMatches(self):
        matches = self.matchCollection.find({}, sort=[("time", pymongo.ASCENDING)])

        for match in matches:
            if match['type'] == 'match':
                self.matchAggregator.processMatch(match)

        self.matchAggregator.syncAggregations()


    def processHistoricOrderBookStream(self):
        messages = self.orderBookCollection.find({}, sort=[("time", pymongo.ASCENDING)])

        for message in messages:
            self.orderBookAggregator.processOrderBookMessage(message)


    def train(self):
        self.predictor = Predictor(self.mongoClient)

        print("Starting Training")
        self.predictor.train()

        # self.predictor.measureCalibration()

        self.predictor.saveModel()


    def runPrediction(self):
        orderBookAggregations = list(self.aggregatedOrderBookCollection.find({}, sort=(("_id", pymongo.DESCENDING),), limit=tradingbot.constants.prediction_sequence_input_length + 1) )

        orderBookAggregations = orderBookAggregations[1:]

        orderBookAggregationTimes = [aggregation['time'] for aggregation in orderBookAggregations]

        timesQuery = {"_id": {"$in": orderBookAggregationTimes}}

        matchAggregations = list(self.aggregatedMatchesCollection.find(timesQuery, sort=(("_id", pymongo.DESCENDING),), limit=tradingbot.constants.prediction_sequence_input_length ))

        # matchAggregationTimes = [aggregation['time'] for aggregation in matchAggregations]

        # print(len(matchAggregations))
        # print(len(orderBookAggregations))

        predictions = self.predictor.predict(matchAggregations, orderBookAggregations)

        orders = []

        for intervalPrediction in predictions[0:1]:
            bestExpectedProfit = 0
            bestSellPrice = None
            bestBuyPrice = None
            bestSellProbability = None
            bestBuyProbability = None
            bestClearProbability = None

            print([p['estimatedProportion'] for p in intervalPrediction['prices']])

            # for sellPriceBucketIndex in range(len(intervalPrediction['prices'])):
            #     if sellPriceBucketIndex == 0 or sellPriceBucketIndex == len(intervalPrediction['prices']) - 1:
            #         continue # skip the first and last range
            #
            #     # Determine the probability that an order with this specific sell price would clear in this time
            #     # we calculate this by multiplying together the likelihood that all prices above this price will
            #     # fail to clear
            #     sellClearFailureProbability = 1
            #     for sellPriceBucket in intervalPrediction['prices'][sellPriceBucketIndex:]:
            #         sellClearFailureProbability *= sellPriceBucket['volumes'][0]['probability']
            #
            #     for buyPriceBucketIndex in range(sellPriceBucketIndex + 1):
            #         if buyPriceBucketIndex == 0 or buyPriceBucketIndex == len(intervalPrediction['prices']) - 1:
            #             continue  # skip the first and last range
            #
            #         buyClearFailureProbability = 1
            #         for buyPriceBucket in intervalPrediction['prices'][:(buyPriceBucketIndex + 1)]:
            #             buyClearFailureProbability *= buyPriceBucket['volumes'][0]['probability']
            #
            #         sellSuccessProbability = 1.0 - sellClearFailureProbability
            #         buySuccessProbability = 1.0 - buyClearFailureProbability
            #
            #         jointSuccessProbability = buySuccessProbability * sellSuccessProbability
            #
            #         sellPrice = intervalPrediction['prices'][sellPriceBucketIndex]['priceStart']
            #         buyPrice = intervalPrediction['prices'][buyPriceBucketIndex]['priceEnd']
            #
            #         spread = sellPrice - buyPrice
            #
            #         expectedProfit = spread * jointSuccessProbability
            #
            #         if bestExpectedProfit is None or expectedProfit > bestExpectedProfit:
            #             bestExpectedProfit = expectedProfit
            #             bestSellPrice = sellPrice
            #             bestBuyPrice = buyPrice
            #             bestClearProbability = jointSuccessProbability
            #             bestSellProbability = sellSuccessProbability
            #             bestBuyProbability = buySuccessProbability


            print(f"Expected Profit: {bestExpectedProfit:.2f}")
            if bestExpectedProfit > 0:
                spread = bestSellPrice - bestBuyPrice

                adjustment = 0.9

                orders.append({
                    "_id": intervalPrediction['start'],
                    "start": intervalPrediction['start'],
                    "end": intervalPrediction['end'],
                    "buyPrice": bestBuyPrice + spread * (adjustment/2),
                    "sellPrice": bestSellPrice - spread * (adjustment/2),
                    "clearProbability": bestClearProbability,
                    "buyProbability": bestBuyProbability,
                    "sellProbability": bestSellProbability
                })
            else:
                print(f"No expected profit")

        for order in orders:
            print(f"Making order for {order['start']} to {order['end']}: buy at {order['buyPrice']:.1f} sell at {order['sellPrice']:.1f}, with expected clear probability of {order['clearProbability']:.3f}. Buy Prob: {order['buyProbability']:.3f}. Sell Prob: {order['sellProbability']:.3f}")
            self.myOrdersCollection.replace_one({"_id": order['_id']}, order, upsert=True)

    def measureActualProfit(self):
        now = datetime.now() + timedelta(hours=4)

        pastOrders = self.myOrdersCollection.find({"end": {"$lte": now}}, sort=(("_id", pymongo.DESCENDING),), limit=1)

        for pastOrder in pastOrders:
            aggregatedMatches = self.aggregatedMatchesCollection.find({
                "_id": {"$gte": pastOrder['start'], "$lt": pastOrder['end']}
            })

            didClearBuy = False
            didClearSell = False

            highestPrice = None
            lowestPrice = None

            for match in aggregatedMatches:
                for price, volume in match['matches']:
                    if price <= pastOrder['buyPrice']:
                        didClearBuy = True
                    if price >= pastOrder['sellPrice']:
                        didClearSell = True

                    if lowestPrice is None:
                        lowestPrice = price
                    else:
                        lowestPrice = min(price, lowestPrice)

                    if highestPrice is None:
                        highestPrice = price
                    else:
                        highestPrice = max(price, highestPrice)

            print(f"Interval: {pastOrder['start']} - {pastOrder['end']}. Buy: {pastOrder['buyPrice']:.2f}. Lowest: {lowestPrice:.2f}. Sell: {pastOrder['sellPrice']:.2f}. Highest: {highestPrice:.2f}. Did clear buy? {didClearBuy}. Did clear sell? {didClearSell}")

    def run(self):
        self.predictor = Predictor(self.mongoClient)
        self.predictor.loadModel()

        while True:
            start = datetime.now()

            self.runPrediction()

            self.measureActualProfit()

            nextMinute = start + timedelta(minutes=1)
            sleepTime = max(0, (nextMinute - datetime.now()).total_seconds())
            time.sleep(sleepTime)
