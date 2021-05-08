from match_stream_aggregator import MatchStreamAggregator
from order_book_stream_aggregator import OrderBookStreamAggregator
from predictor import Predictor
import pymongo
from data_gathering import TraderDataGatherer
from datetime import datetime, timedelta
from pprint import pprint
import time


class TradingBot:
    def __init__(self):
        self.mongoClient = pymongo.MongoClient('mongodb://localhost:27017/')

        self.matchCollection = self.mongoClient.trader.matches
        self.matchCollection.create_index("time")

        self.orderBookCollection = self.mongoClient.trader.order_book
        self.orderBookCollection.create_index("time")

        self.aggregatedMatchesCollection = self.mongoClient.trader.aggregated_matches
        self.aggregatedOrderBookCollection = self.mongoClient.trader.aggregated_order_book

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
        self.clearHistoricMatchCollections()

        self.predictor = Predictor(self.mongoClient)

        print("Processing the order book history")
        self.processHistoricOrderBookStream()
        print("Processing the match history")
        self.processHistoricMatches()
        print("Starting Training")
        self.predictor.train()

        self.predictor.saveModel()


    def runPrediction(self):
        orderBookAggregations = list(self.aggregatedOrderBookCollection.find({}, sort=(("_id", pymongo.DESCENDING),), limit=16) )

        orderBookAggregations = orderBookAggregations[1:]

        orderBookAggregationTimes = [aggregation['time'] for aggregation in orderBookAggregations]

        timesQuery = {"_id": {"$in": orderBookAggregationTimes}}

        matchAggregations = list(self.aggregatedMatchesCollection.find(timesQuery, sort=(("_id", pymongo.DESCENDING),), limit=15 ))

        prediction = self.predictor.predict(matchAggregations, orderBookAggregations)

        pprint(prediction)

    def run(self):
        self.predictor = Predictor(self.mongoClient)
        self.predictor.loadModel()

        while True:
            self.processHistoricOrderBookStream()
            self.processHistoricMatches()

            self.runPrediction()

            time.sleep(10)

        # self.dataGatherer = TraderDataGatherer(saveToDisc=False)
        #
        # self.dataGatherer.add_match_hook(self.matchAggregator.processMatch)
        # self.dataGatherer.add_order_book_hook(self.orderBookAggregator.processOrderBookMessage)
        # self.dataGatherer.start()

        # def onOrderBookAggregationHook(aggregation):
        #     self.matchAggregator.syncAggregations()

        # self.orderBookAggregator.addAggregationHook(onOrderBookAggregationHook)


bot = TradingBot()

# bot.train()
bot.run()
