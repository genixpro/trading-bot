from tradingbot.match_stream_aggregator import MatchStreamAggregator
from tradingbot.order_book_stream_aggregator import OrderBookStreamAggregator
from tradingbot.predictor import Predictor
import pymongo
from tradingbot.data_gathering import TraderDataGatherer
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

        self.myOrdersCollection = self.mongoClient.trader.my_orders
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

        predictions = self.predictor.predict(matchAggregations, orderBookAggregations)

        orders = []

        for intervalPrediction in predictions[0:1]:
            lowestClearPrice = None
            highestClearPrice = None
            for pricePrediction in intervalPrediction['prices']:
                # We find the price that has a greater then 95% probability of having at least some volume
                # print(pricePrediction['volumes'][0]['probability'])
                if pricePrediction['volumes'][0]['probability'] < 0.05:
                    if lowestClearPrice is None:
                        lowestClearPrice = pricePrediction['priceEnd']
                    else:
                        lowestClearPrice = min(lowestClearPrice, pricePrediction['priceEnd'])

                    if highestClearPrice is None:
                        highestClearPrice = pricePrediction['priceStart']
                    else:
                        highestClearPrice = max(highestClearPrice, pricePrediction['priceStart'])

            if lowestClearPrice is not None and highestClearPrice is not None:
                spreadProfit = highestClearPrice - lowestClearPrice
                print(f"Estimate spread: {spreadProfit:.2f}")
                if spreadProfit > 0:
                    orders.append({
                        "_id": intervalPrediction['start'],
                        "start": intervalPrediction['start'],
                        "end": intervalPrediction['end'],
                        "buyPrice": lowestClearPrice,
                        "sellPrice": highestClearPrice
                    })
            else:
                print(f"No clearing price.")

        for order in orders:
            print(f"Making order: buy at {order['buyPrice']:.1f} sell at {order['sellPrice']:.1f}")
            self.myOrdersCollection.insert_one(order)

    def measureActualProfit(self):
        now = datetime.now() + timedelta(hours=4)

        pastOrders = self.myOrdersCollection.find({"end": {"$lte": now}}, sort=(("_id", pymongo.DESCENDING),), limit=1)

        for pastOrder in pastOrders:
            aggregatedMatches = self.aggregatedMatchesCollection.find({
                "_id": {"$gte": pastOrder['start'], "$lt": pastOrder['end']}
            })

            didClearBuy = False
            didClearSell = False

            for match in aggregatedMatches:
                for price, volume in match['matches']:
                    if price < pastOrder['buyPrice']:
                        didClearBuy = True
                    if price > pastOrder['sellPrice']:
                        didClearSell = True

            print(f"Interval: {pastOrder['start']} - {pastOrder['end']}. Did clear buy? {didClearBuy}. Did clear sell? {didClearSell}")

    def run(self):
        self.predictor = Predictor(self.mongoClient)
        self.predictor.loadModel()

        while True:
            start = datetime.now()
            self.processHistoricOrderBookStream()
            self.processHistoricMatches()

            self.runPrediction()

            self.measureActualProfit()

            nextMinute = start + timedelta(minutes=1)
            sleepTime = max(0, (nextMinute - datetime.now()).total_seconds())
            time.sleep(sleepTime)

        # self.dataGatherer = TraderDataGatherer(saveToDisc=False)
        #
        # self.dataGatherer.add_match_hook(self.matchAggregator.processMatch)
        # self.dataGatherer.add_order_book_hook(self.orderBookAggregator.processOrderBookMessage)
        # self.dataGatherer.start()

        # def onOrderBookAggregationHook(aggregation):
        #     self.matchAggregator.syncAggregations()

        # self.orderBookAggregator.addAggregationHook(onOrderBookAggregationHook)


