from match_stream_aggregator import MatchStreamAggregator
from order_book_stream_aggregator import OrderBookStreamAggregator
from predictor import Predictor
import pymongo



class TradingBot:
    def __init__(self):
        self.mongoClient = pymongo.MongoClient('mongodb://localhost:27017/')

        self.matchAggregator = MatchStreamAggregator(self.mongoClient)
        self.matchCollection = self.mongoClient.trader.matches
        self.matchCollection.create_index("time")

        self.orderBookAggregator = OrderBookStreamAggregator(self.mongoClient)
        self.orderBookCollection = self.mongoClient.trader.order_book
        self.orderBookCollection.create_index("time")
        
        self.predictor = None


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

        print("Processing the order book history")
        self.processHistoricOrderBookStream()
        print("Processing the match history")
        self.processHistoricMatches()
        print("Starting Training")
        self.predictor.train()



bot = TradingBot()

bot.train()
