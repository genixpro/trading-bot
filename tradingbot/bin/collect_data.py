from tradingbot.data_gathering import TraderDataGatherer
from tradingbot.match_stream_aggregator import MatchStreamAggregator
from tradingbot.order_book_stream_aggregator import OrderBookStreamAggregator
import pymongo



def main():
    dataGatherer = TraderDataGatherer(saveToDisc=True)

    mongoClient = pymongo.MongoClient('mongodb://localhost:27017/')
    matchAggregator = MatchStreamAggregator(mongoClient, saveToDisc=True)
    orderBookAggregator = OrderBookStreamAggregator(mongoClient, saveToDisc=True)

    dataGatherer.add_match_hook(matchAggregator.processMatch)
    dataGatherer.add_order_book_hook(orderBookAggregator.processOrderBookMessage)
    dataGatherer.start()



if __name__ == "__main__":
    main()
