from datetime import datetime
from pprint import pprint
from tradingbot.constants import match_price_buckets, getMatchPriceBucket, order_book_price_buckets, getOrderBookPriceBucket
import tradingbot.constants
import numpy

class OrderBookStreamAggregator:
    aggregation_keep_in_memory_seconds = 180

    def __init__(self, mongo_client, saveToDisc=True):
        self.aggregations = {}

        self.mongo_client = mongo_client

        self.aggregated_order_book_collection = self.mongo_client[tradingbot.constants.aggregated_data_database_name].aggregated_order_book

        self.most_recent_message_time_processed = None
        self.messages_processed = 0
        
        self.current_bids = {}
        self.current_asks = {}

        self.saveToDisc = saveToDisc

        self.hooks = []

    def processOrderBookMessage(self, message):
        if message['type'] == 'snapshot':
            self.current_bids = {
                price: amount for price, amount in message['bids']
            }
            self.current_asks = {
                price: amount for price, amount in message['asks']
            }
        elif message['type'] == 'l2update':
            time = self.getMinuteForMessage(message)

            if self.most_recent_message_time_processed is not None:
                if time > self.most_recent_message_time_processed:
                    aggregation = self.createAggregationFromCurrentOrderBook()
                    if self.saveToDisc:
                        self.aggregated_order_book_collection.replace_one({"_id": aggregation['_id']}, aggregation, upsert=True)
                    for hook in self.hooks:
                        hook(aggregation)

            for change in message['changes']:
                if change[0] == 'buy':
                    if float(change[2]) == 0:
                        try:
                            del self.current_bids[change[1]]
                        except KeyError:
                            print(f"Error in processing the order book! I received an update for which there was no key in our existing order book")
                    else:
                        self.current_bids[change[1]] = change[2]

                elif change[0] == 'sell':
                    if float(change[2]) == 0:
                        try:
                            del self.current_asks[change[1]]
                        except KeyError:
                            print(f"Error in processing the order book! I received an update for which there was no key in our existing order book")
                    else:
                        self.current_asks[change[1]] = change[2]

            self.most_recent_message_time_processed = time
        self.messages_processed += 1

        if self.messages_processed % 100000 == 0:
            print(f"Processed {self.messages_processed:,} messages.")

    def getMinuteForMessage(self, message):
        timeStringWithoutSeconds = message['time'].strftime("%Y-%m-%dT%H:%M:00")
        return datetime.fromisoformat(timeStringWithoutSeconds)


    def createAggregationFromCurrentOrderBook(self):
        largestBid = numpy.max([float(key) for key in self.current_bids])
        smallestAsk = numpy.min([float(key) for key in self.current_asks])

        midPoint = (largestBid + smallestAsk) / 2.0

        histogram = {
            bucket: 0 for bucket in order_book_price_buckets
        }

        for price, amount in self.current_bids.items():
            relativePrice = float(price) / midPoint
            bucket = getOrderBookPriceBucket(relativePrice)

            histogram[bucket] += float(amount)

        for price, amount in self.current_asks.items():
            relativePrice = float(price) / midPoint

            bucket = getOrderBookPriceBucket(relativePrice)

            histogram[bucket] += float(amount)

        return {
            "_id": self.most_recent_message_time_processed,
            "time": self.most_recent_message_time_processed,
            "histogram": {
                str(bucket): volume for bucket, volume in histogram.items()
            }
        }

    def addAggregationHook(self, func):
        self.hooks.append(func)

