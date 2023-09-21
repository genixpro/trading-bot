import cbpro
from pprint import pprint
import time
from pymongo import MongoClient
from datetime import datetime, timedelta
import tradingbot.constants

mongo_client = MongoClient('mongodb://localhost:27017/')

class TraderDataGatherer(cbpro.WebsocketClient):
    def __init__(self, saveToDisc=True):
        super(TraderDataGatherer, self).__init__()

        self.saveToDisc = saveToDisc

        self.match_hooks = []
        self.order_book_hooks = []

    def on_open(self):
        self.url = "wss://ws-feed.pro.coinbase.com/"
        self.products = ["ETH-USD"]
        self.channels = ["matches", "level2"]
        self.matches_collection = mongo_client[tradingbot.constants.raw_data_database_name].matches
        self.order_book_collection = mongo_client[tradingbot.constants.raw_data_database_name].order_book

        self.snapshotToSave = None

        self.matches_collection.create_index("time")

    def on_message(self, msg):
        if msg['type'] == 'match':
            time = datetime.fromisoformat(msg['time'][:msg['time'].find('.')])
            msg['time'] = time
            if self.saveToDisc:
                self.matches_collection.insert_one(msg)

            for hook in self.match_hooks:
                hook(msg)

        if msg['type'] == 'snapshot':
            self.snapshotToSave = msg
        if msg['type'] == 'l2update':
            time = datetime.fromisoformat(msg['time'][:msg['time'].find('.')])
            msg['time'] = time

            if self.snapshotToSave is not None:
                self.snapshotToSave['time'] = (time - timedelta(milliseconds=5))
                if self.saveToDisc:
                    self.order_book_collection.insert_one(self.snapshotToSave)
                for hook in self.order_book_hooks:
                    hook(self.snapshotToSave)
                self.snapshotToSave = None

            if self.saveToDisc:
                self.order_book_collection.insert_one(msg)
                for hook in self.order_book_hooks:
                    hook(msg)

    def add_match_hook(self, func):
        self.match_hooks.append(func)

    def add_order_book_hook(self, func):
        self.order_book_hooks.append(func)

    def on_close(self):
        pass


# wsClient = cbpro.WebsocketClient(
#     url="wss://ws-feed.pro.coinbase.com",
#     products="BTC-USD",
#     mongo_collection=mongo_client.trader.matches,
#     channels=['matches'],
#     should_print=True
# )




# wsClient.start()






# key = ""
# secret = ""
# passphrase = ""
#
#
# auth_client = cbpro.AuthenticatedClient(key, secret, passphrase)
