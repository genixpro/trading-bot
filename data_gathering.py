import cbpro
from pprint import pprint
import time
from pymongo import MongoClient
from datetime import datetime, timedelta

mongo_client = MongoClient('mongodb://localhost:27017/')

class TraderDataGatherer(cbpro.WebsocketClient):
    def on_open(self):
        self.url = "wss://ws-feed.pro.coinbase.com/"
        self.products = ["ETH-USD"]
        self.channels = ["matches", "level2"]
        self.matches_collection = mongo_client.trader.matches
        self.order_book_collection = mongo_client.trader.order_book

        self.snapshotToSave = None

        self.matches_collection.create_index("time")

    def on_message(self, msg):
        if msg['type'] == 'match':
            self.matches_collection.insert_one(msg)
        if msg['type'] == 'snapshot':
            self.snapshotToSave = msg
        if msg['type'] == 'l2update':
            time = datetime.fromisoformat(msg['time'][:msg['time'].find('.')])
            msg['time'] = time

            if self.snapshotToSave is not None:
                self.snapshotToSave['time'] = (time - timedelta(milliseconds=50))
                self.order_book_collection.insert_one(self.snapshotToSave)
                self.snapshotToSave = None

            self.order_book_collection.insert_one(msg)
    def on_close(self):
        pass


wsClient = TraderDataGatherer()
wsClient.start()
# time.sleep(60)
# wsClient.close()




# wsClient = cbpro.WebsocketClient(
#     url="wss://ws-feed.pro.coinbase.com",
#     products="BTC-USD",
#     mongo_collection=mongo_client.trader.matches,
#     channels=['matches'],
#     should_print=True
# )




# wsClient.start()






# key = "7dcefb9d7a59d723bb0c6d2467fd7225"
# secret = "Pqxh4/c7/+4sZB6qGDAtPGn354b923gYelupdah6+HccOp6zfghg+9qcDTV5H4+laMwxhgs2UbMafVoy8YN4Ew=="
# passphrase = "h7ghsarydw"
#
#
# auth_client = cbpro.AuthenticatedClient(key, secret, passphrase)
