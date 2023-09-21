import cbpro
from pprint import pprint
import time
from pymongo import MongoClient
from datetime import datetime, timedelta
import tradingbot.constants

import asyncio
from binance import AsyncClient, BinanceSocketManager


mongo_client = MongoClient('mongodb://localhost:27017/')

class Arbitrager(cbpro.WebsocketClient):
    def __init__(self):
        super(Arbitrager, self).__init__()

    def on_open(self):
        self.url = "wss://ws-feed.pro.coinbase.com/"
        self.products = ["ETH-USD", "BTC-USD", "ETH-BTC"]
        self.channels = ["ticker"]

        self.lastETHUSD = 1
        self.lastBTCUSD = 1
        self.lastETHBTC = 1

    def on_message(self, msg):
        if msg['type'] == 'ticker':
            if msg['product_id'] == 'ETH-USD':
                self.lastETHUSD = float(msg['price'])
            if msg['product_id'] == 'BTC-USD':
                self.lastBTCUSD = float(msg['price'])
            if msg['product_id'] == 'ETH-BTC':
                self.lastETHBTC = float(msg['price'])
                print(datetime.now(), 'coinbase', f"{self.lastETHBTC:.6f}")

                # self.checkArbitrage()

    def checkArbitrage(self):
        pair1 = (self.lastETHUSD / self.lastETHBTC) - self.lastBTCUSD
        # print("profit 1 ", pair1)


    def on_close(self):
        pass


a = Arbitrager()
a.start()


async def main():
    client = await AsyncClient.create()
    bm = BinanceSocketManager(client)
    # start any sockets here, i.e a trade socket
    ts = bm.symbol_ticker_socket('ETHBTC')
    # then start receiving messages
    async with ts as tscm:
        while True:
            res = await tscm.recv()
            print(datetime.now(), 'binance ', f"{float(res['b']):.6f}")

    await client.close_connection()

if __name__ == "__main__":

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())






