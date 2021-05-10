from tradingbot.bot import TradingBot



def main():
    bot = TradingBot()

    bot.clearHistoricMatchCollections()
    print("Processing the order book stream")
    bot.processHistoricOrderBookStream()

    print("Processing the matches stream")
    bot.processHistoricMatches()




if __name__ == "__main__":
    main()
