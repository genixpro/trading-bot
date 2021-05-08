from tradingbot.data_gathering import TraderDataGatherer



def main():
    wsClient = TraderDataGatherer(saveToDisc=True)
    wsClient.start()



if __name__ == "__main__":
    main()
