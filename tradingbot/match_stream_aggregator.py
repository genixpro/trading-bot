from datetime import datetime
from pprint import pprint
import numpy
import tradingbot.constants


class MatchStreamAggregator:
    aggregation_keep_in_memory_seconds = 180

    def __init__(self, mongo_client, saveToDisc=True):
        self.aggregations = {}

        self.mongo_client = mongo_client

        self.aggregated_matches_collection = self.mongo_client[tradingbot.constants.aggregated_data_database_name].aggregated_matches

        self.most_recent_match_time_processed = None
        self.matches_processed = 0

        self.saveToDisc = saveToDisc

        self.hooks = []

    def processMatch(self, match):
        aggregation = self.getAggregationForMatch(match)
        aggregation['matches'].append([float(match['price']), float(match['size'])])

        self.matches_processed += 1
        self.most_recent_match_time_processed = self.getMinuteForMatch(match)

        if self.matches_processed % 1000 == 0:
            self.syncAggregations()
            self.removeOldAggregations()

        if self.matches_processed % 100000 == 0:
            print(f"Processed {self.matches_processed:,} matches.")

    def getMinuteForMatch(self, match):
        minute = match['time'].strftime("%Y-%m-%dT%H:%M:00")

        return minute

    def getAggregationForMatch(self, match):
        minute = self.getMinuteForMatch(match)

        try:
            return self.aggregations[minute]
        except KeyError:
            aggregation = {
                "_id": datetime.fromisoformat(minute),
                "time": datetime.fromisoformat(minute),
                "matches": []
            }

            self.aggregations[minute] = aggregation

            return aggregation

    def syncAggregations(self):
        for aggregation in self.aggregations.values():
            averagePrice = numpy.average([match[0] for match in aggregation['matches']],
                                      weights=[match[1] for match in aggregation['matches']])

            # Do a weighted average to see what the typical price for this aggregation was
            aggregation['averagePrice'] = averagePrice

            if self.saveToDisc:
                self.aggregated_matches_collection.replace_one({"_id": aggregation['_id']}, aggregation, upsert=True)

            for hook in self.hooks:
                hook(aggregation)

    def removeOldAggregations(self):
        if self.most_recent_match_time_processed is None:
            return

        mostRecentTimeObj = datetime.fromisoformat(self.most_recent_match_time_processed)

        toDelete = []
        for key, aggregation in self.aggregations.items():
            timeDiff = (mostRecentTimeObj - datetime.fromisoformat(key))
            if timeDiff.total_seconds() > self.aggregation_keep_in_memory_seconds:
                toDelete.append(key)

        for key in toDelete:
            del self.aggregations[key]

    def addAggregationHook(self, func):
        self.hooks.append(func)

