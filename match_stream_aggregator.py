from datetime import datetime
from pprint import pprint
from constants import relative_price_buckets, getPriceBucket


class MatchStreamAggregator:
    aggregation_keep_in_memory_seconds = 180

    def __init__(self, mongo_client):
        self.aggregations = {}

        self.mongo_client = mongo_client

        self.aggregated_matches_collection = self.mongo_client.trader.aggregated_matches

        self.most_recent_match_time_processed = None
        self.matches_processed = 0

    def processMatch(self, match):
        aggregation = self.getAggregationForMatch(match)
        relativePrice = float(match['price']) / aggregation['basePrice']

        priceBucket = getPriceBucket(relativePrice)

        aggregation['volumes'][str(priceBucket)] += float(match['size'])

        self.matches_processed += 1
        self.most_recent_match_time_processed = self.getTimeForMatch(match)

        if self.matches_processed % 1000 == 0:
            self.syncAggregations()
            self.removeOldAggregations()

    def getTimeForMatch(self, match):
        time = datetime.fromisoformat(match['time'][:match['time'].find('.')])
        minute = time.strftime("%Y-%m-%dT%H:%M:00")

        return minute

    def getAggregationForMatch(self, match):
        minute = self.getTimeForMatch(match)

        try:
            return self.aggregations[minute]
        except KeyError:
            aggregation = {
                "_id": datetime.fromisoformat(minute),
                "time": datetime.fromisoformat(minute),
                "basePrice": float(match['price']), # Todo, we need to find a better way of setting the base price to aggregate against
                "volumes": {
                    str(priceRange): 0 for priceRange in relative_price_buckets
                }
            }

            self.aggregations[minute] = aggregation

            return aggregation

    def syncAggregations(self):
        for aggregation in self.aggregations.values():
            self.aggregated_matches_collection.replace_one({"_id": aggregation['_id']}, aggregation, upsert=True)

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


