import numpy
from pprint import pprint

raw_data_database_name = "trader"
aggregated_data_database_name = "trader2"
my_orders_database_name = "trader"

match_price_buckets = [
    (0.0000, 0.9950),
    (0.9950, 0.9960),
    (0.9960, 0.9970),
    (0.9970, 0.9980),
    (0.9980, 0.9985),
    (0.9985, 0.9990),
    (0.9990, 0.9995),
    (0.9995, 1.0000),
    (1.0000, 1.0005),
    (1.0005, 1.0010),
    (1.0010, 1.0015),
    (1.0015, 1.0020),
    (1.0020, 1.0030),
    (1.0030, 1.0040),
    (1.0040, 1.0050),
    (1.0050, 100000),
]

def getMatchPriceBucket(relativePrice):
    for bucket in match_price_buckets:
        if relativePrice >= bucket[0] and relativePrice < bucket[1]:
            return bucket
    raise ValueError(f"Unable to find a match price bucket for relative price {relativePrice}")


volume_buckets = [
    (0, 0.0001),
    (0.0001, 1000000),
    # (0.0001, 1),
    # (1, 2),
    # (2, 3),
    # (3, 5),
    # (5, 10),
    # (10, 20),
    # (20, 40),
    # (40, 80),
    # (80, 160),
    # (160, 1000000),
]

def getVolumeBucket(volume):
    for bucket in volume_buckets:
        if volume >= bucket[0] and volume < bucket[1]:
            return bucket
    raise ValueError(f"Unable to find a volume bucket for volume {volume}")






order_book_price_range_start = 0.98
order_book_price_range_increment = 0.002
order_book_price_range_end = 1.02

order_book_price_buckets = [
    (0.0000, order_book_price_range_start)
]

for v in numpy.arange(order_book_price_range_start, order_book_price_range_end, order_book_price_range_increment):
    order_book_price_buckets.append((
        float(f"{v:.4f}"),
        float(f"{v + order_book_price_range_increment:.4f}"),
    ))

order_book_price_buckets.append((
    order_book_price_buckets[-1][1],
    100000000,
))

def getOrderBookPriceBucket(relativePrice):
    for bucket in order_book_price_buckets:
        if relativePrice >= bucket[0] and relativePrice < bucket[1]:
            return bucket
    raise ValueError(f"Unable to find a order book price bucket for relative price {relativePrice}")


prediction_intervals = [
    (2, 30),
    # (5, 10),
    # (10, 15)
]


prediction_sequence_input_length = 60
