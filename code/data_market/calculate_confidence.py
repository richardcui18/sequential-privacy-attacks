from itertools import product

def confidence_upper_bound(true_intent, published_intent):
    length = len(true_intent)
    prod = 1.0
    for i in range(length):
        true_intent_i = true_intent[i]
        published_intent_i = published_intent[i]
        prod *= len(true_intent_i) / len(published_intent_i)
    return prod


def confidence_lower_bound(published_intent):
    length = len(published_intent)
    prod = 1.0
    for i in range(length):
        published_intent_i = published_intent[i]
        prod *= 1 / len(published_intent_i)
    return prod


def lambda_privacy_published_intent_lower_bound(lambda_value, true_intent):
    length = len(true_intent)
    prod = 1.0
    for i in range(length):
        true_intent_i = true_intent[i]
        prod *= len(true_intent_i)
    return prod / lambda_value


def compute_PI_size(published_intent):
    PI = list(product(*published_intent))
    return len(PI)