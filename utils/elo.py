import numpy as np


def calculate_elo_change(rating_a, rating_b, result_a, k_factor=32):
    expected_a = 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))
    change_a = k_factor * (result_a - expected_a)
    return change_a


def update_elo_ratings(rating_a, rating_b, result_a, k_factor=32):
    change_a = calculate_elo_change(rating_a, rating_b, result_a, k_factor)
    new_rating_a = rating_a + change_a
    new_rating_b = rating_b - change_a
    return new_rating_a, new_rating_b


def softmax_weights(elos, temperature=1.0):
    if len(elos) == 0:
        return np.array([])
    elos = np.array(elos, dtype=np.float64)
    scaled_elos = elos / temperature
    exp_elos = np.exp(scaled_elos - np.max(scaled_elos))
    weights = exp_elos / np.sum(exp_elos)
    return weights

