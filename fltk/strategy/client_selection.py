import numpy as np


def random_selection(clients, n):
    return np.random.choice(clients, n, replace=False)


def balanced_sampling(clients, groups, round):
    sample = []
    for idx in groups[round % len(groups)]:
        sample.append(clients[idx])

    return sample
