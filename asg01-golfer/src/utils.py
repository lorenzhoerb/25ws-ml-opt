import numpy as np 
from itertools import combinations
from typing import Dict, Tuple

def cost(solution: np.ndarray) -> float:
    '''
    Calculates the the cost for the given solution. Penalizes golfer pairs that play together more than once
    '''
    cost = 0
    pair_counts = get_pair_counts(solution)

    # penalize repeated pairings
    for _, cnt in pair_counts.items():
        if cnt > 1:
            cost += (cnt - 1) ** 2 # first pairing is allowed, extras are bad

    return cost

def get_pair_counts(solution: np.ndarray) -> Dict[Tuple[int, int], int]:
    pair_counts = {}

    for r in solution:
        for group in r:
            for g1, g2 in combinations(group, 2):
                pair = tuple(sorted((g1, g2)))
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
    return pair_counts
