import math 
import random
from itertools import combinations
from typing import Tuple, Dict
import numpy as np

def generate_initial_solution(n_groups: int, n_per_group: int, n_rounds: int) -> np.ndarray:
    n_golfers = n_groups * n_per_group
    schedule = np.zeros((n_rounds, n_groups, n_per_group), dtype=int)

    for r in range(n_rounds):
        # Random permutation of all golfers
        perm = np.random.permutation(n_golfers)
        # Split into groups
        schedule[r] = perm.reshape(n_groups, n_per_group)

    return schedule

def neighbor(solution: np.ndarray) -> np.ndarray :
    '''
    Generates a neighbor solution by choosing a random round and swapping the group of 
    two random golfers in distinct groups.
    '''
    neighbor = solution.copy()

    n_rounds, n_groups, n_per_group = neighbor.shape

    rmd_round = np.random.randint(0, n_rounds)

    rmd_group_1 = np.random.randint(0, n_groups)
    rmd_group_2 = np.random.randint(0, n_groups)
    while rmd_group_1 == rmd_group_2:
        # make sure that rmd_group_1 and rmd_group_2 are different
        rmd_group_2 = np.random.randint(0, n_groups)

    rmd_group_pos_1 = np.random.randint(0, n_per_group)
    rmd_group_pos_2 = np.random.randint(0, n_per_group)

    tmp = neighbor[rmd_round][rmd_group_1][rmd_group_pos_1]

    # swap golfer in different groups
    neighbor[rmd_round][rmd_group_1][rmd_group_pos_1] = neighbor[rmd_round][rmd_group_2][rmd_group_pos_2]
    neighbor[rmd_round][rmd_group_2][rmd_group_pos_2] = tmp

    return neighbor

def neighbor_v2(solution: np.ndarray) -> np.ndarray:
    neighbor = solution.copy()
    
    # Get pair counts
    pair_counts = get_pair_counts(neighbor)  # {pair: count}
    violating_pairs = [pair for pair, count in pair_counts.items() if count > 1]

    if not violating_pairs:
        return neighbor

    # Pick a random violating pair
    g1, g2 = violating_pairs[np.random.randint(len(violating_pairs))]
    golfer_to_move = np.random.choice([g1, g2])

    n_rounds, n_groups, n_per_group = neighbor.shape

    # Find all rounds where the golfer_to_move appears in a violating pair
    rounds_with_violation = [
        r for r in range(n_rounds)
        for group in neighbor[r]
        if golfer_to_move in group and any(p in group for p in (g1, g2))
    ]
    if not rounds_with_violation:
        rmd_round = np.random.randint(n_rounds)
    else:
        rmd_round = np.random.choice(rounds_with_violation)

    # Find the group of golfer_to_move in that round
    group_idx = next(
        i for i, group in enumerate(neighbor[rmd_round]) if golfer_to_move in group
    )
    pos_in_group = np.where(neighbor[rmd_round][group_idx] == golfer_to_move)[0][0]

    # Pick another group to swap with (not the same group)
    other_group_idx = np.random.choice([i for i in range(n_groups) if i != group_idx])
    other_pos = np.random.randint(n_per_group)

    # Swap the golfers
    neighbor[rmd_round][group_idx][pos_in_group], neighbor[rmd_round][other_group_idx][other_pos] = \
        neighbor[rmd_round][other_group_idx][other_pos], neighbor[rmd_round][group_idx][pos_in_group]

    return neighbor



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


def solve_golfer(n_groups: int, n_per_group: int, n_rounds: int) -> Tuple[np.ndarray, float, bool]:
    '''
    Tries to solves the given golfer instance using simulated annealing.
    
    Returns:
        - schedule (np.ndarray): Final 3D schedule array (rounds × groups × players)
        - cost (float): Cost of the final schedule
        - success (bool): True if a valid solution was found (cost == 0)
    '''
    T=200
    alpha=0.98
    min_T=1e-3
    inner_loops=200

    current_solution = generate_initial_solution(n_groups, n_per_group, n_rounds)
    current_cost = cost(current_solution)

    best_solution = current_solution.copy()
    best_cost = current_cost

    while T > min_T:
        for _ in range(inner_loops):

            new_solution = neighbor(current_solution)
            new_cost = cost(new_solution)
            delta = new_cost - current_cost

            # Accept if better OR probabilistically if worse
            if delta <= 0 or random.random() < math.exp(-delta/T):
                current_solution = new_solution

                # Track global best
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_solution = new_solution.copy()
            
            if best_cost == 0:
                # stopping criteria: found optimal solution
                return best_solution, best_cost, best_cost == 0

        T *= alpha


    return best_solution, best_cost, best_cost == 0