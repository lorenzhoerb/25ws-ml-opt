import math 
import random
from typing import Tuple, Callable
import numpy as np
from utils import cost, get_pair_counts

class SimAnnealingGolferSolver:
    '''
    A heuristic solver for the Social Golfer Problem using Simulated Annealing. 
    '''

    def __init__(
        self,
        T: float = 100,
        min_T: float = 1e-3,
        alpha: float = 0.98,
        loops: int = 100,
        cost: Callable[[np.ndarray], float] = cost,
    ) -> None:
        self.T: float = T
        self.min_T: float = min_T
        self.alpha: float = alpha
        self.loops: int = loops

        if cost is None:
            raise ValueError("A cost function must be provided")

        self.cost: Callable[[np.ndarray], float] = cost

    def solve(self, n_groups: int, n_per_group: int, n_rounds: int) -> Tuple[np.ndarray, float, bool]:
        '''
        Tries to solves the given golfer instance using simulated annealing.
        
        Returns:
            - schedule (np.ndarray): Final 3D schedule array (rounds × groups × players)
            - cost (float): Cost of the final schedule
            - success (bool): True if a valid solution was found (cost == 0)
        '''
        local_T = self.T

        current_solution = self._generate_initial_solution(n_groups, n_per_group, n_rounds)
        current_cost = cost(current_solution)

        best_solution = current_solution.copy()
        best_cost = current_cost

        while local_T > self.min_T:
            for _ in range(self.loops):

                new_solution = self._neighbor(current_solution)
                new_cost = self.cost(new_solution)
                delta = new_cost - current_cost

                # Accept if better OR probabilistically if worse
                if delta <= 0 or random.random() < math.exp(-delta/local_T):
                    current_solution = new_solution

                    # Track global best
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_solution = new_solution.copy()
                
                if best_cost == 0:
                    # stopping criteria: found optimal solution
                    return best_solution, best_cost, best_cost == 0

            local_T *= self.alpha


        return best_solution, best_cost, best_cost == 0

    def _generate_initial_solution(self, n_groups: int, n_per_group: int, n_rounds: int) -> np.ndarray:
        n_golfers = n_groups * n_per_group
        schedule = np.zeros((n_rounds, n_groups, n_per_group), dtype=int)

        for r in range(n_rounds):
            # Random permutation of all golfers
            perm = np.random.permutation(n_golfers)
            # Split into groups
            schedule[r] = perm.reshape(n_groups, n_per_group)

        return schedule

    def _neighbor(self, solution: np.ndarray) -> np.ndarray :
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

    def _neighbor_v2(self, solution: np.ndarray) -> np.ndarray:
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