from minizinc import Model, Solver, Instance
from datetime import timedelta
import numpy as np
from typing import Optional


class MiniZincGolferSolver:
    '''
    Solver for the Social Golfer Problem using minizinc.
    Minizinc must be installed locally and the solver must be available.
    List available solvers using `minizinc --solvers`
    '''

    def __init__(
            self, 
            model: str = "./golfers.mzn", 
            solver: str = "gecode",
            time_limit: timedelta = None
        ) -> None :

        self.model = Model(model)
        self.solver = Solver.lookup(solver)
        self.time_limit = time_limit

    def solve(self, n_groups: int, n_per_group: int, n_rounds: int) -> Optional[np.ndarray]:
        assert n_groups  > 0, "n_groups must be > 0"
        assert n_per_group  > 0, "n_per_group must be > 0"
        assert n_rounds  > 0, "n_rounds must be > 0"

        instance = Instance(self.solver, self.model)
        instance["n_groups"] = n_groups
        instance["n_per_group"] = n_per_group
        instance["n_rounds"] = n_rounds

        try:
            result = instance.solve(time_limit=self.time_limit)
        except Exception as e:
            print(f"Solver failed: {e}")
            return None

        if result is None:
            return None

        if not result.status.has_solution():
            return None

        schedule = result["schedule"]
        if not schedule:
            return None
        
        return np.array(schedule)