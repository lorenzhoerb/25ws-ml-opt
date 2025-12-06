"""
This script evaluates Social Golfer Problem instances using two solvers:
1. MiniZinc (exact/optimization-based)
2. Simulated Annealing (heuristic)

For each instance, it measures:
- Whether a solution was found
- Solution cost
- Solver runtime

Results for all instances are collected in a DataFrame and saved to a CSV.
"""
import pandas as pd
from solvers import SimAnnealingGolferSolver, MiniZincGolferSolver
from utils import cost
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from pathlib import Path
from tqdm import tqdm
from datetime import timedelta
from typing import Optional

def evaluate_instance(instance: pd.Series, time_limit: Optional[timedelta] = None) -> pd.Series:
    """
    Evaluate a single SGP instance with both solvers.
    Measures cost and duration for each solver.
    
    :param instance: row from the instances dataframe
    :return: row enriched with solver metrics
    """

    # Instantiate solvers per process to avoid shared state issues
    mznSolver = MiniZincGolferSolver(model="./solvers/golfers.mzn", time_limit=time_limit)
    simSolver = SimAnnealingGolferSolver(T=200, alpha=0.998, loops=1000, time_limit=time_limit, stagnation_limit=None)

    n_groups = int(instance["n_groups"])
    n_per_group = int(instance["n_per_group"])
    n_rounds = int(instance["n_rounds"])

    
    # Run MiniZinc solver
    start = time.time()
    mzn_sol = mznSolver.solve(n_groups, n_per_group, n_rounds) 
    mzn_duration = time.time() - start
    mzn_cost = cost(mzn_sol) if mzn_sol is not None else 0

    
    # Run SimAnnealing solver
    start = time.time()
    sim_sol = simSolver.solve(n_groups, n_per_group, n_rounds) 
    sim_duration = time.time() - start
    sim_cost = cost(sim_sol) if sim_sol is not None else 0

    
    # Add metrics to row
    instance["mzn_has_solution"] = mzn_sol is not None
    instance["mzn_time_exceeded"] = mzn_duration > time_limit.seconds if time_limit else False
    instance["mzn_cost"] = mzn_cost
    instance["mzn_duration_seconds"] = round(mzn_duration, 4)

    instance["sim_has_solution"] = sim_sol is not None
    instance["sim_time_exceeded"] = sim_duration > time_limit.seconds if time_limit else False
    instance["sim_cost"] = sim_cost
    instance["sim_duration_seconds"] = round(sim_duration, 4)

    
    return instance

def evaluate_solvers(df_instances: pd.DataFrame, max_workers: int = 4, instance_time_limit: Optional[timedelta] = None):
    """
    Evaluate all instances in the CSV with MiniZinc and SimAnnealing solvers in parallel.
    Saves the enriched dataframe to a new CSV.
    
    :param instance_csv_path: Path to input instances CSV
    :param output_csv_path: Path to save enriched CSV
    :param max_workers: Number of parallel solver runs
    """
    # Load instances
    
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all instances for evaluation
        futures = [executor.submit(evaluate_instance, row.to_dict(), instance_time_limit) for _, row in df_instances.iterrows()]
        
        # Collect results as they finish
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error evaluating instance: {e}")
    
    # Convert results back to DataFrame
    return pd.DataFrame(results)

# Example usage
if __name__ == "__main__":
    df_instances = pd.read_csv("../data/instances/instances.csv")
    output_csv_path="../data/evaluation/solver_evaluation_results.csv"

    df_results = evaluate_solvers(
        df_instances,
        instance_time_limit=timedelta(seconds=2),
        max_workers=4
    )

    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_csv_path, index=False)
    print(f"Solver evaluation completed. Results saved to {output_csv_path}")