import pandas as pd
from pathlib import Path



def compute_instance_features(row: pd.Series) -> pd.Series:
    '''
    Computes a set of meta-features for a Social Golfer Problem instance.

    Each row corresponds to one problem instance and the following features
    are computed or included:

    Features:
    -----
    - n_groups: Number of groups in each round.
    - n_per_group: Number of players per group.
    - n_rounds: Number of rounds in the schedule.
    - n_players: Total number of players to schedule (n_groups * n_per_group).
    - groups_to_players: Ratio of groups to players (g / n_players).
    - rounds_to_players: Ratio of rounds to players (r / n_players).
    - distinct_pairs: Total number of distinct player pairs 
      that must be scheduled (n_players choose 2).
    - pair_capacity: Total number of player pairs that can be scheduled 
      across all rounds (r * g * (s choose 2)).
    '''

    g = row["n_groups"]
    s = row["n_per_group"]
    r = row["n_rounds"]

    p = g * s  # number of players

    row["n_players"] = p
    row["groups_to_players"] = g / p
    row["rounds_to_players"] = r / p
    row["distinct_pairs"] = p * (p-1) // 2
    row["pair_capacity"] = r * g * (s * (s - 1) // 2)
    return row

def compute_target(row: pd.Series) -> pd.Series:
    """
    Computes the target class for training:
    - 0 if MiniZinc is better or no solution could be found
    - 1 if Simulated Annealing is better

    Rules:
    1. If only one solver found a solution, pick that solver.
    2. If both found a solution, pick the one with lower cost.
    3. If costs are equal, pick the one with lower runtime.
    4. If both solvers didn't find a solution, pick minizinc (0) as default
    """
    
    # Check which solvers found a solution
    mzn_ok = row.get("mzn_has_solution", False)
    sim_ok = row.get("sim_has_solution", False)

    # Only one solver found solution
    if mzn_ok and not sim_ok:
        row["target"] = 0
    elif sim_ok and not mzn_ok:
        row["target"] = 1
    # Both found solution
    elif mzn_ok and sim_ok:
        mzn_cost = row.get("mzn_cost", float("inf"))
        sim_cost = row.get("sim_cost", float("inf"))
        
        if mzn_cost < sim_cost:
            row["target"] = 0
        elif sim_cost < mzn_cost:
            row["target"] = 1
        else:
            # Costs equal, pick faster
            mzn_time = row.get("mzn_duration_seconds", float("inf"))
            sim_time = row.get("sim_duration_seconds", float("inf"))
            row["target"] = 0 if mzn_time <= sim_time else 1
    else:
        # Neither found solution, mark as -1 or drop later
        row["target"] = 0

    return row



def extract_features_and_target(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Applies feature computation and target extraction for all instances.
    '''
    df = df.apply(compute_instance_features, axis=1)
    df = df.apply(compute_target, axis=1)
    return df

if __name__ == "__main__":
    evaluation_csv_path = "../data/evaluation/0002_solver_evaluation_results.csv"
    output_csv_path = "../data/training/training_data_v0002.csv"

    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)

    # Columns to keep for training
    id_col = ["instance_id"]

    feature_cols = [
        "n_groups",
        "n_per_group",
        "n_rounds",
        "n_players",
        "groups_to_players",
        "rounds_to_players",
        "distinct_pairs",
        "pair_capacity"
    ]

    target_col = ["target"] 

    df_eval = pd.read_csv(evaluation_csv_path)
    df_training = extract_features_and_target(df_eval)
    df_training = df_training[id_col + feature_cols + target_col]

    df_training.to_csv(output_csv_path, index=False)
    print(f"Training data saved to {output_csv_path}")