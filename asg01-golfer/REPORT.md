# Report Social Golfer Problem

The Social Golfers Problem was solved with MiniZinc (for exact method) and Simulated Annealing (heuristic method). 

## Instances & Generation

### Instances

An instance is define as the following tuple:
- `n_groups`            ... number of groups per round
- `n_per_group`         ... number of golfers in each group
- `n_rounds`            ... number of rounds played (weeks) to schedule

### Generation

In total 216 instances were generated using the `data/instances/mkInstances.sh`. The instance space is defined as 

- n_groups=3..8
- n_per_group=2..5
- n_rounds=2..10

All instances are saved in an `instances.csv` file, and a corresponding MiniZinc data file (.dzn) was created for each instance in `data/instances/dzn`.

## Cost Function

The cost function evaluates a candidate solution by penalizing repeated pairings of golfers. Its goal is to encourage solutions where no two golfers play together more than once. 

- For every pair of golfers, we count how many times they appear in the same group.
-	If a pair plays together more than once, we add a penalty. The more extra times they play together, the higher the penalty

$$
\text{cost} = \sum_{(i,j)} (\max(0, \text{times\_together}(i,j) - 1))^2
$$

- **0 penalty** means no golfer pair repeats.  
- **Higher cost** means more repeated pairings.

## Solvers

Solvers can be found in `src/solvers`. 

### MiniZinc

The MiniZinc solver is define as an optimization that minimizes the cost. It can be in `golfers.mzn` and its python wrapper in `src/solvers/mzn_solver.py`.

### Simulated Annealing

The Simulated Annealing solver can be found in `src/solvers/sim_annealing.py` and supports the hyper parameters:

- `T` … Initial temperature  
- `min_T` … Minimum temperature to stop cooling  
- `alpha` … Cooling rate  
- `loops` … Iterations per temperature  
- `stagnation_limit` … Max loops without improvement (optional)  
- `time_limit` … Maximum runtime (optional)  
- `cost` … Cost function to evaluate solutions


## Evaluation

The solvers were evaluated against instances and the runtime, cost, and if an solution has been found was recorded. This is done by the scrip `src/evaluate_solver.py`. It is multi-threaded and saves the results in `data/evaluation`.

## Training Dataset Generation

The script `gen_training_data.py` generates the training data. It uses the evaluation results in `data/evaluation` and performs feature extraction and sets the target class.

### Features Extraction 

Following features are extracted:

- `n_groups` ... Number of groups per round  
- `n_per_group` ...  Number of players per group  
- `n_rounds` ... Number of rounds to schedule  
- `n_players` ... Total number of players (`n_groups * n_per_group`)  
- `groups_to_players` ... Ratio of groups to players (`n_groups / n_players`)  
- `rounds_to_players` ... Ratio of rounds to players (`n_rounds / n_players`)  
- `distinct_pairs` ... Total number of distinct player pairs (`n_players choose 2`)  
- `pair_capacity` ... Total number of pairs that can be scheduled across all rounds (`n_rounds * n_groups * (n_per_group choose 2)`)


### Target Label

Selects the better perform in algorithm.

- `0` … MiniZinc
- `1` … Simulated Annealing

The rules for computing the label are:

1. If only one solver found a solution, select that solver.  
2. If both solvers found a solution, select the one with lower cost.  
3. If costs are equal, select the one with lower runtime.  
4. If neither solver found a solution, default to MiniZinc (`0`). 

## AutoML 

The algorithm selection was done with a RandomForest classifier. 

The training data is based on `data/evaluation/0002_solver_evaluation_results.csv`. The evaluation environment was as follows:
- instances: 216
- time_limit: 20 second
- processing time: 15 minutes 

Hyper Parameters Simulated Annealing:
- T: 200
- alpha: 0.998
- loops: 1000
- stagnation_limit: 500


The challenge was to deal with the highly unbalanced  target class. A macro avg F1 score of 63% was achieved. 

The most influential / important feature was pair_capacity after the evaluation. 

> For more detailed comparison of Minizinc, Simulated Annealing and AutoML and result diagrams refer to [src/automl.ipynb](src/automl.ipynb)

## Conclusion

This study evaluated MiniZinc and Simulated Annealing for solving the Social Golfer Problem, with both methods restricted to a 20-second execution limit. MiniZinc returns the best feasible solution if the optimal solution is not found within the time limit, while Simulated Annealing employs early termination based on stagnation thresholds and temperature-based cooling, often completing in under one second.

The results indicate that Simulated Annealing generally outperforms MiniZinc in terms of solution quality and computational efficiency for this evaluation setting. Observation are:

- **Optimal Solution Rate:** The proportion of instances achieving a zero-cost (optimal) solution is higher for Simulated Annealing, particularly as the number of rounds increases. MiniZinc’s performance declines with increasing problem complexity. 
- **Mean and Median Cost:** Simulated Annealing achieves an average cost of approximately 67.70, compared to MiniZinc’s mean of 551.60.

| Metric                  | MiniZinc  | Sim     |
|-------------------------|-----------|---------|
| Optimal Solution Rate    | 21.75%      | 23.15%|
| Time Exceeded Rate       | 78.24%      | 0%    |
| Mean Duration (s)        | 19.46     | 0.12    |
| Median Duration (s)      | 20.98     | 0.11    |
| Mean Cost                | 551.60    | 67.70   |
| Median Cost              | 120.50    | 18.00   |

An AutoML-based selector was trained on instance features to predict the better-performing solver. The model achieved a macro F1 score of 0.66. Its performance is limited by significant class imbalance, as MiniZinc outperformed Simulated Annealing in only 5 of the 140 training instances, making generalization challenging.


> For more detailed comparison of Minizinc, Simulated Annealing and AutoML and result diagrams refer to [src/automl.ipynb](src/automl.ipynb)