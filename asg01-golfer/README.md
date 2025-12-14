# 🏌️‍♂️Social Golfer Problem 

## Problem Description 

The coordinator of a local golf club has come to you with the following problem. In her club, there are 32 social golfers, each of whom play golf once a week, and always in groups of 4. She would like you to come up with a schedule of play for these golfers, to last as many weeks as possible, such that no golfer plays in the same group as any other golfer on more than one occasion.“  

**General Problem:** Schedule `m groups` of `n golfers` over `p weeks`, such that no two golfers play in the same group more than once.

Full problem description: http://www.csplib.org/prob/prob010/index.html 

## Tasks

- **Task 1:** implement solution:
   - **Task 1.1:** using exact method (MiniZinc)
   - **Task 1.2:** using heuristic method (Simulated Annealing)
- **Task 2:** Implement a framework for automated algorithm selection that leverages machine learning techniques.

## Instances

An instance is define as the following tuple:

- `n_groups`            ... number of groups per round
- `n_per_group`         ... number of golfers in each group
- `n_rounds`            ... number of rounds played (weeks) to schedule

A instance has the form of 

```
(n_groups, n_per_group, n_rounds)
```

Instances are stored in `/instances`. 

Run `./mkInstances.sh` to generate:
- `instances.csv`:  list of all instance tuples
- `.dzn` files for MiniZinc in: `instances/dnz`

## Project Structure

- `data/`
  - `instances/`             — instance generation script and instances.csv 
    - `dzn/`                 - dzn instance files
  - `training/`              — ML training datasets
  - `evaluation/`            — evaluation outputs and metrics
- `src/`
  - `automl.ipynb`           — AutoML / algorithm-selection notebook and Comparison of Evaluation Results
  - `gen_training_data.py`   - Script to generate the training dataset. This includes feature extraction and target class selection
  - `evaluate_solvers.py`    - Script to evaluate the solvers on all instances
  - `sim_annealing_cli.py`   - CLI to run CGP with Simulated Annealing 
  - `utils.py`               - Includes the cost function 
  - `solvers/`               — solver implementations

## Quick Start

### Requirements
- Python 3.11 
- MiniZinc

Common commands:
- Make instances
  ```bash
  make genInstances
  ```
- Generate training data:
   ```bash
   python ./src/gen_training_data.py
   ```
- Evaluate solvers:
   ```bash
   python ./src/evaluate_solvers.py
   ```
- Run simulated annealing CLI:
  ```bash
  python ./src/sim_annealing_cli.py
  ```
- Open `src/automl.ipynb` to run auto ML and view comparison