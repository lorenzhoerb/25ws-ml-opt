# Conclusion

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



> For more detail comparison and diagrams refer to `src/automl.ipynb` 