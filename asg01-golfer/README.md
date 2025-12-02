# 🏌️‍♂️Social Golfer Problem 

## Problem Description 

The coordinator of a local golf club has come to you with the following problem. In her club, there are 32 social golfers, each of whom play golf once a week, and always in groups of 4. She would like you to come up with a schedule of play for these golfers, to last as many weeks as possible, such that no golfer plays in the same group as any other golfer on more than one occasion.“  

**General Problem:** Schedule `m groups` of `n golfers` over `p weeks`, such that no two golfers play in the same group more than once.

Full problem description: http://www.csplib.org/prob/prob010/index.html 

## Approaches 

This project implements two solution strategies:

1. Exact Method: MiniZinc
2. Heuristic: Simulated Annealing

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
