import sys
from sim_annealing import SimAnnealingGolferSolver

def print_usage_and_exit():
    print(f"USAGE:")
    print(f"    {sys.argv[0]} <n_groups> <n_per_group> <n_rounds>")
    sys.exit(1)

def fatal_error(msg: str):
    print(f"ERROR: {msg}")
    sys.exit(1)

def parse_var(str_var, var_name):
    var = -1
    try:
        var = int(str_var)
    except:
        fatal_error(f"failed to parse variable '{var_name}': {str_var} must be a number")

    if var < 1:
        fatal_error(f"'{var_name}' must be >= 1: received {var}")
    return var

def parse_instance():
    n_groups = parse_var(sys.argv[1], "n_groups")
    n_per_group = parse_var(sys.argv[2], "n_per_group")
    n_rounds = parse_var(sys.argv[3], "n_rounds")

    return n_groups, n_per_group, n_rounds

    

def main():
    if len(sys.argv) != 4:
        print_usage_and_exit()

    n_groups, n_per_group, n_rounds = parse_instance()


    solver = SimAnnealingGolferSolver(T=300, loops=1000, alpha=0.998)
    print(f"Solving instance (n_groups={n_groups}, n_per_group={n_per_group}, n_rounds={n_rounds}) with hyperparameters (T={solver.T}, min_T={solver.min_T}, alpha={solver.alpha}, loops={solver.loops})...")

    sol, cost, is_optimal = solver.solve(n_groups, n_per_group, n_rounds)

    print("--------------------------")

    print(f"Optimal Solution:\t", is_optimal)
    print(f"Cost:\t\t\t", cost)
    print("Solution: \n")
    print(sol)



if __name__ == "__main__":
    main()





