include "globals.mzn";

% data parameters 
int: n_groups;
int: n_per_group;
int: n_rounds;

%% constraint checks
constraint assert(n_groups >= 1, "Invalid datafile: " ++
                    "n_groups should be at least 1");
constraint assert(n_per_group >= 1, "Invalid datafile: " ++
                    "n_per_group should be at least 1");
constraint assert(n_rounds >= 1, "Invalid datafile: " ++
                    "n_rounds should be at least 1");

int: n_golfers = n_groups * n_per_group;


% index sets
set of int: Groups = 1..n_groups;
set of int: GroupCapacity = 1..n_per_group;
set of int: Golfers = 1..n_golfers;
set of int: Rounds = 1..n_rounds;


array[Rounds,Groups,GroupCapacity] of var Golfers: schedule;

% C1: Ensure that every golfer is scheduled exactly once in each round
constraint forall(r in Rounds)(
    alldifferent(gr in Groups, c in GroupCapacity) (schedule[r,gr,c])
);

% C2: Ensure that any pair of golfers plays together at most once across all rounds
% for any two golfers g1 and g2 where g1 != g2, they can't play in a group more then once
constraint forall(g1, g2 in Golfers where g1 < g2)(
    count(r in Rounds, gr in Groups, c, c2 in GroupCapacity where c < c2)(
        schedule[r,gr,c] == g1 /\
        schedule[r,gr,c2] == g2
    ) <= 1
);


solve satisfy;


% Readable output
output [
  "Social Golfer Schedule:\n\n"
] ++
[
  "Round " ++ show(r) ++ ": " ++
  let {
    string: round_output = concat([
      "[" ++
      concat([
        show(schedule[r,g,p]) ++ (if p < n_per_group then "," else "" endif)
        | p in GroupCapacity
      ]) ++ "] "
      | g in Groups
    ])
  } in round_output ++ "\n"
  | r in Rounds
];