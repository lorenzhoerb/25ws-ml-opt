#!/bin/bash

min_n_groups=3
max_n_groups=8

min_n_per_group=2
max_n_per_group=5

min_n_rounds=2
max_n_rounds=10

instance_count=1

csv_file="instances.csv"

mkdir -p ./dzn

echo "instance_id;n_groups;n_per_group;rounds" > $csv_file

for (( group=min_n_groups; group<=max_n_groups; group++ ))
do
    for (( per_group=min_n_per_group; per_group<=max_n_per_group; per_group++ ))
    do
        for (( round=min_n_rounds; round<=max_n_rounds; round++ ))
        do
            instance_id="g${instance_count}"
            file="${instance_id}_${group}_${per_group}_${round}"
            echo "Generating instance ${file}..."

            # appends to instance.csv
            echo "${instance_id};${group};${per_group};${round}" >> $csv_file

            # generate MiniZinc data file 
            dzn_file=./dzn/${file}.dzn
            echo "" > $dzn_file
            echo "n_groups = ${group};" >> $dzn_file
            echo "n_groups = ${per_group};" >> $dzn_file
            echo "n_groups = ${round};" >> $dzn_file

            ((instance_count++))
        done
    done
done