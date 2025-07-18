import pandas as pd
import os
import math
from itertools import combinations
import hmm_rl
import baseline_algorithm
import load_dataset
import itertools
import json
from options import Options
import sys
import draw_graph
sys.stdout.reconfigure(line_buffering=True)

# helper function to generate all possible subsets given a PI state, with size less than some threshold (max TI size)
def generate_subsets_from_pi(state, max_size, min_size):
    def get_subsets(dim, current_subset, current_size):
        if dim == len(state):
            if min_size <= current_size <= max_size:
                result.append(current_subset)
            return
        
        for size in range(1, len(state[dim]) + 1):
            for combination in combinations(state[dim], size):
                new_size = current_size * size
                if new_size <= max_size:
                    get_subsets(dim + 1, current_subset + [list(combination)], new_size)
    
    result = []
    get_subsets(0, [], 1)
    return result

# helper function to determine whether a TI state is a subset of a PI state
def is_subset(ti_state, pi_state):
    for dim1, dim2 in zip(ti_state, pi_state):
        if not set(dim1).issubset(set(dim2)):
            return False
    return True

def generate_possible_ti_states_from_pi_states(pi_true_sequence_with_feature_names, beta):
    # calculate PI sizes
    pi_sizes = []
    for pi in pi_true_sequence_with_feature_names:
        cur_size = 1
        for pi_dim in pi:
            cur_size *= len(pi_dim)
        pi_sizes.append(cur_size)

    # calculate TI size upper bounds
    ti_sizes_upper_bound = [math.floor(pi_size * lambda_value) for pi_size in pi_sizes]

    # calculate possible TI states
    possible_ti_states = []
    for i in range(len(pi_true_sequence_with_feature_names)):
        lower_bound = ti_sizes_upper_bound[i] - beta
        if lower_bound < 1:
            lower_bound = 1
        possible_ti_states += generate_subsets_from_pi(pi_true_sequence_with_feature_names[i], ti_sizes_upper_bound[i], lower_bound)
        possible_ti_states_lengths = []
        for state in possible_ti_states:
            size = 1
            for dim in state:
                size *= len(dim)
            possible_ti_states_lengths.append(size)
    for ti_state in possible_ti_states.copy():
        # Count the number of observed PI states that this TI state is a subset of
        subset_count = 0
        for pi_state in pi_true_sequence_with_feature_names:
            if is_subset(ti_state, pi_state):
                subset_count += 1
        # remove TI states that are not a subset of any observed PI states
        if subset_count == 0:
            possible_ti_states.remove(ti_state)
    
    return possible_ti_states, ti_sizes_upper_bound

def generate_grids_from_state(state_representation):
    combinations = list(itertools.product(*state_representation))
    grids_list = [list(map(lambda x: [x], combination)) for combination in combinations]
    
    return grids_list

if __name__ == '__main__':
    args = Options()
    args = args.parser.parse_args()

    # Setup
    total_time = args.total_time
    lambda_value = args.lambda_value
    seed = 65

    # Initialize data
    adult_data = pd.read_csv('data/adult_processed_5_attributes_1.csv')
    data_cube, unique_values_on_each_dimension = load_dataset.df2data_cube_five_attributes(adult_data)
    cost_cube = load_dataset.generate_unit_cost_data_cube_same_price(data_cube)
    ti_true_sequence_with_feature_names = load_dataset.create_true_intent_random_walk(unique_values_on_each_dimension, total_time, seed)

    # Generate true TI states using random walk
    ti_true_sequence = []
    for time in range(len(ti_true_sequence_with_feature_names)):
        possible_grids = generate_grids_from_state(ti_true_sequence_with_feature_names[time])
        ti_true_sequence_at_time_i = []
        for grid_features in possible_grids:
            grid_index = []
            for dim in range(len(grid_features)):
                grid_index.append(unique_values_on_each_dimension[dim].index(grid_features[dim][0]))
            ti_true_sequence_at_time_i.append(grid_index)
        ti_true_sequence.append(ti_true_sequence_at_time_i)
    
    ti_true_lengths = [len(ti_state) for ti_state in ti_true_sequence]

    # Generate true PI states from TI states using random sampling
    pi_true_sequence_with_feature_names = load_dataset.generate_pi_states_from_ti_states_random_sample(ti_true_sequence_with_feature_names, ti_true_lengths,
                                                                                                        unique_values_on_each_dimension, lambda_value, seed)
    
    pi_true_lengths = []
    for pi_state in pi_true_sequence_with_feature_names:
        size = 1
        for pi_state_dim in pi_state:
            size *= len(pi_state_dim)
        pi_true_lengths.append(size)
    
    ### ATTACK
    verbose = args.verbose
    verbose = verbose.lower() == "true"
    seed_attacker = 10
    num_iter = args.num_iter

    # hyperparameters
    # deltas = [0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
    # betas = [0,2,5,10,15]
    # k = [0,1,2,3,4]
    # gammas = [0,3,5,6,8]

    deltas = [0.7]
    betas = [0]
    k = [3]
    gammas = [5]

    all_results = {
        "config": {
            "deltas": deltas,
            "betas": betas,
            "window_sizes": k,
            "gammas": gammas,
            "lambda_value": lambda_value,
            "total_time": total_time,
            "num_iter": num_iter,
            "seed_attacker": seed_attacker,
            "verbose": verbose
        },
        "experiments": []
    }
    for delta in deltas:
        for beta in betas:
            for window_size in k:
                for pi_size_bound_threshold_attacker in gammas:
                    print("************************************")
                    print(f"Running HMM-RL algorithm with delta ({delta}), beta ({beta}), k ({window_size}), and gamma ({pi_size_bound_threshold_attacker})...")
                    # generate possible TI states
                    possible_ti_states, ti_sizes_upper_bound = generate_possible_ti_states_from_pi_states(pi_true_sequence_with_feature_names, beta)
                    
                    # run RL-based algorithm
                    results_weighted = hmm_rl.run_rl_algorithm(pi_true_sequence_with_feature_names, possible_ti_states, ti_sizes_upper_bound, 
                                    lambda_value, total_time, delta, ti_true_sequence, num_iter, unique_values_on_each_dimension, 
                                    ti_true_sequence_with_feature_names, pi_size_bound_threshold_attacker, seed_attacker, window_size, verbose)
                    
                    f1_score_list, f1_score = hmm_rl.ti_prediction_f1_score(ti_true_sequence, results_weighted[-1], True, unique_values_on_each_dimension)

                    # run baseline algorithm
                    f1_score_list_bl, f1_score_bl = baseline_algorithm.pi_uniform_attack(total_time, pi_true_sequence_with_feature_names, ti_sizes_upper_bound, 
                                                                                        lambda_value, ti_true_sequence, unique_values_on_each_dimension)

                    print("Baseline F1 Score:", f1_score_bl)
                    print('HMM-RL F1 Score:', f1_score)
                    print("************************************")

                    experiment_result = {
                        "parameters": {
                            "delta": delta,
                            "beta": beta,
                            "k": window_size,
                            "gamma": pi_size_bound_threshold_attacker
                        },
                        "hmm_rl_result": {
                            "f1_score": f1_score,
                            "f1_score_list_all_pass": results_weighted[-2],
                            "f1_score_list_last_pass": f1_score_list
                        },
                        "baseline_result": {
                            "f1_score": f1_score_bl,
                            "f1_score_list": f1_score_list_bl
                        }
                    }
                    all_results["experiments"].append(experiment_result)

                    # draw_graph.draw_accuracy_vs_pass_for_all_days_accuracy(results_weighted[-2], f1_score_list_bl)
                    # draw_graph.draw_accuracy_vs_day_for_prediction_accuracy(f1_score_list, f1_score_list_bl)


    output_file = "code/data_market/output/hmm_rl_results.json"
    os.makedirs("code/data_market/output", exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=4)

    print(f"All results saved to {output_file}")