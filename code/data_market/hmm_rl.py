import itertools
import numpy as np
from hmmlearn import hmm
import math
from itertools import combinations
import random
from sklearn.preprocessing import normalize
import sys
import os

pass_num = 0

def generate_pi_states_from_ti_states_random_sample(ti_hat, unique_values_on_each_dimension, lambda_value, pi_size_bound_threshold, seed):
    random.seed(seed)

    ti_i_size = 1
    for dim in ti_hat:
        ti_i_size *= len(dim)

    pi_i_size_lower_bound = math.ceil(ti_i_size / lambda_value)

    possible_pi_states = generate_subsets_from_pi(unique_values_on_each_dimension, pi_i_size_lower_bound + pi_size_bound_threshold, pi_i_size_lower_bound)
    random.shuffle(possible_pi_states)

    for state in possible_pi_states:
        if is_subset(ti_hat, state):
            pi_i = state
            break
    return [sorted(pi_i[0]), sorted(pi_i[1]), sorted(pi_i[2]), sorted(pi_i[3]), sorted(pi_i[4])]

# Helper function: find all grids that are within a given PI
def generate_grids_from_pi(published_intent):
    combinations = list(itertools.product(*published_intent))
    ti_list = [list(map(lambda x: [x], combination)) for combination in combinations]
    
    return ti_list

# Helper function to determine whether a TI state is a subset of a PI state
def is_subset(ti_state, pi_state):
    for num_dim in range(len(ti_state)):
        if not set(ti_state[num_dim]) <= (set(pi_state[num_dim])):
            return False
    
    return True

# Helper function to get PI index from state
def get_pi_index(pi_state, pi_state_dict):
    for key in pi_state_dict.keys():
        if pi_state_dict[key] == pi_state:
            return key
    return -1

def check_zero_emission_prob(emission_prob_matrix, immutable_indices):
    for index in immutable_indices:
        emission_prob_matrix[index[0]][index[1]] = 0
    for row_id in range(len(emission_prob_matrix)):
        if np.sum(emission_prob_matrix[row_id]) < 1e-30:
            emission_prob_matrix[row_id] = np.ones(len(emission_prob_matrix[row_id]))
            emission_prob_matrix[row_id] = emission_prob_matrix[row_id] / np.sum(emission_prob_matrix[row_id])
    return emission_prob_matrix

# Helper function to generate all possible subsets given a PI state, with size less than some threshold (max TI size)
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

def ti_prediction_f1_score(actual_ti_sequence, predicted_ti_sequence, need_preprocess, unique_values_on_each_dimension):
    accuracy_list = []

    # preprocess predicted sequence
    if need_preprocess:
        predicted_ti_sequence_processed = []
        for predicted_ti in predicted_ti_sequence:
            predicted_ti_sequence_processed.append(generate_grids_from_pi(predicted_ti))
    else:
        predicted_ti_sequence_processed = predicted_ti_sequence

    # calculate recall
    for time in range(len(actual_ti_sequence)):
        overlap_num = 0
        ti_true_at_time_i = []
        
        for grid in actual_ti_sequence[time]:
            ti_true_at_time_i_grid = []
            for i in range(len(unique_values_on_each_dimension)):
                ti_true_at_time_i_grid.append([unique_values_on_each_dimension[i][grid[i]]])
            ti_true_at_time_i.append(ti_true_at_time_i_grid)

        for grid_state in predicted_ti_sequence_processed[time]:
            # calculate overlap number
            if grid_state in ti_true_at_time_i:
                overlap_num += 1
        
        # calculate F1 score
        recall_at_time_i = overlap_num / len(ti_true_at_time_i)
        precision_at_time_i = overlap_num / len(predicted_ti_sequence_processed[time])
        if recall_at_time_i + precision_at_time_i != 0:
            accuracy_at_time_i = 2 * recall_at_time_i * precision_at_time_i / (recall_at_time_i + precision_at_time_i)
        else: 
            accuracy_at_time_i = 0
        accuracy_list.append(accuracy_at_time_i)
    return accuracy_list, np.mean(accuracy_list)

def run_rl_algorithm(pi_true_sequence_with_feature_names, possible_ti_states, ti_sizes_upper_bound, 
                     lambda_value, total_time, delta, 
                     ti_true_sequence, num_iter, unique_values_on_each_dimension,
                     ti_true_sequence_with_feature_names, pi_size_bound_threshold, seed, window_size, verbose):
    # initialization
    observed_sequence = []
    pi_state_dict = {}
    pi_counter = 0
    ti_state_dict = {}
    ti_counter = 0

    pi_union = unique_values_on_each_dimension
    min_pi_size = math.floor(min(ti_sizes_upper_bound) / lambda_value)
    pi_union_size = 1
    for dim in pi_union:
        pi_union_size *= len(dim)
    
    random.seed(100)
    possible_pi_states = generate_subsets_from_pi(pi_union, pi_union_size, min_pi_size)
    
    if verbose:
        print("Number of possible PI combinations:", len(possible_pi_states))
        print("Number of possible TI states:", len(possible_ti_states))
    
    # building TI index to state dict
    for ti_state in possible_ti_states:
        if ti_state not in ti_state_dict.values():
            ti_state_dict[ti_counter] = ti_state
            ti_counter += 1

    # check if all PI states is a superset of some TI state
    for pi_state in possible_pi_states:
        subset_num = 0
        for ti_state in possible_ti_states:
            if is_subset(ti_state, pi_state):
                subset_num += 1
        if subset_num == 0:
            possible_pi_states.remove(pi_state)
    
    possible_pi_states = possible_pi_states + [pi for pi in pi_true_sequence_with_feature_names if pi not in possible_pi_states]

    # check if all TI states is a subset of some observed PI state
    for ti_state in possible_ti_states:
        subset_num = 0
        for pi_state in pi_true_sequence_with_feature_names:
            if is_subset(ti_state, pi_state):
                subset_num += 1
        if subset_num == 0:
            possible_ti_states.remove(ti_state)
    
    # build PI index to state dict
    for pi in possible_pi_states:
        if pi not in pi_state_dict.values():
            pi_state_dict[pi_counter] = pi
            observed_sequence.append(pi_counter)
            pi_counter += 1

    observed_sequence = []
    for pi_observed_state in pi_true_sequence_with_feature_names:
        observed_sequence.append(get_pi_index(pi_observed_state, pi_state_dict))

    observed_sequence = np.array(observed_sequence)

    ## Initialize for Baum-Welch
    num_hidden_states = len(ti_state_dict)
    num_observed_states = len(pi_state_dict)

    transition_prior = np.ones((num_hidden_states, num_hidden_states))

    immutable_index = []
    emission_prior = np.ones((num_hidden_states, num_observed_states))
    for ti_index, ti_state in ti_state_dict.items():
        for pi_index, pi_state in pi_state_dict.items():
            if is_subset(ti_state, pi_state):
                emission_prior[ti_index][pi_index] = 1
            else:
                immutable_index.append((ti_index, pi_index))
    
    initial_distribution_prior = np.ones(num_hidden_states)

    # start of RL-based algorithm

    transition_prob_matrix_forward = transition_prior.copy()
    transition_prob_matrix_backward = transition_prior.copy()
    emission_prob_matrix = emission_prior.copy()

    if verbose:
        print('-----------')
        print('Pass 0')

    transition_prob_matrix_forward, emission_prob_matrix, post_initial_distribution, model, ti_prediction_for_this_pass = one_pass(
        transition_prob_matrix_forward, emission_prob_matrix, initial_distribution_prior, total_time, delta, lambda_value, 
        observed_sequence, pi_true_sequence_with_feature_names, immutable_index, num_hidden_states, num_observed_states,
                 ti_state_dict, pi_state_dict, ti_true_sequence_with_feature_names,
             unique_values_on_each_dimension, pi_size_bound_threshold, seed, verbose)
    
    f1_score_day_pass = []
    f1_score_day_pass.append(ti_prediction_f1_score(ti_true_sequence, ti_prediction_for_this_pass, True, unique_values_on_each_dimension)[0])

    ti_hat_for_all_passes = [ti_prediction_for_this_pass]

    transition_matrices = [transition_prob_matrix_forward]

    num_passes = 1
    global pass_num
    pass_num += 1

    direction = 'backward'

    for j in range(num_iter-1):
        if verbose:
            print('-----------')
            print('Pass', j+1)

        # run RL-algorithm in other direction
        observed_sequence = observed_sequence[::-1]
        pi_true_sequence_with_feature_names = pi_true_sequence_with_feature_names[::-1]
        ti_sizes_upper_bound = ti_sizes_upper_bound[::-1]
        ti_true_sequence_with_feature_names = ti_true_sequence_with_feature_names[::-1]

        if direction == 'backward':
            transition_prob_matrix_backward, emission_prob_matrix, post_initial_distribution, model, ti_prediction_for_this_pass = one_pass(
            transition_prob_matrix_backward, emission_prob_matrix, post_initial_distribution, total_time, delta, lambda_value,
            observed_sequence, pi_true_sequence_with_feature_names, immutable_index, num_hidden_states, num_observed_states,
                 ti_state_dict, pi_state_dict, ti_true_sequence_with_feature_names, unique_values_on_each_dimension, pi_size_bound_threshold, seed, verbose)
            transition_matrices.append(transition_prob_matrix_backward)
        else:
            transition_prob_matrix_forward, emission_prob_matrix, post_initial_distribution, model, ti_prediction_for_this_pass = one_pass(
            transition_prob_matrix_forward, emission_prob_matrix, post_initial_distribution, total_time, delta, lambda_value,
            observed_sequence, pi_true_sequence_with_feature_names, immutable_index, num_hidden_states, num_observed_states,
                 ti_state_dict, pi_state_dict, ti_true_sequence_with_feature_names, unique_values_on_each_dimension, pi_size_bound_threshold, seed, verbose)
            transition_matrices.append(transition_prob_matrix_forward)

        
        pass_num += 1
        
        ti_hat_for_all_passes.append(ti_prediction_for_this_pass)
        
        if direction == 'backward':
            f1_list = ti_prediction_f1_score(ti_true_sequence, ti_prediction_for_this_pass[::-1], True, unique_values_on_each_dimension)[0]
        else:
            f1_list = ti_prediction_f1_score(ti_true_sequence, ti_prediction_for_this_pass, True, unique_values_on_each_dimension)[0]
        
        f1_score_day_pass.append(f1_list)

        if j > 2 * window_size:
            matrices_within_range = transition_matrices[-window_size:]
            stacked_matrices = np.stack(matrices_within_range)
            average_matrix = np.mean(stacked_matrices, axis=0)
            if direction == 'backward':
                transition_prob_matrix_forward = average_matrix
            else:
                transition_prob_matrix_backward = average_matrix

        # change direction variable
        if direction == 'forward':
            direction = 'backward'
        else:
            direction = 'forward'
        
        num_passes += 1

    if direction == 'backward':
        return transition_prob_matrix_forward, emission_prob_matrix, post_initial_distribution, num_passes, model, ti_hat_for_all_passes, f1_score_day_pass, ti_prediction_for_this_pass
    else:
        return transition_prob_matrix_backward, emission_prob_matrix, post_initial_distribution, num_passes, model, ti_hat_for_all_passes, f1_score_day_pass, ti_prediction_for_this_pass[::-1]
    
# Define a function to suppress stdout and stderr
class SuppressPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

def one_pass(transition_prob_matrix, emission_prob_matrix, initial_distribution, total_time, delta, lambda_value,
             observed_sequence, pi_true_sequence_with_feature_names, immutable_index,
             num_hidden_states, num_observed_states, ti_state_dict, pi_state_dict, ti_true_sequence_with_feature_names,
             unique_values_on_each_dimension, pi_size_bound_threshold, seed, verbose):
    ti_prediction_for_this_pass = []
    
    model = hmm.CategoricalHMM(n_components=num_hidden_states, init_params='', random_state=100, n_features=num_observed_states,
                               algorithm='map')
    
    model.transmat_ = normalize(transition_prob_matrix, axis=1, norm = 'l1')
    model.emissionprob_ = normalize(emission_prob_matrix, axis=1, norm = 'l1')
    model.startprob_ = initial_distribution / np.sum(initial_distribution)

    with SuppressPrint():
        model.fit(observed_sequence.reshape(-1,1))
    post_transition_prob_matrix = model.transmat_
    post_emission_prob_matrix = model.emissionprob_
    post_initial_distribution = model.startprob_
    
    # Laplace smoothing to avoid zero-sum rows
    laplace_factor = 1e-10
    post_transition_prob_matrix += laplace_factor
    post_emission_prob_matrix += laplace_factor
    
    post_emission_prob_matrix = check_zero_emission_prob(post_emission_prob_matrix, immutable_index)

    # Normalize transition and emission matrices
    row_sums = post_transition_prob_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # prevent division by zero
    post_transition_prob_matrix = post_transition_prob_matrix / row_sums
    model.transmat_ = post_transition_prob_matrix

    row_sums = post_emission_prob_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    post_emission_prob_matrix = post_emission_prob_matrix / row_sums
    model.emissionprob_ = post_emission_prob_matrix

    start_sum = np.sum(model.startprob_)
    if np.isnan(start_sum) or start_sum == 0:
        model.startprob_ = np.ones(num_hidden_states) / num_hidden_states

    prev_day_rewarded = False
    first_day = True
    ti_prev_hat_index = 0

    pi_true_index = []
    for state in pi_true_sequence_with_feature_names:
        pi_true_index.append(get_pi_index(state, pi_state_dict))
    
    for i in range(total_time):
        skip_reward = False
        # Predict TI i+1 hat using baum welch results
        ti_i_hat_index = model.predict(observed_sequence.reshape(-1,1))[i]
        ti_i_hat_state = ti_state_dict[ti_i_hat_index]

        # Predict PI i+1 hat using TI i+1 hat using random sampling
        pi_i_hat_state = generate_pi_states_from_ti_states_random_sample(ti_i_hat_state, unique_values_on_each_dimension, lambda_value, pi_size_bound_threshold, seed)
        pi_i_hat_index = get_pi_index(pi_i_hat_state, pi_state_dict)
        
        if pi_i_hat_index == -1:
            skip_reward = True
            
        if pass_num <= 3:
            # RL steps
            if verbose:
                print('************** Day', i, '*******************')
                print('PI F1:', calc_f1(pi_i_hat_state, pi_true_sequence_with_feature_names[i]))
                print('TI F1:', calc_f1(ti_i_hat_state, ti_true_sequence_with_feature_names[i]))

        if not skip_reward:
            post_transition_prob_matrix, post_emission_prob_matrix, prev_day_rewarded = accuracy_reward(pi_i_hat_state, pi_true_sequence_with_feature_names[i], 
                                                                                                        ti_i_hat_index, ti_prev_hat_index, delta,
                                                                                                        post_transition_prob_matrix, post_emission_prob_matrix, 
                                                                                                        pi_state_dict, prev_day_rewarded, first_day, verbose)
            
            
        # prepare for next iteration
        
        model.transmat_ = normalize(post_transition_prob_matrix, axis = 1, norm = 'l1')
       
        post_emission_prob_matrix = check_zero_emission_prob(post_emission_prob_matrix, immutable_index)
        model.emissionprob_ = normalize(post_emission_prob_matrix, axis = 1, norm = 'l1')
        model.startprob_ = [0 if np.isnan(prob) else prob for prob in model.startprob_]
        if np.sum(model.startprob_) < 1e-10:
            model.startprob_ = [1/len(model.startprob_) for _ in range(len(model.startprob_))]
        if max(model.startprob_) == 1.0:
            model.startprob_ = [prob + laplace_factor for prob in model.startprob_]
        model.startprob_ = model.startprob_ / np.clip(np.sum(model.startprob_), 1e-10, None)

        # update ti_prev
        ti_prev_hat_index = ti_i_hat_index
        first_day = False

    
    post_initial_distribution = np.zeros(num_hidden_states)
    post_initial_distribution[model.predict(observed_sequence.reshape(-1,1))[-1]] = 1

    post_transition_prob_matrix = normalize(post_transition_prob_matrix, axis = 1, norm = 'l1')
    post_transition_prob_matrix = post_transition_prob_matrix * (len(post_transition_prob_matrix[0]) * 10)
    post_emission_prob_matrix = check_zero_emission_prob(post_emission_prob_matrix, immutable_index)
    post_emission_prob_matrix = normalize(post_emission_prob_matrix, axis = 1, norm = 'l1')
    post_emission_prob_matrix = post_emission_prob_matrix * (len(post_emission_prob_matrix) * 10)
    
    ti_prediction_for_this_pass_index = model.predict(observed_sequence.reshape(-1,1))
    ti_prediction_for_this_pass = []
    for index in ti_prediction_for_this_pass_index:
        ti_prediction_for_this_pass.append(ti_state_dict[index])

    return post_transition_prob_matrix, post_emission_prob_matrix, np.array(post_initial_distribution), model, ti_prediction_for_this_pass

def assess_f1(pi_2_hat_state, pi_2_true_state, delta):
    overlap_num = 0
    pi_2_hat_grids = generate_grids_from_pi(pi_2_hat_state)
    pi_2_true_grids = generate_grids_from_pi(pi_2_true_state)
    pi_2_hat_intersect_pi_2_true = [grid for grid in pi_2_true_grids if grid in pi_2_hat_grids]
    overlap_num = len(pi_2_hat_intersect_pi_2_true)
    
    # calculate recall and precision
    recall_at_time_i = overlap_num / len(pi_2_true_grids)
    precision_at_time_i = overlap_num / len(pi_2_hat_grids)
    if recall_at_time_i + precision_at_time_i != 0:
        return (2 * recall_at_time_i * precision_at_time_i / (recall_at_time_i + precision_at_time_i)) >= delta
    else: 
        return 0 >= delta

def calc_f1(pi_2_hat_state, pi_2_true_state):
    overlap_num = 0
    pi_2_hat_grids = generate_grids_from_pi(pi_2_hat_state)
    pi_2_true_grids = generate_grids_from_pi(pi_2_true_state)
    pi_2_hat_intersect_pi_2_true = [grid for grid in pi_2_true_grids if grid in pi_2_hat_grids]
    overlap_num = len(pi_2_hat_intersect_pi_2_true)
    
    # calculate recall and precision
    recall_at_time_i = overlap_num / len(pi_2_true_grids)
    precision_at_time_i = overlap_num / len(pi_2_hat_grids)
    if recall_at_time_i + precision_at_time_i != 0:
        return 2 * recall_at_time_i * precision_at_time_i / (recall_at_time_i + precision_at_time_i)
    else: 
        return 0

def accuracy_reward(pi_2_hat_state, pi_2_true_state, ti_index, ti_prev_hat_index, delta, transition_prob_matrix, emission_prob_matrix, 
                    pi_state_dict, prev_day_rewarded, first_day, verbose):
    # if accuracy constraint satisfied, reward
    reward_factor = 10000
    if assess_f1(pi_2_hat_state, pi_2_true_state, delta):
        if prev_day_rewarded and not first_day:
            transition_prob_matrix[ti_prev_hat_index][ti_index] += reward_factor
            
            ## emission reward
            pi_index = get_pi_index(pi_2_true_state, pi_state_dict)
            emission_prob_matrix[ti_index][pi_index] += reward_factor
        
        if verbose:
            print('Accuracy reward for transition and emission matrices')
        prev_day_rewarded = True

    # penalize otherwise
    else:
        # transition
        if prev_day_rewarded and not first_day:
            transition_prob_matrix[ti_prev_hat_index][ti_index] = 0
        
            # emission
            pi_index = get_pi_index(pi_2_true_state, pi_state_dict)
            emission_prob_matrix[ti_index][pi_index] = 0
       
        if verbose:
            print('Accuracy penalize for transition and emission matrices')
        prev_day_rewarded = False
    
    # normalize and return
    transition_prob_matrix = normalize(transition_prob_matrix, axis = 1, norm = 'l1')
    emission_prob_matrix = normalize(emission_prob_matrix, axis = 1, norm = 'l1')

    return transition_prob_matrix, emission_prob_matrix, prev_day_rewarded