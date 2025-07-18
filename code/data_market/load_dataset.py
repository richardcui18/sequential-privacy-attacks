import numpy as np
import random
from itertools import combinations
import math

# Helper function to determine whether a TI state is a subset of a PI state
def is_subset(ti_state, pi_state):
    for num_dim in range(len(ti_state)):
        if not set(ti_state[num_dim]) <= (set(pi_state[num_dim])):
            return False
    return True

# Helper function to generate all possible PI states, with size between some interval
def generate_possible_pi_states_given_interval(unique_values_on_each_dimension, max_size, min_size):
    def get_subsets(dim, current_subset, current_size):
        if dim == len(unique_values_on_each_dimension):
            if min_size <= current_size <= max_size:
                result.append(current_subset)
            return
        
        for size in range(1, len(unique_values_on_each_dimension[dim]) + 1):
            for combination in combinations(unique_values_on_each_dimension[dim], size):
                new_size = current_size * size
                if new_size <= max_size:
                    get_subsets(dim + 1, current_subset + [list(combination)], new_size)
    
    result = []
    get_subsets(0, [], 1)
    return result

def df2data_cube_five_attributes(df):
    # get the column names of the dataframe
    column_names = df.columns.values.tolist()
    dimension_1 = column_names[0]
    dimension_2 = column_names[1]
    dimension_3 = column_names[2]
    dimension_4 = column_names[3]
    dimension_5 = column_names[4]
    # get the unique values of each column
    dimension_1_unique = df[dimension_1].unique().tolist()
    dimension_2_unique = df[dimension_2].unique().tolist()
    dimension_3_unique = df[dimension_3].unique().tolist()
    dimension_4_unique = df[dimension_4].unique().tolist()
    dimension_5_unique = df[dimension_5].unique().tolist()
    # get the number of unique values of each column
    dimension_1_size = len(dimension_1_unique)
    dimension_2_size = len(dimension_2_unique)
    dimension_3_size = len(dimension_3_unique)
    dimension_4_size = len(dimension_4_unique)
    dimension_5_size = len(dimension_5_unique)
    # create a data cube
    data_cube = np.zeros((dimension_1_size, dimension_2_size, dimension_3_size, dimension_4_size, dimension_5_size),
                         dtype=int)
    # store the data into the data cube
    for i in range(len(df)):
        data_cube[dimension_1_unique.index(df[dimension_1][i])][dimension_2_unique.index(df[dimension_2][i])][
            dimension_3_unique.index(df[dimension_3][i])][dimension_4_unique.index(df[dimension_4][i])][
            dimension_5_unique.index(df[dimension_5][i])] += 1
    unique_values_on_each_dimension = [sorted(dimension_1_unique), sorted(dimension_2_unique), sorted(dimension_3_unique), sorted(dimension_4_unique),
                                       sorted(dimension_5_unique)]
    return data_cube, unique_values_on_each_dimension


def generate_unit_cost_data_cube_same_price(data_cube):
    # generate a np array with the same shape as the data cube and all elements are 1
    unit_cost_data_cube = np.ones(data_cube.shape)
    return unit_cost_data_cube

def create_true_intent_random_walk(unique_values_on_each_dimension, total_time, random_seed):
    random.seed(random_seed)

    def choose_direction_and_length(start, length, dimension_length):
        direction = random.choice([-1, 1])
        
        new_start = start + direction

        if new_start < 0:
            new_start = 0
        if new_start >= dimension_length:
            new_start = dimension_length - 1

        possible_changes = [-1, 0, 1]

        if length == 1:
            possible_changes.remove(-1)
        
        if length == dimension_length or new_start + length >= dimension_length:
            possible_changes.remove(1)

        length_change = random.choice(possible_changes)
        new_length = length + length_change

        return new_start, new_length

    # Initialize the random walk for each dimension
    current_positions = []
    # dim_num = 0
    for dimension in unique_values_on_each_dimension:
        start_index = random.randint(0, len(dimension) - 1)
        max_length = len(dimension) - start_index
        subset_length = random.randint(1,max_length)
        
        current_positions.append((start_index, subset_length))
    

    walk_sequence = []

    for _ in range(total_time):
        step = []
        for dim_index, dimension in enumerate(unique_values_on_each_dimension):
            start, length = current_positions[dim_index]
            new_start, new_length = choose_direction_and_length(start, length, len(dimension))
            current_positions[dim_index] = (new_start, new_length)
            subset = dimension[new_start:new_start + new_length]
            step.append(subset)
        walk_sequence.append(step)
    
    return walk_sequence

def generate_pi_states_from_ti_states_random_sample(ti_true_sequence_with_feature_names, ti_true_sizes, unique_values_on_each_dimension, lambda_value, seed):
    random.seed(seed)
    
    pi_true_sequence_with_feature_names = []

    total_size = 1
    for dim in unique_values_on_each_dimension:
        total_size *= len(dim)

    for i in range(len(ti_true_sequence_with_feature_names)):
        ti_i = ti_true_sequence_with_feature_names[i]
        ti_i_size = ti_true_sizes[i]
        pi_i_size_lower_bound = math.ceil(ti_i_size / lambda_value)

        # possible_pi_states = generate_possible_pi_states_given_interval(unique_values_on_each_dimension, total_size, pi_i_size_lower_bound)
        possible_pi_states = generate_possible_pi_states_given_interval(unique_values_on_each_dimension, pi_i_size_lower_bound+5, pi_i_size_lower_bound)
        random.shuffle(possible_pi_states)

        for state in possible_pi_states:
            if is_subset(ti_i, state):
                pi_i = state
                break
        pi_true_sequence_with_feature_names.append([sorted(pi_i[0]), sorted(pi_i[1]), sorted(pi_i[2]), sorted(pi_i[3]), sorted(pi_i[4])])
    
    return pi_true_sequence_with_feature_names