import random
import numpy as np
import trajectory_data_processing
import uniform_attack

def get_dataset(dataset_name, deviation_amount_user, lambda_value):
    if dataset_name == 'geolife':
        return get_geolife_dataset(deviation_amount_user, lambda_value)
    else:
        raise ValueError(f"Dataset {dataset_name} not found")

def get_geolife_dataset(deviation_amount_user, lambda_value):
    # Setup
    sample_frequency_minutes = 0.3
    trajectory_grid_resolution_lon_limit = 0.001
    trajectory_grid_resolution_lat_limit = 0.001
    seed = 65
    num_trajectories = 10

    dataset = {
        'unique_values_on_each_dimension': [],
        'tl_true_sequences': [],
        'tl_true_sequences_with_feature_names': [],
        'pr_true_sequences_with_feature_names': [],
        'total_times': [],
        'cut_num_to_lon': [],
        'cut_num_to_lat': []
    }

    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Create true TL sequence
    unique_values_on_each_dimension, _, _, tl_true_sequences, tl_true_sequences_with_feature_names, cut_num_to_lon, cut_num_to_lat, _ = trajectory_data_processing.create_tl_sequence(sample_frequency_minutes, trajectory_grid_resolution_lon_limit, trajectory_grid_resolution_lat_limit, num_trajectories)

    # Create true PR sequence
    pr_true_sequences_with_feature_names = []
    total_times = []

    for i in range(len(tl_true_sequences)):
        tl_true_sequence_with_feature_names = tl_true_sequences_with_feature_names[i]

        pr_true_sequence_with_feature_names = []

        for i, tl_i in enumerate(tl_true_sequence_with_feature_names):
            pr_i = uniform_attack.expansion_w_tl_at_center_with_deviation(unique_values_on_each_dimension, true_location = tl_i, 
                        attack_type= 'PR_uniform_attack', lambda_value = lambda_value, random_seed=seed+i, deviation_amount=deviation_amount_user)
            pr_true_sequence_with_feature_names.append([sorted(pr_i[0], key=int), sorted(pr_i[1], key=int)])
        
        pr_true_sequences_with_feature_names.append(pr_true_sequence_with_feature_names)
    
        total_times.append(len(tl_true_sequence_with_feature_names))

    dataset['unique_values_on_each_dimension'] = unique_values_on_each_dimension
    dataset['pr_true_sequences_with_feature_names'] = pr_true_sequences_with_feature_names
    dataset['total_times'] = total_times
    dataset['tl_true_sequences'] = tl_true_sequences
    dataset['tl_true_sequences_with_feature_names'] = tl_true_sequences_with_feature_names
    dataset['cut_num_to_lon'] = cut_num_to_lon
    dataset['cut_num_to_lat'] = cut_num_to_lat

    return dataset