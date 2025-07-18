import calculate_confidence
from itertools import combinations
import hmm_rl

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

## Baseline: Uniform attack
def pi_uniform_attack(total_time, pi_true_sequence_with_feature_names, ti_sizes_upper_bound, lambda_value, ti_true_sequence, 
                      unique_values_on_each_dimension):
    predicted_ti_from_em = []

    for i in range(total_time):
        possible_ti_list_time_i = generate_subsets_from_pi(pi_true_sequence_with_feature_names[i], ti_sizes_upper_bound[i], 1)

        confidence_list = []
        for true_intent in possible_ti_list_time_i:
            confidence = calculate_confidence.confidence_upper_bound(true_intent, pi_true_sequence_with_feature_names[i])
            if confidence <= lambda_value:
                confidence_list.append(confidence)
            else:
                confidence_list.append(0)

        ti_i_hat_em_attack = []

        combined_list = list(zip(confidence_list, possible_ti_list_time_i))
        sorted_combined_list = sorted(combined_list, key=lambda x: x[0], reverse=True)
        ti_i_hat_em_attack = sorted_combined_list[0][1]
        
        predicted_ti_from_em.append(ti_i_hat_em_attack)
    
    f1_score_list_bl, f1_score_bl = hmm_rl.ti_prediction_f1_score(ti_true_sequence, predicted_ti_from_em, True, unique_values_on_each_dimension)
    
    return f1_score_list_bl, f1_score_bl