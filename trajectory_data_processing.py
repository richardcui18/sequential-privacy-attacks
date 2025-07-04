import pandas as pd
import numpy as np
import os
import itertools
import csv
import hmm_rl

def read_one_trajectory(person_id, trajectory_id, sample_frequency_minutes, max_frequency_minutes = 1):
    # Read geolife dataset
    file_path='data/geolife/'+str(person_id)+'/Trajectory/'+str(trajectory_id)+'.plt'
    
    df = pd.read_csv(file_path, skiprows=6, delimiter=',')
    df.columns=['Latitude', 'Longitude', '0', 'Altitude', 'Time past 12/40/1899', 'Date', 'Time']
    df.drop(columns=['0', 'Altitude'], inplace=True)

    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    filtered_rows = []

    last_time = df['Datetime'].iloc[0]
    filtered_rows.append(df.iloc[0])

    for index in range(1, len(df)):
        current_time = df['Datetime'].iloc[index]
        
        if pd.Timedelta(minutes=max_frequency_minutes) >= (current_time - last_time) and (current_time - last_time) >= pd.Timedelta(minutes=sample_frequency_minutes):
            filtered_rows.append(df.iloc[index])
            last_time = current_time

    filtered_df = pd.DataFrame(filtered_rows)
    return filtered_df

def get_trajectories_in_geographic_range(sample_frequency_minutes, lon_range, lat_range):
    results = []

    for person_id in range(182):
        print(f"Processing person {person_id} of 181...")
        person_id_str = f"{person_id:03d}"
        trajectory_files = list_all_trajectory_files(person_id_str)

        for file_name in trajectory_files:
            trajectory = read_one_trajectory(person_id_str, file_name[:-4], sample_frequency_minutes)
            if len(trajectory) <= 5 or len(trajectory) >= 30:
                continue
            bounding_box_lon, bounding_box_lat = calculate_one_bounding_box(trajectory)
            if bounding_box_lon[0] >= lon_range[0] and bounding_box_lon[1] <= lon_range[1] and bounding_box_lat[0] >= lat_range[0] and bounding_box_lat[1] <= lat_range[1]:
                results.append((person_id_str, file_name[:-4]))

    return results

def list_all_trajectory_files(person_id):
    folder_path = f"data/geolife/{person_id}/Trajectory/"
    return [f for f in os.listdir(folder_path) if f.endswith('.plt')]

def save_to_csv(data, file_name, column_names, folder_name='data/geolife/processed'):
    os.makedirs(folder_name, exist_ok=True)
    
    file_path = os.path.join(folder_name, file_name)

    print(f"Process trajectory data saved to {folder_name}/{file_name}.")
    
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(column_names)
        writer.writerows(data)

def get_processed_data_geographic_range(num_trajectories):
    processed_data = pd.read_csv('data/geolife/processed/processed_geographic_range')
    processed_data['Person ID'] = processed_data.index.map(lambda x: f"{x:03d}")
    processed_data['Trajectory IDs'] = processed_data['Trajectory IDs'].map(str)
    processed_data['ID Pairs'] = list(zip(processed_data['Person ID'], processed_data['Trajectory IDs']))
    return processed_data.sample(n=num_trajectories, random_state=1)

def calculate_one_bounding_box(trajectory_df):
    longitudes = trajectory_df['Longitude'].tolist()
    latitudes=trajectory_df['Latitude'].tolist()
    bounding_box_lon = [min(longitudes)-0.01, max(longitudes)+0.01]
    bounding_box_lat = [min(latitudes)-0.01, max(latitudes)+0.01]
    return [bounding_box_lon, bounding_box_lat]

def calculate_max_bounding_box(trajectory_ids_pairs, sample_frequency_minutes):
    max_bounding_box_lon_range = [float('inf'),float('-inf')]
    max_bounding_box_lat_range = [float('inf'),float('-inf')]
    for trajectory_ids_pair in trajectory_ids_pairs:
        person_id = trajectory_ids_pair[0]
        trajectory_id = trajectory_ids_pair[1]
        trajectory_df = read_one_trajectory(person_id, trajectory_id, sample_frequency_minutes)
        bounding_box_lon_range, bounding_box_lat_range = calculate_one_bounding_box(trajectory_df)
        if max_bounding_box_lon_range[0] > bounding_box_lon_range[0]:
            max_bounding_box_lon_range[0] = bounding_box_lon_range[0]
        if max_bounding_box_lon_range[1] < bounding_box_lon_range[1]:
            max_bounding_box_lon_range[1] = bounding_box_lon_range[1]
        if max_bounding_box_lat_range[0] > bounding_box_lat_range[0]:
            max_bounding_box_lat_range[0] = bounding_box_lat_range[0]
        if max_bounding_box_lat_range[1] < bounding_box_lat_range[1]:
            max_bounding_box_lat_range[1] = bounding_box_lat_range[1]
    return max_bounding_box_lon_range, max_bounding_box_lat_range

def lat_lon_to_area(lat_range, lon_range, reference_latitude=39.990):
    height = lat_range * 111139 

    lon_m_per_degree = 111319 * np.cos(np.radians(reference_latitude))
    width = lon_range * lon_m_per_degree  

    area = height * width
    return area

def calculate_cut_lengths(id_pairs, sample_frequency_minutes, trajectory_grid_resolution_lon_limit, trajectory_grid_resolution_lat_limit):
    longtiude_cut_length = trajectory_grid_resolution_lon_limit
    latitude_cut_length = trajectory_grid_resolution_lat_limit

    longtiude_cut_length=0.1
    latitude_cut_length=0.1
    max_in_same_grid_lat = 1
    max_in_same_grid_lon=1

    for idx, (person_id, trajectory_id) in enumerate(id_pairs):
        trajectory_df=read_one_trajectory(str(person_id).zfill(3), trajectory_id, sample_frequency_minutes)

        longitudes = trajectory_df['Longitude'].tolist()
        pairwise_diffs_lon = [abs(a - b) for a, b in itertools.combinations(longitudes, 2)]
        pairwise_diffs_sorted_lon = sorted(pairwise_diffs_lon)
        min_diff_lon = pairwise_diffs_sorted_lon[0]
        if min_diff_lon < longtiude_cut_length:
            i=0
            while pairwise_diffs_sorted_lon[i] <= trajectory_grid_resolution_lon_limit:
                if i == len(pairwise_diffs_lon)-1:
                    break
                i+=1
            longtiude_cut_length=pairwise_diffs_sorted_lon[i]
            if i> max_in_same_grid_lon:
                max_in_same_grid_lon=i

        latitudes = trajectory_df['Latitude'].tolist()
        pairwise_diffs_lat = [abs(a - b) for a, b in itertools.combinations(latitudes, 2)]
        pairwise_diffs_sorted_lat = sorted(pairwise_diffs_lat)
        min_diff_lat = pairwise_diffs_sorted_lat[0]
        if min_diff_lat < latitude_cut_length:
            i=0
            while pairwise_diffs_sorted_lat[i] <= trajectory_grid_resolution_lat_limit:
                if i == len(pairwise_diffs_lat)-1:
                    break
                i+=1
            latitude_cut_length=pairwise_diffs_sorted_lat[i]
            if i> max_in_same_grid_lat:
                max_in_same_grid_lat=i
        
    print('Area of one grid (m^2):', lat_lon_to_area(latitude_cut_length, longtiude_cut_length))
    
    return longtiude_cut_length, latitude_cut_length


def create_tl_sequence(sample_frequency_minutes, trajectory_grid_resolution_lon_limit, trajectory_grid_resolution_lat_limit, num_trajectories):
    processed_df = get_processed_data_geographic_range(num_trajectories)
    id_pairs = list(processed_df['ID Pairs'])
    bounding_box = calculate_max_bounding_box(id_pairs, sample_frequency_minutes)
    bounding_box_lon, bounding_box_lat = bounding_box

    theoretical_max_without_constraint = 0
    
    longitude_cut_length, latitude_cut_length = calculate_cut_lengths(id_pairs, sample_frequency_minutes, trajectory_grid_resolution_lon_limit, trajectory_grid_resolution_lat_limit)
    
    lon_start, lon_end = bounding_box_lon
    lat_start, lat_end = bounding_box_lat
    num_lon_cuts = int(np.ceil((lon_end - lon_start) / longitude_cut_length))
    num_lat_cuts = int(np.ceil((lat_end - lat_start) / latitude_cut_length))

    cut_num_to_lon = np.linspace(lon_start, lon_end, num_lon_cuts + 1)
    cut_num_to_lat = np.linspace(lat_start, lat_end, num_lat_cuts + 1)

    # initialize unique_values_on_each_dimension
    unique_values_on_each_dimension = [
        [str(i) for i in range(num_lon_cuts)],  
        [str(i) for i in range(num_lat_cuts)] 
    ]
    
    # initialize data_cube and cost_cube
    data_cube = np.ones((num_lon_cuts, num_lat_cuts))
    cost_cube = data_cube.copy()

    # initialize tl_true_sequences and tl_true_sequences_with_feature_names
    tl_true_sequences = []
    tl_true_sequences_with_feature_names = []

    for person_id, trajectory_id in id_pairs:
        trajectory_df = read_one_trajectory(str(person_id).zfill(3), trajectory_id, sample_frequency_minutes)

        lon_sequence = trajectory_df['Longitude'].tolist()
        lat_sequence = trajectory_df['Latitude'].tolist()

        tl_sequence = [] 
        tl_sequence_with_names = []

        for lon, lat in zip(lon_sequence, lat_sequence):
            lon_index = find_grid_index(lon, lon_start, longitude_cut_length, num_lon_cuts)
            lat_index = find_grid_index(lat, lat_start, latitude_cut_length, num_lat_cuts)

            tl_sequence.append([[lon_index, lat_index]])

            tl_sequence_with_names.append([[str(lon_index)], [str(lat_index)]])

            cur_theoretical_max = max(hmm_rl.calc_euclidean_distance([[lon_index], [lat_index]], [['0'], ['0']], cut_num_to_lon, cut_num_to_lat),
            hmm_rl.calc_euclidean_distance([[lon_index], [lat_index]], [[str(len(cut_num_to_lon)-1)], ['0']], cut_num_to_lon, cut_num_to_lat),
            hmm_rl.calc_euclidean_distance([[lon_index], [lat_index]], [['0'], [str(len(cut_num_to_lat)-1)]], cut_num_to_lon, cut_num_to_lat),
            hmm_rl.calc_euclidean_distance([[lon_index], [lat_index]], [[str(len(cut_num_to_lon)-1)], [str(len(cut_num_to_lat)-1)]], cut_num_to_lon, cut_num_to_lat))

            if cur_theoretical_max > theoretical_max_without_constraint:
                theoretical_max_without_constraint = cur_theoretical_max

        tl_true_sequences.append(tl_sequence)
        tl_true_sequences_with_feature_names.append(tl_sequence_with_names)

    return unique_values_on_each_dimension, data_cube, cost_cube, tl_true_sequences, tl_true_sequences_with_feature_names, cut_num_to_lon, cut_num_to_lat, theoretical_max_without_constraint

# Helper function to find grid index based on longitude/latitude value
def find_grid_index(value, start, cut_length, num_cuts):
    index = int((value - start) / cut_length)
    return min(max(0, index), num_cuts - 1)


if __name__ == '__main__':
    sample_frequency_minutes = 0.3

    results = get_trajectories_in_geographic_range(sample_frequency_minutes, [116.28, 116.33], [39.95, 40.0])
    save_to_csv(results, 'processed_geographic_range', ['Trajectory IDs'])