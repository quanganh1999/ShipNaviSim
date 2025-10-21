import polars as pl
from ship_env import ShipEnvironment, read_position_data, chunk_to_traj, create_minari_dataset
from datetime import timedelta
import utils
import pandas as pd
import numpy as np
import gymnasium as gym
import minari

def replay_ship_env(id_ship, env, render = False):        
    obs, info = env.reset(seed=42, options = {'ego_pos': id_ship})
    actions = info['actions']
    if(actions is None):
        return None
    for i in range(1000):        
        action = actions[i]
        observation, _, terminated, truncated, info = env.step(action)
        # print(i, observation)
        # print()
        if terminated or truncated:          
            break    
    if render:
        env.render()    
    return info # to do stats

def create_vessel_trajectory_df(id_ship, trajs, overlap_idx, vessel_ids):
    """
    Create a dataframe with main vessel and other vessels' trajectories,
    only including timestamps from other vessels that match the main vessel.
    
    Parameters:
    id_ship (int): Index of the main vessel in trajs list
    trajs (list): List of trajectory arrays, each with columns [time, x, y, speed, heading, course]
    overlap_idx (list): List of lists containing indices of overlapping vessels for each vessel
    vessel_ids (list): List of tuples containing (ship_id, mmsi, imo) for each vessel
    
    Returns:
    pd.DataFrame: DataFrame with all vessel trajectories
    """
    # Get the main vessel trajectory
    main_traj = trajs[id_ship]
    
    # Get the main vessel identification
    main_ship_id = vessel_ids[id_ship]
    
    # Create dataframe for main vessel
    main_df = pd.DataFrame({
        'type': 'ego',
        'ship_id': main_ship_id,
        'timestamp': main_traj[:, 0],
        'lon': main_traj[:, 1],
        'lat': main_traj[:, 2],
        'speed': main_traj[:, 3],
        'heading': main_traj[:, 4],
        'course': main_traj[:, 5]
    })
    
    # Create list to hold all dataframes
    all_dfs = [main_df]
    
    # Get the set of timestamps from the main vessel
    main_timestamps = set(main_traj[:, 0])
    
    # Get the list of overlapping vessel indices
    overlapping_vessels = overlap_idx[id_ship] if id_ship < len(overlap_idx) else []
    
    # Process each overlapping vessel
    for other_id in overlapping_vessels:
        # Get other vessel trajectory
        other_traj = trajs[other_id]
        
        # Get other vessel identification
        other_ship_id = vessel_ids[other_id]
        
        # Filter the trajectory to only include rows with timestamps in main_timestamps
        mask = np.isin(other_traj[:, 0], list(main_timestamps))
        filtered_traj = other_traj[mask]
        
        if len(filtered_traj) > 0:
            other_df = pd.DataFrame({
                'type': 'other',
                'ship_id': other_ship_id,
                'timestamp': filtered_traj[:, 0],
                'lon': filtered_traj[:, 1],
                'lat': filtered_traj[:, 2],
                'speed': filtered_traj[:, 3],
                'heading': filtered_traj[:, 4],
                'course': filtered_traj[:, 5]
            })
            all_dfs.append(other_df)
    
    # Concatenate all dataframes
    if len(all_dfs) > 1:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df = combined_df.sort_values(['type', 'timestamp'])
    else:
        combined_df = main_df
    
    return combined_df

if __name__ == '__main__':
    plan_start_time = 0
    plan_end_time = 24  
    region_of_interest = {"LON": (103.82, 103.88), "LAT": (1.15, 1.22)}
    region_of_interest_array = np.array([list(region_of_interest[k]) for k in ["LON", "LAT"]]).T
    print(region_of_interest_array)
    from split_chunks import seperate_chunks
    # Read static data from multiple files
    data_folder = "./raw_data/"
    # Read position data from multiple files
    position_file_pattern = data_folder+"synthetic_ais_data.csv"
    df = read_position_data(position_file_pattern)          
    geo_filters = [(v[0] <= pl.col(k)) & (pl.col(k) <= v[1]) for k, v in region_of_interest.items()]     
    # Create time filter for planning horizon (7:00 AM to 9:00 AM)    
    if plan_end_time > plan_start_time:
        time_filter = (pl.col("TIMESTAMP_UTC").dt.hour() >= plan_start_time) & (pl.col("TIMESTAMP_UTC").dt.hour() < plan_end_time)
    else:
        time_filter = (pl.col("TIMESTAMP_UTC").dt.hour() >= plan_start_time) | (pl.col("TIMESTAMP_UTC").dt.hour() < plan_end_time)
    # Create heading filter for NULL and 511 values
    heading_filter = (pl.col("HEADING").is_not_null()) & (pl.col("HEADING") != 511)
    # Create status filter for removing non-zero status value
    status_filter = (pl.col("STATUS") == 0)
    # Combine all filters
    all_filters = geo_filters + [time_filter, heading_filter, status_filter]
    df = df.filter(*all_filters)           
    print(df) 
    threshold = timedelta(minutes=30)
    inter_pol = timedelta(seconds=10)
    num_samples = 5
    chunks = seperate_chunks(df, threshold, num_samples)    
    print(chunks.describe())    
    chunks = chunks.sort('start_time')
    print(chunks[:10])
    num_ships = len(chunks)       
    trajs = []    
    times = []
    tw_lst = []
    vessel_ids = []
    for j in range(num_ships):      
        traj, time = chunk_to_traj(chunks[j], inter_pol, region_of_interest_array)             
        vessel_ids.append(chunks[j]["SHIP_ID"][0])
        trajs.append(traj)        
        times.append(time)
        tw_lst.append((times[-1][0], times[-1][-1]))        
    overlap_idx = utils.find_overlapping_intervals(tw_lst)
    
    #Create environment
    max_size = max([len(val) for val in overlap_idx])   
    env = ShipEnvironment(trajs, times, overlap_idx, region_of_interest, n_neighbor_agents=10) 
    
    #Render 
    env = gym.wrappers.RecordVideo(env, "tmp/video-1.mp4")
    env.metadata['render_fps'] = 10    
    replay_ship_env(id_ship=0, env=env, render=True)    
    
    #Create minari dataset for IL
    # Uncomment for creating offline dataset. Note that, we only need to create dataset once.
    create_minari_dataset(env, dataset_name="Maritime-Expert-v1",num_ships=num_ships)
    
    #Load dataset 
    dataset = minari.load_dataset("Maritime-Expert-v1")

    for episode_data in dataset.iterate_episodes():
        observations = episode_data.observations
        actions = episode_data.actions
        rewards = episode_data.rewards
        terminations = episode_data.terminations
        truncations = episode_data.truncations
        infos = episode_data.infos
        
    #Compute metrics for all expert trajs:
    lst_infos = []
    for id_ship in range(num_ships):
        print(id_ship)
        val = replay_ship_env(id_ship, env)
        if(val is None):
            continue
        val["ship_id"] = chunks[id_ship]["SHIP_ID"][0]
        val["start_time"] = chunks[id_ship]["start_time"][0]
        lst_infos.append(val)
    df_exp_infos = pd.DataFrame(lst_infos)
    df_exp_infos.drop(["CPD"], axis=1, inplace=True)
    print(df_exp_infos)
    df_exp_infos.to_csv(f"exp_stats_{plan_start_time}to{plan_end_time}.csv", sep=";", index=False) 
    
