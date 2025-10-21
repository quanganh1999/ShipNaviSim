import numpy as np
from datetime import datetime, timezone
import pandas as pd
import os

def create_directory(dir_name):
    try:
        # Check if directory doesn't exist
        if not os.path.exists(dir_name):
            # Create directory
            os.makedirs(dir_name)
            print(f"Directory '{dir_name}' created successfully")
        else:
            print(f"Directory '{dir_name}' already exists")
    except Exception as e:
        print(f"Error creating directory: {e}")


def haversine_distance(lon1, lat1, lon2, lat2):
    #Reference: https://www.geeksforgeeks.org/haversine-formula-to-find-distance-between-two-points-on-a-sphere/
    """Calculate the great circle distance between two points on the earth"""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r * 1000  # Convert to meters

def find_indices_in_range_sorted(arr, low, high):
    arr = np.array(arr)
    start = np.searchsorted(arr, low, side='left')
    end = np.searchsorted(arr, high, side='right')
    return np.arange(start, end)

def compute_heading(lon1, lat1, lon2, lat2):
    """
    Compute the heading (initial bearing) between two points on the earth's surface.
    
    :param lon1: Longitude of the first point (in degrees)
    :param lat1: Latitude of the first point (in degrees)
    :param lon2: Longitude of the second point (in degrees)
    :param lat2: Latitude of the second point (in degrees)
    :return: Heading in degrees (0-360)
    """
    # Convert latitude and longitude to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Compute differences
    dlon = lon2 - lon1
    
    # Calculate heading
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    heading = np.arctan2(y, x)
    
    # Convert to degrees and normalize to 0-360
    heading_deg = np.degrees(heading)
    heading_normalized = (heading_deg + 360) % 360
    
    return heading_normalized

def normalize_angle(angle):
    """Normalize angle to [0, 2π)"""
    return angle % (2 * np.pi)

def shortest_angular_distance(start, end):
    """Calculate the shortest angular distance between two angles"""
    diff = normalize_angle(end - start)
    if diff > np.pi:
        diff -= 2 * np.pi
    return diff

def shortest_path_angle_interp(x, xp, fp):
    """
    Perform shortest path angle interpolation
    
    Parameters:
    x : array_like
        The x-coordinates at which to evaluate the interpolated values.
    xp : 1-D sequence of floats
        The x-coordinates of the data points.
    fp : 1-D sequence of floats
        The y-coordinates of the data points (angles in radians), same length as xp.
    
    Returns:
    array_like
        The interpolated values, same shape as x.
    """
    # x = np.asarray(x)
    # xp = np.asarray(xp)
    # fp = np.asarray(fp)
    
    # Normalize all angles to [0, 2π)
    fp_normalized = normalize_angle(fp)
    
    # Calculate cumulative angular distances
    angular_distances = np.array([shortest_angular_distance(fp_normalized[i], fp_normalized[i+1]) 
                                  for i in range(len(fp_normalized)-1)])
    cumulative_distances = np.concatenate(([0], np.cumsum(angular_distances)))
    
    # Perform linear interpolation on cumulative distances
    interpolated_distances = np.interp(x, xp, cumulative_distances)
    
    # Convert back to angles
    return normalize_angle(fp_normalized[0] + interpolated_distances)

def calculate_trajectory_drift(traj_x, traj_y, traj_heading):        
    # Calculate differences in x and y
    dx = np.diff(traj_x)
    dy = np.diff(traj_y)
    
    # Calculate course over ground (COG)
    cog = np.zeros_like(traj_heading)
    cog[1:] = np.arctan2(dy, dx)
    
    # Calculate drift angle
    drift = cog - traj_heading
    
    # Normalize drift angle to -π to π range
    drift = (drift + np.pi) % (2 * np.pi) - np.pi
    
    return drift

def cal_course(x1, y1, x2, y2, heading):
    """
    Parameters:
    x1, y1 (float): Coordinates of the first point
    x2, y2 (float): Coordinates of the second point
    heading (float): Heading at the second point (in radians)
    """
    # Calculate differences
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return heading
    # Calculate course over ground (COG)
    cog = np.arctan2(dy, dx)
    return cog

def calculate_drift_two_points(x1, y1, x2, y2, heading):
    """
    Calculate the drift angle given two points and the heading at the second point.
    
    Parameters:
    x1, y1 (float): Coordinates of the first point
    x2, y2 (float): Coordinates of the second point
    heading (float): Heading at the second point (in radians)
    
    Returns:
    float: Drift angle in radians
    """
    # Calculate differences
    dx = x2 - x1
    dy = y2 - y1
    
    # Calculate course over ground (COG)
    cog = np.arctan2(dy, dx)
    
    # Calculate drift angle
    drift = cog - heading
    
    # Normalize drift angle to -π to π range
    drift = (drift + np.pi) % (2 * np.pi) - np.pi
    
    return drift

def find_overlapping_intervals(intervals):
    n = len(intervals)
    result = [[] for _ in range(n)]
    
    # Sort intervals based on start time
    sorted_intervals = sorted(enumerate(intervals), key=lambda x: x[1][0])
    
    for i in range(n):
        current_idx, (start, end) = sorted_intervals[i]
        
        # Check for overlaps with intervals that start after the current one
        for j in range(i + 1, n):
            next_idx, (next_start, next_end) = sorted_intervals[j]
            
            if next_start <= end:
                result[current_idx].append(next_idx)
                result[next_idx].append(current_idx)
            else:
                # Since intervals are sorted, we can break once we find a non-overlapping interval
                break
    
    return result


def to_ego_frame(ego_state, neighbor_state):
    # Unpack ego state
    x_ego, y_ego, s_ego, h_ego = ego_state

    # Unpack neighbor state
    x_other, y_other, s_other, h_other = neighbor_state

    # Translation
    x_translated = x_other - x_ego
    y_translated = y_other - y_ego
    
    # Rotation
    cos_h = np.cos(-h_ego)
    sin_h = np.sin(-h_ego)
    
    x_rotated = x_translated * cos_h - y_translated * sin_h
    y_rotated = x_translated * sin_h + y_translated * cos_h
    
    # Adjust the heading    
    h_relative = (h_other - h_ego) % (2 * np.pi)
    
    return np.array([x_rotated, y_rotated, s_other, h_relative])

def transform_neighbors_to_ego_frame(ego_obs, neighbors_obs):
    # Get the current ego state
    current_ego_state = ego_obs[-1]
    
    # Prepare output array
    transformed_neighbors = np.zeros_like(neighbors_obs)
    
    # Transform each neighbor's observation history
    for i in range(neighbors_obs.shape[0]):  # For each neighbor
        for j in range(neighbors_obs.shape[1]):  # For each time step in history
            transformed_neighbors[i, j] = to_ego_frame(current_ego_state, neighbors_obs[i, j])
    
    return transformed_neighbors

def encode_unix_time(timestamp):
    """
    Convert Unix timestamp to time features.
    
    Args:
        timestamp: Unix timestamp (integer)
    
    Returns:
        numpy array of time features [hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos]
    """
    # Convert to datetime
    dt = datetime.fromtimestamp(timestamp, timezone.utc)
    
    # Minute features (60-minute cycle)
    minute = dt.minute
    minute_sin = np.sin(2 * np.pi * minute / 60)
    minute_cos = np.cos(2 * np.pi * minute / 60)
    
    # Hour features (24-hour cycle)
    hour = dt.hour
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    
    # Day of week features (7-day cycle)
    weekday = dt.weekday()
    day_sin = np.sin(2 * np.pi * weekday / 7)
    day_cos = np.cos(2 * np.pi * weekday / 7)
    
    # Month features (12-month cycle)
    month = dt.month - 1
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    
    return np.array([
        minute_sin, minute_cos,
        hour_sin, hour_cos,
        day_sin, day_cos,
        month_sin, month_cos
    ], dtype = np.float32)    

def find_max_overlaps(time_windows):
    """
    Find the maximum number of overlapping time windows.
    
    Args:
        time_windows: List of tuples (start_time, end_time)
        
    Returns:
        int: Maximum number of overlapping time windows
        float: Time point where maximum overlap occurs
    """
    # Create events with time and type (+1 for start, -1 for end)
    events = []
    for start, end in time_windows:
        events.append((start, 1))
        events.append((end, -1))
    
    # Sort events by time, if times are equal, process end before start
    events.sort(key=lambda x: (x[0], -x[1]))
    
    current_overlaps = 0
    max_overlaps = 0
    max_overlap_time = 0
    
    # Process events in chronological order
    for time, event_type in events:
        current_overlaps += event_type
        
        if current_overlaps > max_overlaps:
            max_overlaps = current_overlaps
            max_overlap_time = time
    
    return max_overlaps, max_overlap_time


def convert_trajectories_to_csv(trajs, times, overlap_idx):
    """
    Convert trajectory data to a CSV format, using sliding windows to create 10-timestep scenes.
    
    Parameters:
    trajs: list of numpy arrays, each with shape (n_timesteps, 4) containing [x, y, speed, heading]
    times: list of numpy arrays, each with shape (n_timesteps,) containing timestamps
    overlap_idx: list of lists containing indices of overlapping trajectories
    
    Returns:
    pandas.DataFrame: DataFrame with columns [SCENE_ID, TIMESTAMP, TRACK_ID, OBJECT_TYPE, X, Y, YAW]
    """
    all_records = []
    scene_counter = 0
    
    # Process each trajectory
    for i, (traj, time_array) in enumerate(zip(trajs, times)):
        # Get overlapping trajectories
        overlapping_trajs = overlap_idx[i]
        
        # Calculate number of scenes needed to cover all timesteps
        num_timesteps = len(time_array)
        
        # If trajectory length <= 10, create one scene
        if num_timesteps <= 10:
            start_indices = [0]
        else:
            # Calculate start indices for sliding windows
            # Regular windows from beginning
            regular_starts = list(range(0, num_timesteps - 9, 10))
            
            # Add final window if needed to capture remaining timesteps
            if num_timesteps - regular_starts[-1] > 10:
                regular_starts.append(num_timesteps - 10)
            
            start_indices = regular_starts
        
        # Process each scene
        for start_idx in start_indices:
            end_idx = min(start_idx + 10, num_timesteps)
            
            # Process the timestamps for ego vessel
            for t_idx in range(start_idx, end_idx):
                timestamp = time_array[t_idx]
                
                # Add record for the ego vessel (AGENT)
                record = {
                    'SCENE_ID': scene_counter,
                    'TIMESTAMP': f"{timestamp}",
                    'TRACK_ID': i,
                    'OBJECT_TYPE': 'AGENT',
                    'X': f"{traj[t_idx, 0]:.5f}",
                    'Y': f"{traj[t_idx, 1]:.5f}",
                    'YAW': f"{traj[t_idx, 3]:.5f}"
                }
                all_records.append(record)
                
                # Add records for overlapping trajectories (OTHER vessels)
                for other_idx in overlapping_trajs:
                    other_time_array = times[other_idx]
                    other_traj = trajs[other_idx]
                    
                    # Find the closest timestamp in the other trajectory
                    closest_t_idx = np.abs(other_time_array - timestamp).argmin()
                    
                    # Only add if the timestamps are exactly the same
                    if other_time_array[closest_t_idx] == timestamp:
                        record = {
                            'SCENE_ID': scene_counter,
                            'TIMESTAMP': f"{timestamp}",
                            'TRACK_ID': other_idx,
                            'OBJECT_TYPE': 'OTHER',
                            'X': f"{other_traj[closest_t_idx, 0]:.5f}",
                            'Y': f"{other_traj[closest_t_idx, 1]:.5f}",
                            'YAW': f"{other_traj[closest_t_idx, 3]:.5f}"
                        }
                        all_records.append(record)
            
            scene_counter += 1
    
    # Create DataFrame and sort by timestamp within each scene
    df = pd.DataFrame(all_records)
    df = df.sort_values(['SCENE_ID', 'TIMESTAMP'])
    
    # Convert TIMESTAMP back to float for proper sorting
    df['TIMESTAMP'] = df['TIMESTAMP'].astype(float)
    
    return df