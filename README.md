# ShipNaviSim

## News
- [2025/10/22] The code base of ShipNaviSim is available online. 
## Overview

The `ShipNaviSim` is a Gymnasium-compatible environment for simulating maritime vessel navigation in real-world scenarios. It allows for training and evaluating reinforcement learning agents on ship trajectory planning tasks with realistic ship dynamics, collision avoidance, and geographic constraints.

## Getting Started

> **📝 Quick Reference:** For complete working examples and end-to-end pipelines, check out:
> - **[`test_env.py`](test_env.py)** - Demonstrates environment setup, AIS data preparation, trajectory processing, video recording, Minari dataset creation, and metrics computation
> - **[`IL-BC.py`](IL-BC.py)** - Complete Behavioral Cloning (imitation learning) training and evaluation pipeline with multiple configuration options
>
> These files provide production-ready code that you can adapt for your own maritime navigation tasks.

## Installation

Set up the environment using a combination of conda and pip:

```bash
# First, create and activate the conda environment
conda env create -f environment.yml
conda activate ship-env

# Then install additional dependencies with pip
pip install -r requirements.txt
```

## Environment Details

### State Space

The environment operates with discrete timesteps, where each step represents `second_perts` seconds (default: 10 seconds) of simulation time. This means actions occur and observations are updated at 10-second intervals by default.

The observation space is a dictionary with the following components:

- **ego**: Ego vessel state history with shape `(observation_history_length + 1, features_per_agent)`
  - Features: x, y, speed, heading, [drift (optional)]
  - Temporal order: Ordered from past to present [t-(observation_history_length), ..., t-2, t-1, t], where t is the current timestep. The most recent observation is at index [-1].
  - Padding: When historical observations are not available (e.g., at the beginning of an episode), values of `[0, 0, 0, 0]` (or `[0, 0, 0, 0, 0]` with drift) are used

- **neighbors**: Neighboring vessels state history with shape `(n_neighbor_agents, observation_history_length + 1, features_per_agent)`
  - Features: x, y, speed, heading, [drift (optional)]
  - Temporal order: Same as ego, with each neighbor having history [t-(observation_history_length), ..., t-2, t-1, t]
  - Padding: When there are fewer neighboring vessels than `n_neighbor_agents` or a vessel is not present at a particular timestep, arrays of zeros are used

- **goal**: Target position coordinates with shape `(2,)`

- **nearest_dis** (optional): Distances to nearest vessels with shape `(n_neighbor_agents,)`
  - Included when `use_dis_fea=True`
  - Padding: When there are fewer neighboring vessels than `n_neighbor_agents`, maximum possible distance (diagonal of the region of interest) is used

- **time** (optional): Encoded time features (hour, day, month) with shape `(8,)`
  - Included when `use_time_fea=True`

The environment automatically manages these padding values to ensure consistent observation shapes regardless of the number of available vessels or history length.

### Action Space

The action space is a 3-dimensional continuous space:
- **dx**: Change in x-position (meters)
- **dy**: Change in y-position (meters)
- **dheading**: Change in heading (radians)

Actions can be normalized (-1 to 1) or in physical units depending on the `scale_act` parameter.

### Rewards

The environment provides flexible reward components that can be customized:
- **Goal-reaching reward**: Encourages reaching the destination
- **Near-miss penalty**: Penalizes close encounters with other vessels
- **Acceleration penalty**: Discourages excessive acceleration
- **Steering penalty**: Discourages sharp turns at high speeds

### Termination Conditions

An episode terminates when:
- The ego vessel reaches the goal position
- The ego vessel moves outside the region of interest
- The maximum number of steps is reached

## Usage

### Basic Usage

```python
import gymnasium as gym
from ship_env import ShipEnvironment

# Prepare trajectories, times and overlap_idx (see data preparation section)
env = ShipEnvironment(
    ship_trajectories=trajs,
    ship_times=times,
    overlap_idx=overlap_idx,
    region_of_interest=region_of_interest,
    ego_pos=0,  # Set ego vessel index
    observation_history_length=10,
    n_neighbor_agents=5
)

# Reset environment and get initial observation
obs, info = env.reset(seed=42)

# Simulation loop
done = False
while not done:
    # Generate action (replace with your agent's policy)
    action = env.action_space.sample()
    
    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Render environment
    env.render()
    
    done = terminated or truncated

env.close()
```

## AIS Data Format

The environment uses AIS (Automatic Identification System) data to simulate realistic vessel navigation scenarios. The input CSV data should follow this format:

### Data Schema

| Column | Type | Description | Units |
|--------|------|-------------|-------|
| `SHIP_ID` | Integer | Unique identifier for each vessel | - |
| `TIMESTAMP_UTC` | DateTime | UTC timestamp of the observation | `YYYY-MM-DD HH:MM:SS` |
| `LAT` | Float | Latitude coordinate | Decimal degrees |
| `LON` | Float | Longitude coordinate | Decimal degrees |
| `STATUS` | Integer | Vessel navigation status code | See below |
| `HEADING` | Float | Vessel heading/direction | Degrees (0-360°) |
| `SPEED_KNOTSX10` | Float | Vessel speed × 10 | Knots × 10 |

### Status Codes

The `STATUS` field follows AIS navigational status conventions:

| Code | Description |
|------|-------------|
| 0 | Under way using engine |
| 1 | At anchor |
| 2 | Not under command |
| 3 | Restricted maneuverability |
| 4 | Constrained by draught |
| 5 | Moored |
| 6 | Aground |
| 7 | Engaged in fishing |
| 8 | Under way sailing |
| 15 | Not defined |

### Example Data

```csv
SHIP_ID;TIMESTAMP_UTC;LAT;LON;STATUS;HEADING;SPEED_KNOTSX10
549;2017-08-05 23:03:40;1.202073;103.841191;0;107.24;58.17
549;2017-08-05 23:03:50;1.202167;103.841443;0;107.40;58.01
1024;2017-08-05 23:04:00;1.186234;103.853421;0;45.12;102.34
```

### Data Requirements

- **Delimiter**: Semi-colon (`;`) separated values
- **Time Resolution**: Typically 10-second intervals (configurable via `second_perts`)
- **Coordinate System**: WGS84 geographic coordinates
- **Speed Conversion**: Divide `SPEED_KNOTSX10` by 10 to get actual speed in knots
- **Temporal Continuity**: Each vessel should have continuous time series data

### Sample Dataset

A synthetic AIS dataset is provided in `raw_data/synthetic_ais_data.csv` for testing and development purposes. This dataset contains simulated vessel trajectories by running our trained BC policy in the Singapore Strait region.

### Data Preparation

The environment requires vessel trajectory data to be preprocessed:

1. Load position data from AIS records
2. Filter data based on region of interest and other criteria
3. Separate continuous trajectories into chunks
4. Interpolate trajectories to regular time intervals
5. Find overlapping vessel trajectories
6. Convert to format required by the environment

**Region of Interest:** The region of interest (ROI) defines the geographic boundaries of your navigation area. It is specified as a dictionary with longitude and latitude ranges in WGS84 decimal degrees: `{"LON": (min_lon, max_lon), "LAT": (min_lat, max_lat)}`. Episodes terminate if the ego vessel exits these boundaries. Choose regions with sufficient vessel traffic (typical areas: 5-50 km²). The ROI is also converted to array format `[[min_lon, min_lat], [max_lon, max_lat]]` for internal functions.

Example:
```python
from ship_env import chunk_to_traj, read_position_data
from split_chunks import seperate_chunks
import utils
import numpy as np

# Define region of interest
region_of_interest = {"LON": (103.82, 103.88), "LAT": (1.15, 1.22)}
region_of_interest_array = np.array([list(region_of_interest[k]) for k in ["LON", "LAT"]]).T

# Read position data
df = read_position_data("path/to/position_data.csv")

# Apply filters
geo_filters = [(v[0] <= pl.col(k)) & (pl.col(k) <= v[1]) for k, v in region_of_interest.items()]
df = df.filter(*geo_filters)

# Separate into chunks
threshold = timedelta(minutes=30)
chunks = seperate_chunks(df, threshold, minimum_sample_count=5)

# Process chunks into trajectories
trajs = []
times = []
for chunk in chunks:
    traj, time = chunk_to_traj(chunk, timedelta(seconds=10), region_of_interest_array)
    trajs.append(traj)
    times.append(time)

# Find overlapping trajectories
tw_lst = [(t[0], t[-1]) for t in times]
overlap_idx = utils.find_overlapping_intervals(tw_lst)
```

## Creating Datasets

The environment can be used to create offline datasets using Minari:

```python
from ship_env import create_minari_dataset

# Create dataset with expert trajectories
create_minari_dataset(env, "maritime_navigation_dataset", num_ships=100)
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `ship_trajectories` | List of trajectory arrays | Required |
| `ship_times` | List of timestamp arrays | Required |
| `overlap_idx` | List of overlapping vessel indices | Required |
| `region_of_interest` | Geographic boundaries | Required |
| `ego_pos` | Index of ego vessel | 0 |
| `observation_history_length` | Number of past observations | 10 |
| `n_neighbor_agents` | Number of neighbors to track | 5 |
| `render_histL` | Length of history to render | 1000 |
| `normalize_xy` | Whether to normalize coordinates | False |
| `max_steps` | Maximum steps per episode | 1000 |
| `second_perts` | Seconds per timestep | 10 |
| `use_time_fea` | Use time features | False |
| `drop_neighbor` | Exclude neighbor observations | False |
| `use_dis_fea` | Include distance features | False |
| `use_drift_fea` | Include drift features | False |
| `use_FoR` | Use frame of reference transformation | False |
| `scale_act` | Scale actions to [-1, 1] | False |

## Metrics

The environment tracks numerous metrics for evaluating navigation performance:

| Metric | Description |
|--------|-------------|
| `avg_speed` | Average speed over episode |
| `min_speed` | Minimum speed |
| `max_speed` | Maximum speed |
| `avg_acceleration` | Average acceleration |
| `min_acceleration` | Minimum acceleration |
| `max_acceleration` | Maximum acceleration |
| `avg_curvature` | Average path curvature |
| `min_curvature` | Minimum curvature |
| `max_curvature` | Maximum curvature |
| `nearmiss_rate` | Percentage of timesteps with near misses |
| `gc_ade` | Average displacement error |
| `mae_steer` | Mean absolute error in steering |
| `mae_accel` | Mean absolute error in acceleration |
| `curv_change_rate` | Rate of curvature changes (maneuvers) |
| `min_drift` | Minimum drift angle |
| `max_drift` | Maximum drift angle |
| `avg_drift` | Average drift angle |
| `eps_length` | Episode length |
| `avg_neigh` | Average number of neighboring vessels |
| `max_neigh` | Maximum number of neighboring vessels |

### Computing Metrics for Multiple Trajectories

You can compute and save metrics for a batch of trajectories using the following code:

```python
# Compute metrics for all expert trajectories
lst_infos = []
for id_ship in range(num_ships):
    print(id_ship)
    val = replay_ship_env(id_ship, env)
    if(val is None):
        continue
    val["ship_id"] = chunks[id_ship]["SHIP_ID"][0]
    val["start_time"] = chunks[id_ship]["start_time"][0]
    lst_infos.append(val)

# Create DataFrame with metrics and save to CSV
df_exp_infos = pd.DataFrame(lst_infos)
df_exp_infos.drop(["CPD"], axis=1, inplace=True)
print(df_exp_infos)
df_exp_infos.to_csv(f"exp_stats_{plan_start_time}to{plan_end_time}.csv", sep=";", index=False)
```

This code replays each trajectory in the environment, collects the metrics, and saves them to a CSV file for further analysis.

## Visualization

The environment includes a pygame-based visualization to help understand vessel behavior:

```python
obs, info = env.reset(seed=42, options={'ego_pos': 0})
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    env.render()  # Visualize current state
```

The visualization displays:
- Ego vessel (red)
- Neighboring vessels (blue)
- Vessel trajectories
- Goal position (star icon)
- Current metrics (step count, CPD, distance to goal, curvature)

## Advanced Features

### Frame of Reference Transformation

When `use_FoR=True`, neighboring vessel observations are transformed to the ego vessel's reference frame, which can make learning easier for RL agents.

### Expert Demonstration

The environment can extract expert actions from real trajectories for imitation learning:

```python
obs, info = env.reset(seed=42, options={'ego_pos': 0})
expert_actions = info['actions']  # Expert actions from real data
```

### Behavioral Cloning (BC) Training

The repository includes a complete implementation for training navigation agents using Behavioral Cloning on expert trajectories. BC learns to imitate expert vessel navigation behavior from the AIS data.

#### Overview

Behavioral Cloning trains a neural network policy to replicate expert actions by supervised learning on state-action pairs extracted from real vessel trajectories. This approach is particularly effective for maritime navigation where high-quality expert demonstrations are available from AIS data.

#### Configuration

The BC training is configured using a dataclass with the following key parameters:

```python
@dataclass
class TrainConfig:
    mode: str = "train"              # "train", "eval", or "train-eval"
    project: str = "Maritime"        # WandB project name
    group: str = "BC"                # WandB group name
    name: str = "BC-deterministic-256-128hid-256batch-combMLPTanh"
    env: str = "Maritime-Expert-v1"  # Minari dataset name
    use_combMLP: bool = True         # Use combined MLP architecture
    use_det: bool = True             # Deterministic policy
    max_timesteps: int = int(1e6)    # Total gradient updates
    hid_size: int = 256              # Hidden layer size
    batch_size: int = 256            # Training batch size
    use_her: bool = False            # Hindsight Experience Replay
    seed: int = 42                   # Random seed
    device: str = "cuda"             # Training device
    wandb: bool = False              # Enable WandB logging
    ckpoint_folder: str = "./ckpoints"
    ckpoint_path: str = "./ckpoints/BC-deterministic-256-128hid-256batch-combMLPTanh-Maritime-Expert-v1.th"
```

#### Training a BC Agent

**Step 1: Create a Minari Dataset**

First, create a dataset from expert trajectories:

```python
from ship_env import create_minari_dataset

# Create dataset with expert trajectories
create_minari_dataset(env, "Maritime-Expert-v1", num_ships=100)
```

**Step 2: Train the BC Policy**

Run the training script with your desired configuration:

```bash
# Basic training
python IL-BC.py --mode train --env Maritime-Expert-v1

# Training with custom parameters
python IL-BC.py --mode train \
    --env Maritime-Expert-v1 \
    --batch_size 256 \
    --hid_size 256 \
    --use_det True \
    --use_combMLP True \
    --wandb True
```

Or programmatically:

```python
import pyrallis
from IL-BC import TrainConfig, load_ships_and_play

config = TrainConfig(
    mode="train",
    env="Maritime-Expert-v1",
    batch_size=256,
    hid_size=256,
    use_det=True,
    wandb=False
)

load_ships_and_play(config)
```

**Step 3: Monitor Training**

Training logs are saved to `logs/BC/` and can be visualized with TensorBoard:

```bash
tensorboard --logdir logs/BC/
```

If WandB is enabled, metrics are automatically uploaded to Weights & Biases.

#### Policy Architectures

The implementation supports two policy types:

**1. Stochastic Policy** (`use_det=False`)
- Standard Actor-Critic architecture
- Outputs action distribution (mean and std)
- Suitable for exploration during training

**2. Deterministic Policy** (`use_det=True`)
- Direct action prediction
- No sampling, outputs exact action
- Better for precise trajectory following

Both policies support:
- **CombinedExtractor**: Separate networks for each observation key, then concatenate
- **CombinedNormExtractor**: Adds running normalization to combined features
- **NewCombinedNormExtractor** (`use_combMLP=True`): Custom MLP with Tanh activation

#### Hindsight Experience Replay (HER)

Enable HER to improve sample efficiency by creating additional training data:

```python
config = TrainConfig(
    use_her=True,  # Enable HER
    # ... other parameters
)
```

HER replaces the original goal with actually achieved positions, helping the agent learn even from failed trajectories.

#### Evaluation

**Evaluate a Trained Policy:**

```bash
# Evaluate only
python IL-BC.py --mode eval --ckpoint_path ./ckpoints/your_model.th

# Train and evaluate
python IL-BC.py --mode train-eval
```

**Evaluation Output:**

The evaluation runs the trained policy on all trajectories and saves comprehensive metrics:

```python
# Example output statistics
df_bc_infos[["gc_ade", "mae_steer", "mae_accel"]].describe()
#        gc_ade     mae_steer   mae_accel
# count  100.00     100.00      100.00
# mean   12.45      0.089       0.034
# std    8.23       0.045       0.021
# min    2.11       0.012       0.005
# max    45.67      0.234       0.098
```

Results are saved to: `{name}{env}_stats.csv`

#### Key Metrics for BC Evaluation

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| `gc_ade` | Average Displacement Error | Lower is better - measures trajectory accuracy |
| `mae_steer` | Mean Absolute Steering Error | Lower is better - steering precision |
| `mae_accel` | Mean Absolute Acceleration Error | Lower is better - speed control accuracy |
| `nearmiss_rate` | Near-miss percentage | Lower is better - safety measure |
| `eps_length` | Episode length | Shorter indicates efficiency |

### Recording Videos

The environment can be wrapped with Gymnasium's RecordVideo wrapper:

```python
env = gym.wrappers.RecordVideo(env, "videos/", episode_trigger=lambda ep_id: True)
```

## Notes

- The environment is designed for research purposes and should be validated before use in real-world applications.
- Performance depends on the quality and coverage of the input AIS data.
- Computational requirements increase with the number of neighboring vessels and observation history length.

