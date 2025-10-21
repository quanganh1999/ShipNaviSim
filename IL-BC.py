from dataclasses import dataclass, asdict
import pyrallis
import gymnasium as gym
import pygame
import wandb
import pandas as pd
from gymnasium import spaces
from datetime import timedelta
import numpy as np
import polars as pl
from typing import List
import utils
import minari
from imitation.data import rollout, types
from det_bc import evaluate_policy_with_finalinfo
from imitation.util import logger as imit_logger
from typing import (
    Type
)
from imitation.algorithms import bc
from stable_baselines3.common import policies, torch_layers
from torch import nn
import torch as th
from imitation.util import networks, util    
from stable_baselines3.common.vec_env import DummyVecEnv
import det_bc
import polars as pl
from ship_env import ShipEnvironment, read_position_data, chunk_to_traj
from datetime import timedelta
import utils
import pandas as pd
import numpy as np
import gymnasium as gym
import minari

class CombinedNormExtractor(torch_layers.CombinedExtractor):
    """
    Combined features extractor for Dict observation spaces.
    Flatten obs of each key then concat and normalize
    """
    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_output_dim: int = 256,
        normalized_image: bool = False,
        normalize_class: Type[nn.Module] = networks.RunningNorm,
    ) -> None:
        super().__init__(observation_space, cnn_output_dim, normalized_image)
        self.normalize = normalize_class(self.features_dim)  # type: ignore[call-arg]
    
    def forward(self, observations: th.Tensor) -> th.Tensor:
        flattened = super().forward(observations)
        return self.normalize(flattened)

@dataclass
class TrainConfig:
    #Mode (train or eval or train-eval for both)
    mode: str = "train"
    # wandb project name
    project: str = "Maritime"
    # wandb group name
    group: str = "BC"
    # wandb run name
    name: str = "BC-deterministic-256-128hid-256batch-combMLPTanh"
    # training dataset name
    env: str = "Maritime-Expert-v1"
    #use CombinedMLP
    use_combMLP: bool = True
    #use_deterministic policy or not:
    use_det: bool = True
    # total gradient updates during training
    max_timesteps: int = int(1e6)
    #NN hid size
    hid_size = 256
    # training batch size
    batch_size: int = 256
    #Using HER 
    use_her: bool = False
    # training random seed
    seed: int = 42
    # training device
    device: str = "cuda"
    # wandb or not
    wandb: bool = False
    #checkpoint path
    ckpoint_folder: str = "./ckpoints"
    #checkpoint path
    ckpoint_path:str = "./ckpoints/BC-deterministic-256-128hid-256batch-combMLPTanh-Maritime-Expert-v1.th"

def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"] +"-"+config["env"],
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=False,  # optional        
    )
    wandb.run.save()

        
@pyrallis.wrap()        
def load_ships_and_play(cfg: TrainConfig):    
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
    
    #BC 
    #Convert minari dataset to trajectories of sb3    
    if cfg.mode == "train":
        if cfg.wandb:
            wandb_init(asdict(cfg))
        rng = np.random.default_rng(42)
        minari_dataset = minari.load_dataset(cfg.env)
        trajs: List[types.TrajectoryWithRew] = []
        count_nb = 0
        for episode_data in minari_dataset.iterate_episodes():
            observations = episode_data.observations                       
            actions = episode_data.actions
            rewards = episode_data.rewards
            terminations = episode_data.terminations
            truncations = episode_data.truncations
            infos = episode_data.infos                   
            new_obs = types.maybe_wrap_in_dictobs(observations)        
            # print(new_obs)
            trajs.append(types.TrajectoryWithRew(new_obs, actions, None, True, rewards))
            if cfg.use_her:
                her_observations = observations.copy()
                for i in range(len(her_observations['ego']) - 1):            
                    her_observations['goal'][i] = her_observations['ego'][i+1][-1][:2]  
                new_obs = types.maybe_wrap_in_dictobs(her_observations)        
                # print(new_obs)
                trajs.append(types.TrajectoryWithRew(new_obs, actions, None, True, rewards))
            if(len(actions) > 120):
                count_nb += 1
            assert len(new_obs) == len(actions) + 1
            assert len(rewards) == len(actions)
            #     
        rng.shuffle(trajs)
        stats = rollout.rollout_stats(trajs)
        print("num over 20 mins", count_nb)
        print(stats)   
        #Training part    
        transitions = rollout.flatten_trajectories(trajs)    
        log_dir = "logs/BC"
        custom_logger = imit_logger.configure(
            folder=log_dir,
            format_strs=["tensorboard", "stdout"],
        )    
        # custom_logger=None
        if cfg.use_det is False:      
            policy = policies.ActorCriticPolicy(
                        observation_space=env.observation_space,
                        action_space=env.action_space,
                        net_arch=[cfg.hid_size, cfg.hid_size],
                        # log_std_init = 1, #deterministic policy 
                        # Set lr_schedule to max value to force error if policy.optimizer
                        # is used by mistake (should use self.optimizer instead).
                        lr_schedule=lambda _: th.finfo(th.float32).max,
                        # features_extractor_class=torch_layers.CombinedExtractor
                        features_extractor_class=CombinedNormExtractor,
                    )    
            bc_trainer = bc.BC(
                observation_space=env.observation_space,
                action_space=env.action_space,
                demonstrations=transitions,
                rng=rng,
                policy=policy,
                l2_weight=0.001,
                # ent_weight=0,
                batch_size=cfg.batch_size,
                custom_logger=custom_logger
                )        
        else:
            if cfg.use_combMLP is False:
                policy = det_bc.DetPolicy(
                            observation_space=env.observation_space,
                            action_space=env.action_space,
                            net_arch=[cfg.hid_size, cfg.hid_size],                    
                            features_extractor_class=CombinedNormExtractor,
                        )       
            else:
                policy = det_bc.DetPolicy(
                            observation_space=env.observation_space,
                            action_space=env.action_space,
                            net_arch=[cfg.hid_size],                    
                            features_extractor_class=det_bc.NewCombinedNormExtractor,
                        )            
            bc_trainer = det_bc.BC(
                observation_space=env.observation_space,
                action_space=env.action_space,
                demonstrations=transitions,
                rng=rng,
                policy=policy,
                l2_weight=0.001,            
                batch_size=cfg.batch_size,
                custom_logger=custom_logger
                )
        bc_trainer.train(n_epochs=300)  
        utils.create_directory(cfg.ckpoint_folder)
        util.save_policy(bc_trainer.policy, f"{cfg.ckpoint_folder}/{cfg.name}-{cfg.env}.th")    
    if "eval" in cfg.mode:        
        #Evaluating part
        if cfg.use_det is False:
            eval_policy = bc.reconstruct_policy(f"{cfg.ckpoint_path}")
        else:
            eval_policy = det_bc.reconstruct_policy(f"{cfg.ckpoint_path}")
        env = DummyVecEnv([lambda: env])
        lst_infos = []
        for id_ego in range(num_ships):
            env.set_options({'ego_pos': id_ego})
            _, _, infos = evaluate_policy_with_finalinfo(eval_policy, env, 1, return_episode_rewards=True, render=False)
            print(id_ego)
            val = infos[0]
            val.pop('TimeLimit.truncated', None)
            val.pop('terminal_observation', None)
            val["ship_id"] = chunks[id_ego]["SHIP_ID"][0]
            val["start_time"] = chunks[id_ego]["start_time"][0]
            lst_infos.append(val)            
        df_bc_infos = pd.DataFrame(lst_infos)
        df_bc_infos.drop(["CPD"], axis=1, inplace=True)
        print(df_bc_infos)                
        print(df_bc_infos[["gc_ade", "mae_steer", "mae_accel"]].describe())
        df_bc_infos.to_csv(f"{cfg.name+cfg.env}_stats.csv", sep=";", index=False)
    env.close()
    

if __name__ == "__main__":
    th.manual_seed(42)
    load_ships_and_play()