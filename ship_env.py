import gymnasium as gym
import pygame
from datetime import timedelta
import numpy as np
import polars as pl
import utils
import minari
from collections import deque
from pygame import gfxdraw

class ShipEnvironment(gym.Env):
    def __init__(self, ship_trajectories, ship_times, overlap_idx, region_of_interest,
                 ego_pos: int = 0, observation_history_length: int = 10, n_neighbor_agents: int = 5, render_histL: int = 1000,
                 normalize_xy: bool = False, max_steps: int = 1000, second_perts = 10, use_time_fea=False,
                 drop_neighbor=False, use_dis_fea=False, use_drift_fea=False, use_FoR = False, scale_act=False):
        super(ShipEnvironment, self).__init__()        
        self.ship_trajectories = ship_trajectories
        self.ship_times = ship_times
        self.overlap_idx = overlap_idx
        self.region_of_interest = region_of_interest
        self.ego_pos = ego_pos
        self.observation_history_length = observation_history_length
        self.n_neighbor_agents = n_neighbor_agents
        self.normalize_xy = normalize_xy
        self.max_steps = max_steps
        self.second_perts = second_perts #seconds per timestep
        self.CPD = 0 #Closest Point Distance
        self.render_histL = render_histL
        self.nearmiss_threshold = 555
                        
        # Calculate the dimensions of the area in meters
        origin_lon, origin_lat = region_of_interest['LON'][0], region_of_interest['LAT'][0]
        # self.max_x = region_of_interest['LON'][1] - origin_lon
        # self.max_y = region_of_interest['LAT'][1] - origin_lat
        self.max_x = utils.haversine_distance(origin_lon, origin_lat, region_of_interest['LON'][1], origin_lat)
        self.max_y = utils.haversine_distance(origin_lon, origin_lat, origin_lon, region_of_interest['LAT'][1])        
        
        # Define action space (dx, dy, dheading)        
        if scale_act is False:
            self.action_space = gym.spaces.Box(
                low=np.array([-self.max_x/20, -self.max_y/20, -np.pi/2]),
                high=np.array([self.max_x/20, self.max_y/20, np.pi/2]),
                dtype=np.float32
            )
        else:
            self.action_space = gym.spaces.Box(
                low=np.array([-1, -1, -1]),
                high=np.array([1, 1, 1]),
                dtype=np.float32
            )
        
        self.features_per_agent = 4  # x, y, speed, heading
        if use_drift_fea:
            self.features_per_agent += 1 # x, y, speed, heading, drift
        # Define observation space
        ego_obs_shape = (self.observation_history_length + 1, self.features_per_agent)
        neighbor_obs_shape = (self.n_neighbor_agents, self.observation_history_length + 1, self.features_per_agent)

        if self.normalize_xy:
            obs_low = np.array([-1, -1, 0, 0], dtype=np.float32)
            obs_high = np.array([1, 1, 1000, 2*np.pi], dtype=np.float32)
        else:
            obs_low = np.array([0, 0, 0, 0], dtype=np.float32)
            obs_high = np.array([self.max_x, self.max_y, 1000, 2*np.pi], dtype=np.float32)
            if use_drift_fea:
                obs_low = np.array([0, 0, 0, 0, -2*np.pi], dtype=np.float32)
                obs_high = np.array([self.max_x, self.max_y, 1000, 2*np.pi, 2*np.pi], dtype=np.float32)
        
        self.padded_val = np.array([0, 0, 0, 0], dtype=np.float32)        
        if use_drift_fea:
            self.padded_val = np.array([0, 0, 0, 0, 0], dtype=np.float32)        
        obs_space_dict = {
            'ego': gym.spaces.Box(
                low=np.broadcast_to(obs_low,  ego_obs_shape),
                high=np.broadcast_to(obs_high,  ego_obs_shape),
                shape=ego_obs_shape,
                dtype=np.float32
            ),
            'neighbors': gym.spaces.Box(
                low=np.broadcast_to(obs_low,  neighbor_obs_shape),
                high=np.broadcast_to(obs_high,  neighbor_obs_shape),
                shape=neighbor_obs_shape,
                dtype=np.float32
            ),
            'nearest_dis':gym.spaces.Box(
                low=0,
                high=np.sqrt(self.max_x**2+self.max_y**2),
                shape=(self.n_neighbor_agents,),
                dtype=np.float32
            ),
            'time': gym.spaces.Box(low=np.array([-1]*8), high=np.array([1]*8), dtype=np.float32),
            "goal":gym.spaces.Box(
                low=obs_low[:2],
                high=obs_high[:2],
                dtype=np.float32
            ),                
        }
        if use_FoR:
            #Due to rotating we can't control
            new_obs_low = np.array([-np.inf, -np.inf, 0, 0], dtype=np.float32)
            new_obs_high = np.array([np.inf, np.inf, 1000, 2*np.pi], dtype=np.float32)
            obs_space_dict['neighbors'] = gym.spaces.Box(
                low=np.broadcast_to(new_obs_low,  neighbor_obs_shape),
                high=np.broadcast_to(new_obs_high,  neighbor_obs_shape),
                shape=neighbor_obs_shape,
                dtype=np.float32
            )
        self.use_drift_fea = use_drift_fea
        self.drop_neighbor = drop_neighbor
        self.use_dis_fea = use_dis_fea
        self.use_FoR = use_FoR
        self.use_time_fea = use_time_fea
        self.scale_act = scale_act
        if drop_neighbor:
            obs_space_dict.pop("neighbors", None)
        if use_dis_fea is False:
            obs_space_dict.pop("nearest_dis", None)
        if use_time_fea is False:
            obs_space_dict.pop("time", None)
        self.observation_space = gym.spaces.Dict(obs_space_dict)
        
        self.goal_threshold = 200  # meters, can be adjusted
        self.max_dx = 0
        self.max_dy = 0
        self.count_dx = []
        # Pygame initialization
        self.render_mode = "rgb_array1"
        pygame.init()
        self.screen_width = 1500
        self.screen_height = 1500
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.font = pygame.font.Font(None, 36)         
        star_img = pygame.image.load('./star-icon.jpg')
        self.star_img = pygame.transform.scale(star_img, (30, 30))  # Adjust size as needed
    
    def render(self):
        self.screen.fill((255, 255, 255))  # Fill screen with white

        # Calculate scaling factors
        scale_x = self.screen_width / self.max_x
        scale_y = self.screen_height / self.max_y

        # Draw ego ship
        ego_x, ego_y = self.current_state['x'], self.current_state['y']
        ego_heading = self.current_state['heading']
        if(len(self.save_past_ego) != 0):                  
            ego_heading = utils.cal_course(self.save_past_ego[0][0],self.save_past_ego[0][1], ego_x, ego_y, ego_heading)
        
        if self.normalize_xy:
            ego_x = (ego_x + 1) * self.max_x / 2
            ego_y = (ego_y + 1) * self.max_y / 2
        
        screen_x = int(ego_x * scale_x)
        screen_y = int(self.screen_height - ego_y * scale_y)  # Flip y-coordinate
        
        # Draw custom ego ship
        cartwidth = 25.0
        cartheight = 10.0
        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r+(r-l)/4, (t+b)/2), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-ego_heading)  # Note the negative sign
            coord = (coord[0] + screen_x, coord[1] + screen_y)
            pole_coords.append(coord)
    
        ship_color = (255, 0, 0)  # Red for ego ship
        gfxdraw.aapolygon(self.screen, pole_coords, ship_color)
        gfxdraw.filled_polygon(self.screen, pole_coords, ship_color)
        
        # Draw ego ship's history
        ego_obs = self.current_obs['ego']
        # for i in range(len(ego_obs) - 1):  # Exclude the current state
        #     if not np.array_equal(ego_obs[i], self.padded_val):  # Check if the observation is not a padded value        
        for i in range(min(len(self.save_past_ego), self.render_histL)):
            val_hist_obs = self.save_past_ego[i]            
            if not np.array_equal(val_hist_obs, self.padded_val):
                hist_x, hist_y = val_hist_obs[:2]
                if self.normalize_xy:
                    hist_x = (hist_x + 1) * self.max_x / 2
                    hist_y = (hist_y + 1) * self.max_y / 2
                hist_screen_x = int(hist_x * scale_x)
                hist_screen_y = int(self.screen_height - hist_y * scale_y)  # Flip y-coordinate
                pygame.draw.circle(self.screen, (255, 0, 0), (hist_screen_x, hist_screen_y), 3)
        #Draw expert_obs
        if not np.array_equal(self.expert_obs[-1], self.padded_val):       
            for i in range(min(len(self.save_past_exp)-1, self.render_histL)):
                val_hist_obs = self.save_past_exp[i]            
                if not np.array_equal(val_hist_obs, self.padded_val):
                    hist_x, hist_y = val_hist_obs[:2]
                    if self.normalize_xy:
                        hist_x = (hist_x + 1) * self.max_x / 2
                        hist_y = (hist_y + 1) * self.max_y / 2
                    hist_screen_x = int(hist_x * scale_x)
                    hist_screen_y = int(self.screen_height - hist_y * scale_y)  # Flip y-coordinate
                    pygame.draw.circle(self.screen, (0, 255, 0), (hist_screen_x, hist_screen_y), 3)
        # Draw neighboring ships
        neighbor_obs = self.current_obs['neighbors']
        for neighbor in neighbor_obs:
            if not np.array_equal(neighbor[-1], self.padded_val):
                neighbor_x, neighbor_y = neighbor[-1][:2]
                neighbor_h = neighbor[-1][3]
                if len(neighbor) > 1:
                    neighbor_h = utils.cal_course(neighbor[-2][0], neighbor[-2][1], neighbor_x, neighbor_y, neighbor_h)
                if self.normalize_xy:
                    neighbor_x = (neighbor_x + 1) * self.max_x / 2
                    neighbor_y = (neighbor_y + 1) * self.max_y / 2
                screen_x = int(neighbor_x * scale_x)
                screen_y = int(self.screen_height - neighbor_y * scale_y)  # Flip y-coordinate
                # pygame.draw.circle(self.screen, (0, 0, 255), (screen_x, screen_y), 5)
                cartwidth = 25.0
                cartheight = 10.0
                l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
                pole_coords = []
                for coord in [(l, b), (l, t), (r, t), (r+(r-l)/4, (t+b)/2), (r, b)]:
                    coord = pygame.math.Vector2(coord).rotate_rad(-neighbor_h)  # Note the negative sign
                    coord = (coord[0] + screen_x, coord[1] + screen_y)
                    pole_coords.append(coord)
    
                ship_color = (0, 0, 255)  # Red for ego ship
                gfxdraw.aapolygon(self.screen, pole_coords, ship_color)
                gfxdraw.filled_polygon(self.screen, pole_coords, ship_color)
                #history
                for i in range(len(neighbor) - 1):  # Exclude the current state
                    if not np.array_equal(neighbor[i], self.padded_val):  # Check if the observation is not a padded value
                        hist_x, hist_y = neighbor[i][:2]                    
                        hist_screen_x = int(hist_x * scale_x)
                        hist_screen_y = int(self.screen_height - hist_y * scale_y)  # Flip y-coordinate
                        pygame.draw.circle(self.screen, ship_color, (hist_screen_x, hist_screen_y), 3)

        # Draw goal position
        goal_x, goal_y = self.goal_position
        screen_goal_x = int(goal_x * scale_x)
        screen_goal_y = int(self.screen_height - goal_y * scale_y)  # Flip y-coordinate
        # Calculate the position to center the image at the goal point
        star_rect = self.star_img.get_rect(center=(screen_goal_x, screen_goal_y))
        # Draw the star image
        self.screen.blit(self.star_img, star_rect)
        # pygame.draw.circle(self.screen, (0, 255, 0), (screen_goal_x, screen_goal_y), 5)

        # Draw step count and other information
        step_text = self.font.render(f"Ego id: {self.ego_pos}, Step: {self.current_step}, CPD: {self.CPD:.2f}, Dis-to-goal: {self.compute_dis_to_goal():.2f}, Curv: {self.current_curv:.2f}", 
                                     True, (0, 0, 0))
        self.screen.blit(step_text, (10, 10))
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )        

    def close(self):
        pygame.quit()
    
    def compute_dis_to_goal(self):
        goal_x, goal_y = self.goal_position 
        distance_to_goal = np.sqrt((self.current_state['x'] - goal_x)**2 + (self.current_state['y'] - goal_y)**2)
        return distance_to_goal
    def _check_goal_reached(self):
        # Implement goal-checking logic here        
        return self.compute_dis_to_goal() < self.goal_threshold  
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if options is not None and options.get('ego_pos', None) != None:
            self.ego_pos = options['ego_pos']
        self.current_step = 0
        self.CPD = 0 #Reset Closest Point Distance
        self.lastdheading = 0 
        #Set new goal based on ego agent       
        final_state = self.ship_trajectories[self.ego_pos][-1]
        self.goal_position = (final_state[0], final_state[1])
        ego_goal_x, ego_goal_y = self.goal_position        
        #Need to normalize due to its original
        if self.normalize_xy:            
            ego_goal_x = (ego_goal_x / self.max_x) * 2 - 1
            ego_goal_y = (ego_goal_y / self.max_y) * 2 - 1    
        self.current_obs = {'ego': None, 'neighbors': None, 'FoR_neigh':None, 'nearest_dis': None, 'goal': np.array([ego_goal_x, ego_goal_y])} 
        self.expert_obs = None
        # Initialize current_state based on the first entry of ego ship's trajectory               
        initial_state = self.ship_trajectories[self.ego_pos][0]     
        initial_time = self.ship_times[self.ego_pos][0]
        self.current_obs['ego'] = np.array([self.padded_val]*(self.observation_history_length + 1))
        self.current_obs['ego'][-1] = initial_state        
        self.current_state = {            
            'x': initial_state[0],
            'y': initial_state[1],
            'speed': initial_state[2],
            'heading': initial_state[3],
            'timestamp': initial_time
        }        
        # Set goal_position based on the last entry of ego ship's trajectory                
        self.total_neigh = 0
        self.max_neigh = 0
        self.save_past_ego = deque() #used for saving last ego states
        self.save_past_exp = deque() #used for saving last exp states
        observation = self._get_observation() #Change neighoring agents info
        self.exp_acclerations = []
        self.exp_actions = self.gen_actions_from_data()               
        # Initialize metric        
        self.total_speed = 0
        self.min_speed = float('inf')
        self.max_speed = float('-inf')
        self.total_acceleration = 0
        self.min_acceleration = float('inf')
        self.max_acceleration = float('-inf')
        self.total_curvature = 0
        self.min_curvature = float('inf')
        self.max_curvature = float('-inf')
        self.nearmiss_count = 0
        self.current_curv = 0
        self.totalADE = 0
        self.totalASE = 0
        self.totalAAE = 0
        self.stop_step = 0
        self.count_cur_change = 0
        self.prv_cur = 0
        self.total_veh_in = 0
        self.total_drift = 0
        self.min_drift = float('inf')
        self.max_drift = float('-inf')        
        observation = observation.copy()
        if self.drop_neighbor:
            observation.pop("neighbors", None)
        if self.use_dis_fea is False:
            observation.pop('nearest_dis', None)
        if self.use_FoR is True:                
            observation['neighbors'] = observation['FoR_neigh']
        if self.use_time_fea:
            observation.update({'time': utils.encode_unix_time(initial_time)})
        observation.pop('FoR_neigh', None)        
        return observation, {'goal_position': self.goal_position, 'actions': self.exp_actions}
                
    def step(self, action):
        # Unpack the action
        dx, dy, dheading = action              
        # Scale the actions
        if self.normalize_xy:
            # Convert normalized dx, dy to actual distances
            dx_scaled = dx * (self.max_x / 2)
            dy_scaled = dy * (self.max_y / 2)
        else:
            if self.scale_act is False:
                dx_scaled = dx
                dy_scaled = dy
            else:
                # Scale dx, dy based on the environment size            
                dx_scaled = dx * (self.max_x / 20)  # Max step is 5% of environment width
                dy_scaled = dy * (self.max_y / 20)  # Max step is 5% of environment height

        # Scale dheading (same for both normalized and non-normalized cases)
        if self.scale_act:
            dheading_scaled = dheading * (np.pi / 2)  # Max heading change is 90 degrees
        else:
            dheading_scaled = dheading
        self.lastdheading = dheading_scaled
        self.save_past_ego.appendleft(self.current_obs["ego"][-1])
        # Update position and heading without clipping
        new_x = self.current_state['x'] + dx_scaled       
        new_y = self.current_state['y'] + dy_scaled
        new_heading = (self.current_state['heading'] + dheading_scaled) % (2 * np.pi)

        #Cal drift metrics:
        val_drift = utils.calculate_drift_two_points(self.current_state['x'], self.current_state['y'], new_x, new_y, new_heading)
        self.min_drift = min(self.min_drift, val_drift)
        self.max_drift = max(self.max_drift, val_drift)
        self.total_drift += abs(val_drift)
        
        # Calculate new speed (optional, depends on your requirements)
        new_speed = np.sqrt(dx_scaled**2 + dy_scaled**2)/self.second_perts  # Example: speed as magnitude of movement

        # Update timestamp
        new_timestamp = self.current_state['timestamp'] + self.second_perts  

        # Check if the new position is within the region of interest
        within_region = (0 <= new_x <= self.max_x) and (0 <= new_y <= self.max_y)

        # Store previous state for reward calculation
        previous_state = self.current_state.copy()

        # Update current state
        self.current_state = {
            'x': new_x,
            'y': new_y,
            'heading': new_heading,
            'speed': new_speed,
            'timestamp': new_timestamp
        }
         # Calculate acceleration
        old_speed = previous_state['speed']
        acceleration = (new_speed - old_speed) / self.second_perts

        # Calculate curvature (using the formula for curvature of a parametric curve)
        dx_dt = new_speed * np.cos(new_heading)
        dy_dt = new_speed * np.sin(new_heading)
        d2x_dt2 = acceleration * np.cos(new_heading) - new_speed * np.sin(new_heading) * dheading_scaled
        d2y_dt2 = acceleration * np.sin(new_heading) + new_speed * np.cos(new_heading) * dheading_scaled
        if (dx_dt**2 + dy_dt**2 < 1e-3): 
            curvature = 0   # If there's no movement at all, set curvature to zero
        else:
            #ref: https://mathworld.wolfram.com/Curvature.html#:~:text=)%20then%20gives-,(13),-For%20a%20two
            curvature = (dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / (dx_dt**2 + dy_dt**2)**1.5
            # if curvature > 1:
            #     print(old_speed, previous_state["heading"])
            #     print(new_heading, new_speed)
            #     print((dx_dt**2 + dy_dt**2))
            #     assert 1==-1

        # Update metrics
        self.total_speed += new_speed
        self.min_speed = min(self.min_speed, new_speed)
        self.max_speed = max(self.max_speed, new_speed)
        self.total_acceleration += acceleration
        self.min_acceleration = min(self.min_acceleration, acceleration)
        self.max_acceleration = max(self.max_acceleration, acceleration)
        self.min_curvature = min(self.min_curvature, curvature)
        self.max_curvature = max(self.max_curvature, curvature)
        self.total_curvature += curvature
        if (curvature >= 0 and self.prv_cur < 0) or (curvature < 0 and self.prv_cur >= 0):
            self.count_cur_change += 1
        self.prv_cur = curvature
        #News
        if len(self.exp_actions) > self.current_step:
            exp_x = self.ship_trajectories[self.ego_pos][self.current_step+1][0]
            exp_y = self.ship_trajectories[self.ego_pos][self.current_step+1][1]
            self.totalADE += (new_x - exp_x)**2 + (new_y - exp_y)**2
            self.totalAAE += np.abs(acceleration - self.exp_acclerations[self.current_step])
            self.totalASE += np.abs(dheading_scaled - self.exp_actions[self.current_step][2]*(np.pi / 2))
            self.stop_step += 1   
                  

        if self.CPD < self.nearmiss_threshold: #555 meters mean nearmiss
            self.nearmiss_count += 1

        # Increment step counter
        self.current_step += 1

        # Determine if the episode is done or truncated
        done = self._check_goal_reached()  # Implement this method to check if the goal is reached
        dis_to_goal = self.compute_dis_to_goal()
        truncated = not within_region or self.current_step >= self.max_steps
        terminated = done 

        # Get new observation
        observation = self._get_observation()  #This function also change self.current_obs      

        # Calculate reward
        reward = 0
        if self.normalize_xy:
            # Calculate reward in normalized space
            norm_previous_x = (previous_state['x'] / self.max_x) * 2 - 1
            norm_previous_y = (previous_state['y'] / self.max_y) * 2 - 1
            norm_new_x = (new_x / self.max_x) * 2 - 1
            norm_new_y = (new_y / self.max_y) * 2 - 1
            # reward = -np.sqrt((norm_new_x - norm_previous_x)**2 + (norm_new_y - norm_previous_y)**2)
        else:
            # Calculate reward in original space
            pass
            # reward = -np.sqrt((new_x - previous_state['x'])**2 + (new_y - previous_state['y'])**2)

        # Apply a penalty if the agent moves out of bounds
        if truncated:
            reward = -100  # You can adjust this penalty as needed
        if done:
            reward = 1
            
        # Calculate additional metrics                       
        return_info = {'within_region': within_region, 'CPD': self.CPD, "dis_to_goal": dis_to_goal}
        #Reward features:
        goal_feature = -dis_to_goal
        if truncated:
            goal_feature = -10000
        if done:
            goal_feature = 1
        
        nearmiss_feature = 0
        if self.CPD < self.nearmiss_threshold: #555 meters mean nearmiss
            nearmiss_feature = -1
                
        acceleration_feature = -acceleration
        steer_feature = -new_speed*dheading_scaled
        return_info.update({
            'goal_feature': goal_feature,
            'nearmiss_feature': nearmiss_feature,
            'acceleration_feature': acceleration_feature,
            'steer_feature': steer_feature
        })        
        if terminated or truncated:
            avg_nbneigh = self.total_neigh / (self.current_step+1) 
            avg_speed = self.total_speed / self.current_step
            avg_acceleration = self.total_acceleration / self.current_step
            avg_drift = self.total_drift/self.current_step
            avg_curvature = self.total_curvature / self.current_step
            nearmiss_rate = self.nearmiss_count / self.current_step            
            self.current_curv = curvature
            curv_change_rate = self.count_cur_change/self.current_step
            #News:
            gc_ade = np.sqrt(self.totalADE)/self.stop_step
            mae_steer = self.totalASE/self.stop_step
            mae_accel = self.totalAAE/self.stop_step
            return_info.update({        
            'reach_goal': terminated,                   
            'avg_speed': avg_speed,
            'min_speed': self.min_speed,
            'max_speed': self.max_speed,
            'avg_acceleration': avg_acceleration,
            'min_acceleration': self.min_acceleration,
            'max_acceleration': self.max_acceleration,
            'avg_curvature': avg_curvature,
            'min_curvature': self.min_curvature,
            'max_curvature': self.max_curvature,
            'nearmiss_rate': nearmiss_rate*100, #in percentage
            "gc_ade": gc_ade,
            "mae_steer": mae_steer,
            "mae_accel": mae_accel,
            "curv_change_rate": curv_change_rate*100,
            "min_drift": self.min_drift,
            "max_drift": self.max_drift,
            "avg_drift": avg_drift,
            "eps_length": self.current_step,
            "avg_neigh": avg_nbneigh,
            "max_neigh": self.max_neigh
            })
        observation = observation.copy()        
        if self.drop_neighbor:
            observation.pop("neighbors", None)
        if self.use_dis_fea is False:
            observation.pop('nearest_dis', None)
        if self.use_FoR is True:                
            observation['neighbors'] = observation['FoR_neigh']
        if self.use_time_fea:
            observation.update({'time': utils.encode_unix_time(new_timestamp)})
        observation.pop('FoR_neigh', None)
        return observation, float(reward), terminated, truncated, return_info
    
    def _get_observation(self):
        # self.ego_trajectory = self.ship_trajectories[self.ego_pos]
        # Get current timestamp from ego trajectory
        current_timestamp = self.current_state['timestamp']

        # ego_obs = self._get_agent_observation(self.ego_trajectory, current_timestamp)
        ego_obs = self.current_obs['ego'].copy()        
        ego_obs = np.delete(ego_obs, 0, axis=0)#Remove the first element for consistent history length          
        ego_obs = np.append(ego_obs, [self.padded_val], axis=0)        
                
        #Change ego_obs based on the current_state        
        ego_obs[-1][0] = self.current_state['x']
        ego_obs[-1][1] = self.current_state['y']
        ego_obs[-1][2] = self.current_state['speed']
        ego_obs[-1][3] = self.current_state['heading']                
        if self.use_drift_fea:            
            ego_obs[-1][4] = 0
            if np.array_equal(self.current_obs['ego'][-1], self.padded_val) is False:
                past_x = self.current_obs['ego'][-1][0]
                past_y = self.current_obs['ego'][-1][1]
                ego_obs[-1][4] = utils.calculate_drift_two_points(past_x, past_y, ego_obs[-1][0], ego_obs[-1][1], ego_obs[-1][3])
                                            
        self.current_obs['ego']= ego_obs
        # Get current ego position
        ego_x, ego_y = ego_obs[-1][:2]  # Use the most recent observation       

        # Calculate distances to all other agents at the current timestamp
        distances = []
        for agent_id in self.overlap_idx[self.ego_pos]:
            if agent_id != self.ego_pos:
                trajectory = self.ship_trajectories[agent_id]
                time = self.ship_times[agent_id]
                agent_obs = self._get_agent_observation(trajectory, time, current_timestamp)
                if np.array_equal(agent_obs[-1], self.padded_val) is False:  # Check if valid observation (not padded)
                    agent_x, agent_y = agent_obs[-1][:2]
                    distance = np.sqrt((ego_x - agent_x)**2 + (ego_y - agent_y)**2)
                    distances.append((agent_id, distance))
        
        self.total_neigh += len(distances)
        self.max_neigh = max(self.max_neigh, len(distances))
        #Process exp obs:
        self.expert_obs = self._get_agent_observation(self.ship_trajectories[self.ego_pos], self.ship_times[self.ego_pos], current_timestamp)
        self.save_past_exp.appendleft(self.expert_obs[-1])
        # Sort by distance and get the nearest n_neighbor_agents
        distances.sort(key=lambda x: x[1])
        nearest_neighbors = distances[:self.n_neighbor_agents]
        
        if (len(distances) != 0):
            self.CPD = distances[0][1]
        else:
            self.CPD = 0
        neighbor_obs = []
        nearest_dis_obs = []
        for agent_id, dis in nearest_neighbors:
            neighbor_obs.append(self._get_agent_observation(self.ship_trajectories[agent_id], 
                                                            self.ship_times[agent_id],
                                                            current_timestamp))      
            nearest_dis_obs.append(dis)
        # Pad neighbor observations if necessary
        while len(neighbor_obs) < self.n_neighbor_agents:
            neighbor_obs.append(self._get_padded_observation())
            nearest_dis_obs.append(np.sqrt(self.max_x**2+self.max_y**2))

        # for val in neighbor_obs:
        #     print(val.dtype, val.shape)
        self.current_obs['neighbors'] = np.array(neighbor_obs)            
        if self.use_FoR:            
            self.current_obs["FoR_neigh"] = utils.transform_neighbors_to_ego_frame(self.current_obs['ego'], self.current_obs['neighbors'])                               
        self.current_obs['nearest_dis'] = np.array(nearest_dis_obs, dtype=np.float32)
        # self.current_obs['goal'] = np.array([ego_goal_x, ego_goal_y])
        return self.current_obs    

    def _get_agent_observation(self, trajectory, time, target_timestamp):
        obs = []
        # Find the index of the target timestamp or the nearest earlier timestamp        
        low_t = target_timestamp - self.observation_history_length*self.second_perts             
        lst_ids = utils.find_indices_in_range_sorted(time, 
                                                     low_t,
                                                     target_timestamp)
        if(len(lst_ids) != 0):
            # print(int((trajectory[lst_ids[0]][4] - low_t)/self.second_perts))
            for i in np.arange(0, (time[lst_ids[0]] - low_t)/self.second_perts, dtype=int):
                obs.append(self.padded_val)
                
            for i in lst_ids:       
                if self.use_drift_fea is False:         
                    x, y, speed, heading = trajectory[i]
                else: 
                    x, y, speed, heading, drift = trajectory[i]
                timestamp = time[i]
                if self.normalize_xy:
                    x = (x / self.max_x) * 2 - 1
                    y = (y / self.max_y) * 2 - 1
                if self.use_drift_fea is False:    
                    obs.append(np.array([x, y, speed, heading]))            
                else:
                    obs.append(np.array([x, y, speed, heading, drift]))
                        
        # Pad with -1 if we don't have enough history
        while len(obs) < self.observation_history_length + 1:
            obs.append(self.padded_val)        
        return np.array(obs)
    
    def _get_padded_observation(self):
        return [self.padded_val] * (self.observation_history_length + 1)
    
    def infer_action_from_trajectory(self, timestep):       
        current_state = self.ship_trajectories[self.ego_pos][timestep] 
        next_state = self.ship_trajectories[self.ego_pos][timestep + 1]

        dx = next_state[0] - current_state[0]
        dy = next_state[1] - current_state[1]
        dheading = (next_state[3] - current_state[3] + np.pi) % (2 * np.pi) - np.pi  # Ensure shortest path
        self.max_dx = max(self.max_dx, abs(dx))
        if abs(dx) > 150:
            self.count_dx.append(abs(dx))
        self.max_dy = max(self.max_dy, abs(dy))
        if timestep == 0:
            new_speed = np.sqrt(dx**2 + dy**2)/self.second_perts  # Example: speed as magnitude of movement            
            self.exp_acclerations.append((new_speed - current_state[2])/self.second_perts)
        else:
            new_speed = np.sqrt(dx**2 + dy**2)/self.second_perts  # Example: speed as magnitude of movement            
            prv_state = self.ship_trajectories[self.ego_pos][timestep-1]
            dx_old = current_state[0] - prv_state[0]
            dy_old = current_state[1] - prv_state[1]
            old_speed = np.sqrt(dx_old**2 + dy_old**2)/self.second_perts  # Example: speed as magnitude of movement   
            self.exp_acclerations.append((new_speed - old_speed)/self.second_perts)         
        
        # assert(abs(dheading) <= np.pi/2)       
        # assert(abs(dx) <= (self.max_x / 20))        
        # assert(abs(dy) <=  (self.max_y / 20))
        if abs(dheading) > np.pi/2 or abs(dx) > (self.max_x / 20) or abs(dy) >= (self.max_y / 20):
            return None #Irregular action case return None for ignoring this scenario later
                        
        if self.normalize_xy:
            dx = dx / (self.max_x / 2)
            dy = dy / (self.max_y / 2)        
        elif self.scale_act:
            dx = dx / (self.max_x / 20)
            dy = dy / (self.max_y / 20)
            dheading = dheading / (np.pi / 2)  # Scale back to [-1, 1] range              
        # Clip actions to ensure they're in the [-1, 1] range
        if self.scale_act is False:
            action = np.array([dx, dy, dheading])        
        else:
            action = np.clip([dx, dy, dheading], -1, 1)
        # if action[0] == -1:
        #     print(self.max_x)
        #     print(next_state[0] - current_state[0])
        #     print(next_state[1] - current_state[1])
        #     print(current_state[2], next_state[2])
        #     assert 1 == -1

        return action
    
    def gen_actions_from_data(self):
        exp_traj_length = len(self.ship_trajectories[self.ego_pos]) - 1
        res_lst = []
        for i in range(exp_traj_length):
            val_a = self.infer_action_from_trajectory(i)
            if val_a is None:
                return None
            res_lst.append(val_a)
        return np.array(res_lst, dtype=np.float32)

TIME_COL = "TIMESTAMP_UTC"
"""Convert geographic coordinates to local Cartesian coordinates"""
def lon_to_xpos(lon, origin_lon, origin_lat):
    return utils.haversine_distance(origin_lon, origin_lat, lon, origin_lat)    

def lat_to_ypos(lat, origin_lon, origin_lat):
    return utils.haversine_distance(origin_lon, origin_lat, origin_lon, lat)

def knots_to_ms(speed_knots):
    """Convert speed from 10xknots to meters per second"""
    return speed_knots * 0.0514444

def degrees_to_radians(degrees):
    """Convert angles from degrees to radians"""
    return np.radians(degrees)

def heading_to_2d_radians(heading_degrees):
    # Convert to radians
    heading_radians = np.radians(heading_degrees)
    
    # Adjust for coordinate system difference
    adjusted_radians = -1 * (heading_radians - np.pi/2)
    
    # Normalize to range [0, 2π)
    normalized_radians = adjusted_radians % (2 * np.pi)
    
    return normalized_radians

def radians_2d_to_heading(radians_2d):
    
    # Reverse the coordinate system adjustment
    heading_radians = -1 * radians_2d + np.pi/2
    
    # Convert to degrees
    heading_degrees = np.degrees(heading_radians)
    
    # Normalize to range [0, 360)
    normalized_degrees = heading_degrees % 360
    
    return normalized_degrees

def chunk_to_traj(chunk: pl.DataFrame, interpol_interval: timedelta, region_of_interest_array: np.ndarray):
    # Ensure interpol_interval is in integer seconds
    interpol_seconds = int(interpol_interval.total_seconds())
    if interpol_seconds != interpol_interval.total_seconds():
        raise ValueError("interpol_interval must be in integer seconds")

    # Extract time, latitude, and longitude from the 'all_records' column
    # and create a new DataFrame with only these columns
    df = chunk.select([
        pl.col('all_records').list.eval(pl.element().struct.field(TIME_COL)).alias('time'),
        pl.col('all_records').list.eval(pl.element().struct.field('LAT')).alias('lat'),
        pl.col('all_records').list.eval(pl.element().struct.field('LON')).alias('lon'),
        pl.col('all_records').list.eval(pl.element().struct.field('SPEED_KNOTSX10')).alias('speed'),
        pl.col('all_records').list.eval(pl.element().struct.field('HEADING')).alias('heading'),
    ]).explode(['time', 'lat', 'lon', 'speed', 'heading'])     
    # Find the start and end times, rounded to the nearest interpol_interval
    # start_time will uses ceil and end_time uses floor due to interpolation
    start_time = df['time'].min().replace(microsecond=0)
    start_time = start_time + timedelta(seconds=interpol_seconds - start_time.second % interpol_seconds)            
    end_time = df['time'].max().replace(microsecond=0)
    end_time = end_time - timedelta(seconds=end_time.second % interpol_seconds)    

    # Convert times to seconds since epoch (integer)
    original_time = df['time'].map_elements(lambda x: int(x.timestamp()), return_dtype=pl.Int64)        

    # Define the origin as the southwest corner of the region of interest
    origin_lon, origin_lat = region_of_interest_array[0][0], region_of_interest_array[0][1]
    
    # Convert to numpy arrays for interpolation
    original_time_np = original_time.to_numpy()
    y_np = np.apply_along_axis(lambda x: lat_to_ypos(x, origin_lon, origin_lat), 0, df['lat'].to_numpy())
    x_np = np.apply_along_axis(lambda x: lon_to_xpos(x, origin_lon, origin_lat), 0, df['lon'].to_numpy())
    speed_np = np.apply_along_axis(knots_to_ms, 0, df['speed'].to_numpy())
    heading_np = np.apply_along_axis(heading_to_2d_radians, 0, df['heading'].to_numpy())    
    t_np = np.arange(start_time.timestamp(), end_time.timestamp() + interpol_seconds, interpol_seconds, dtype=np.int64)
    traj_y = np.interp(t_np, original_time_np, y_np)    
    traj_x = np.interp(t_np, original_time_np, x_np)            
    traj_speed = np.interp(t_np, original_time_np, speed_np)
    traj_heading = utils.shortest_path_angle_interp(t_np, original_time_np, heading_np)
    traj = np.column_stack((traj_x, traj_y, traj_speed, traj_heading))    
    traj = traj.astype(np.float32)    
    return traj, t_np

def find_max_overlap_with_ego(df: pl.DataFrame, ego_id: int, n: int) -> pl.DataFrame:
    # Ensure ego_id exists in the DataFrame
    if ego_id not in df['SHIP_ID'].to_list():
        raise ValueError(f"ego_id {ego_id} not found in the DataFrame")

    # Get the ego ship's interval
    ego_row = df.filter(pl.col('SHIP_ID') == ego_id)
    ego_start = ego_row['start_time'][0]
    ego_end = ego_row['end_time'][0]

    # Calculate overlap for each row
    df_with_overlap = df.with_columns(
        overlap = (
            pl.when(pl.col('end_time') <= ego_start)
            .then(0)
            .when(pl.col('start_time') >= ego_end)
            .then(0)
            .otherwise(
                pl.when(pl.col('start_time') <= ego_start)
                .then(
                    pl.when(pl.col('end_time') <= ego_end)
                    .then(pl.col('end_time') - ego_start)
                    .otherwise(ego_end - ego_start)
                )
                .otherwise(
                    pl.when(pl.col('end_time') <= ego_end)
                    .then(pl.col('end_time') - pl.col('start_time'))
                    .otherwise(ego_end - pl.col('start_time'))
                )
            )
        )
    )

    # Sort by overlap (descending) and SHIP_ID (to ensure ego_id is first if tied)
    df_sorted = df_with_overlap.sort(['overlap', 'SHIP_ID'], descending=[True, False])

    # Get top n rows
    result = df_sorted.head(n)

    # Ensure ego_id is in the result
    if ego_id not in result['SHIP_ID'].to_list():
        result = pl.concat([
            ego_row,
            result.filter(pl.col('SHIP_ID') != ego_id).head(n - 1)
        ])
    elif result['SHIP_ID'][0] != ego_id:
        ego_index = result['SHIP_ID'].to_list().index(ego_id)
        result = pl.concat([
            result.slice(1, 1),  # New first row (previously top overlap)
            result.slice(0, 1),  # New second row (ego_id)
            result.slice(2, ego_index - 1) if ego_index > 1 else None,  # Rows between if any
            result.slice(ego_index + 1, None)  # Remaining rows
        ])

    return result

# Function to read static data (vessel information)
def read_static_data(file_pattern):
    return (pl.scan_csv(file_pattern, 
                        separator=";", 
                        null_values=["","NULL"])
            .select(["SHIP_ID", "VESSEL_TYPE"])
            .unique(subset=["SHIP_ID"])  # Remove duplicates based on SHIP_ID
            .collect())

# Function to read position data
def read_position_data(file_pattern):
    return (pl.scan_csv(file_pattern, 
                        separator=";", 
                        null_values=["","NULL"], 
                        try_parse_dates=True)
            .unique(subset=["SHIP_ID", "TIMESTAMP_UTC"])  # Remove duplicates based on SHIP_ID and TIMESTAMP_UTC
            .collect())


def create_minari_dataset(env, dataset_name, num_ships):
    env = minari.DataCollector(env, action_space=env.action_space, observation_space=env.observation_space)         
    for id_ego in range(num_ships):   
        print(id_ego) 
        obs, info = env.reset(seed=42, options = {'ego_pos': id_ego})
        # print(obs['time'])
        actions = info['actions']
        if actions is None: #Irregular action case
            continue
        for i in range(1000):        
            action = actions[i]
            observation, reward, terminated, truncated, info = env.step(action)                         
            if terminated or truncated:          
                break
    dataset = env.create_dataset(dataset_id=dataset_name,                                                            
                                author="qapham",
                                author_email="lovek62uet@gmail.com")

