import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import logging
import warnings
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.offline as pyo
import plotly.graph_objs as go
import seaborn as sns
import gc
from sklearn.metrics import confusion_matrix
import matplotlib.animation as animation
import random
import torch
import time
import plotly.express as px
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist
from multiprocessing import Process, Manager
import gymnasium as gym
from gymnasium.utils import seeding
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.envs import SimpleMultiObsEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import BasePolicy
import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, MlpExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import torch.distributions as distributions
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import get_device
from scipy.fft import fft

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class CustomLSTMFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 256, num_layers: int =3, bidirectional: bool = False):
        super(CustomLSTMFeaturesExtractor, self).__init__(observation_space, features_dim)
        self._features_dim = features_dim

        original_dim = observation_space.shape[0]  # Original observation space
        self.reshape_layer = nn.Linear(original_dim, 256)  # Reshaping layer

        hidden_dim = 128  

        self.lstm = nn.LSTM(input_size=256, hidden_size=hidden_dim,  # Now input_size is 256
                            num_layers=num_layers, batch_first=True,
                            bidirectional=bidirectional)

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.projection = nn.Linear(lstm_output_dim, features_dim) 

    @property
    def features_dim(self):
        return self._features_dim

    @features_dim.setter
    def features_dim(self, value):
        self._features_dim = value

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = self.reshape_layer(observations)
        # Ensure observations are 3D: (batch_size, seq_len, input_dim)
        if observations.ndim == 2:
            observations = observations.unsqueeze(0)  # Add batch_size dimension

        # Pad the tensor with zeros to match the expected size
        target_size = self._features_dim  # Expected size of the last dimension
        current_size = observations.shape[-1]  # Current size of the last dimension
        if current_size < target_size:
            padding = (0, target_size - current_size)
            observations = nn.functional.pad(observations, padding, "constant", 0)

        hidden, _ = self.lstm(observations)  # LSTM output (batch_size, seq_len, hidden_dim)

        # If LSTM is bidirectional, concatenate the hidden states from both directions
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[:, :, :self.lstm.hidden_size], hidden[:, :, self.lstm.hidden_size:]), dim=2)
        else:
            hidden = hidden[:, -1, :]  # Take the last hidden state

        projected = self.projection(hidden)  # Project dimensions to match expected input

        return projected

    def get_features_dim(self) -> int:
        return self._features_dim



class CustomLSTMPolicy(BasePolicy):
    def __init__(self, observation_space, action_space, lr_schedule, vec_input_dim, features_dim=256):
        # Adjust the shape of observation_space to 256
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(256,), dtype=np.float32)
        super(CustomLSTMPolicy, self).__init__(observation_space, action_space, lr_schedule)
        self.vec_input_dim = vec_input_dim
        self.features_dim = features_dim

        # Define the feature extractor
        self.features_extractor = nn.Sequential(
            nn.Linear(self.vec_input_dim, self.features_dim),
            nn.ReLU()
        )

        # Define the MLP extractor
        self.mlp_extractor = nn.Sequential(
            nn.Linear(self.features_dim, self.features_dim),
            nn.ReLU()
        )

        # Define the action network
        self.action_net = nn.Linear(self.features_dim, self.action_space.n)

        # Define the value network
        self.value_net = nn.Linear(self.features_dim, 1)

    def forward(self, obs, deterministic=False):
        features = self.features_extractor(obs)
        policy_features = self.mlp_extractor(features)

        logits = self.action_net(policy_features)
        value = self.value_net(policy_features)

        if logits is None:
            print("Warning: logits is None")
            logits = torch.ones_like(policy_features)
        if value is None:
            print("Warning: value is None")
            value = torch.ones_like(policy_features)

        # Compute the action distribution and sample an action
        action_distribution = distributions.Categorical(logits=logits)
        action = action_distribution.sample()

        # Compute the log-probability of the action if needed
        log_prob = action_distribution.log_prob(action)

        if log_prob is None:
            print("Warning: log_prob is None")
            log_prob = torch.ones_like(action)

        return action, value, log_prob

    def get_action(self, obs, deterministic=False):
            dist = self.action_distribution(self.forward(obs))
            if deterministic:
                action = dist.probs.argmax(dim=1)
            else:
                action = dist.sample()
            return action

    def evaluate_actions(self, obs, actions):
        features = self.features_extractor(obs)
        policy_features = self.mlp_extractor(features)
        logits = self.action_net(policy_features)
        value = self.value_net(policy_features)

        action_distribution = distributions.Categorical(logits=logits)
        log_prob = action_distribution.log_prob(actions)

        return value, log_prob


    



class MlpExtractor(nn.Module):
    def __init__(self, input_dim, net_arch, activation_fn):
        super(MlpExtractor, self).__init__()
        self.input_dim = input_dim
        self.net_arch = net_arch  # Store net_arch
        self.activation_fn = activation_fn  # Store activation_fn
        
        # Create a list of layers using net_arch
        self.layers = nn.ModuleList()
        last_layer_dim = input_dim
        for layer in net_arch:
            self.layers.append(nn.Linear(last_layer_dim, layer))
            last_layer_dim = layer

             # Print the initialization parameters
            #print("Input dimension:", input_dim)
           # print("Network architecture:", net_arch)
            #print("Activation function:", activation_fn)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        for layer in self.layers:
            weight_t = layer.weight.data  # Get the weight tensor

            # Calculate the padding size
            padding_size = weight_t.shape[1] - x.shape[1]
            
            # If padding is needed, pad the input tensor
            if padding_size > 0:
                x = F.pad(x, (0, padding_size))

            x = F.linear(x, weight_t, layer.bias)
        
        # Only return the first two outputs
        return x[:2]
    

    def forward_critic(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        for layer in self.layers:
            #print("Shape of weight  critic tensor:", layer.weight.shape)  # Print the shape of the weight tensor
            #print("Shape of bias critic tensor:", layer.bias.shape)  # Print the shape of the bias tensor
           # print("Shape of x critic before layer:", x.shape) 

            # Transpose the weight tensor
            weight_t = torch.transpose(layer.weight, 0, 1)

            # Perform the linear transformation with the transposed weight tensor
            x = F.linear(x, weight_t, layer.bias)

        return x


    def forward_actor(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        for layer in self.layers:
            weight_t = layer.weight.data  # Get the weight tensor

            # Calculate the padding size
            padding_size = weight_t.shape[1] - x.shape[1]
            
            # If padding is needed, pad the input tensor
            if padding_size > 0:
                x = F.pad(x, (0, padding_size))

            x = F.linear(x, weight_t, layer.bias)
        return x




class CustomLSTMPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space=2, lr_schedule=0.005, *args, **kwargs):
        kwargs.pop('vec_input_dim', None)
        kwargs.pop('features_dim', None)
        
        # Adjust the shape of observation_space to 256
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(256,), dtype=np.float32)
        super(CustomLSTMPolicy, self).__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        
        # Initialize features extractor before referring to it
        self.features_extractor = CustomLSTMFeaturesExtractor(observation_space)
        self.mlp_extractor = MlpExtractor(input_dim=256, net_arch=[256, 256], activation_fn=nn.ReLU)

        # Set input_size to the dimensionality of the extracted features
        input_size = self.features_extractor.features_dim

        # Initialize LSTM with the correct input_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=256)

        self.action_net = nn.Linear(256, 2)
        self.value_net = nn.Linear(256, 1)

        # Policy and value networks
        self.policy_net = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

        self.value_net = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)

        if features.ndim == 4:
            features = features.view(features.size(0), features.size(1), -1)

        lstm_output, _ = self.lstm(features)

        policy_logits = self.policy_net(lstm_output)
        value = self.value_net(lstm_output)
        value = value.squeeze(-1)  # Ensure the value output has the correct shape

        action_distribution = distributions.Categorical(logits=policy_logits)
        action = action_distribution.sample()

        log_prob = action_distribution.log_prob(action)

        return action, value, log_prob

    def extract_features(self, obs: torch.Tensor) -> torch.Tensor:
        try:
            # Reshape the input tensor to match the expected shape
            obs = obs.view(obs.size(0), -1, self.observation_space.shape[0])
            return self.features_extractor(obs)  
        except Exception as e:
            print(f"Error in extract_features: {e}")
            traceback.print_exc()

    def _get_latent(self, obs: torch.Tensor):
        features = self.extract_features(obs)
        outputs = self.mlp_extractor(features)

        # If outputs is a tensor of size 256, reshape it to (2, 128)
        if outputs.size(0) == 256:
            outputs = outputs.view(2, -1)

        latent_pi, latent_vf = outputs  
        return latent_pi, latent_vf

    def get_action(self, obs, deterministic=False):
        dist = self.action_distribution(self.forward(obs))
        if deterministic:
            action = dist.probs.argmax(dim=1)
        else:
            action = dist.sample()
        return action
    




class CustomPPO(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add a fully connected layer to reduce the size from 256 to 1
        self.fc = nn.Linear(256, 1)

    def train(self):
        """
        Update policy using the currently gathered rollout buffer.
        This is a modified version of the original train method to address the dimension mismatch.
        """
        
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses, all_kl_divs, value_losses, policy_losses, clip_fractions = [], [], [], [], []
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(batch_size=self.batch_size):

                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                # Compute actor and critic values
                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)

                # Avoid NaNs
                values = values.view(-1, 1)  # Ensure 'values' has shape [256, 1]

                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Policy loss
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Value loss using the same clipping as in the PPO paper
                if self.clip_range_vf is None:
                    # Unclipped value loss
                    returns = self.fc(rollout_data.returns.view(-1, 256))  # Ensure 'returns' has shape [1, 1]
                    value_loss = F.mse_loss(returns, values)
                else:
                    # Clipped value loss
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                    returns = self.fc(rollout_data.returns.view(-1, 256))  # Ensure 'returns' has shape [1, 1]
                    values_pred = values_pred.view(-1, 1)  # Ensure 'values_pred' has shape [256, 1]
                    value_loss = F.mse_loss(returns, values_pred)

                # Entropy loss favor exploration
                if entropy is None:
                    entropy_loss = -torch.mean(log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Clip gradient norm
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

                self.policy.optimizer.zero_grad()
                loss.backward()
                self.policy.optimizer.step()

                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                value_losses.append(value_loss.item())
                policy_losses.append(policy_loss.item())
                entropy_losses.append(entropy_loss.item())
                approx_kl_divs.append(torch.mean(rollout_data.old_log_prob - log_prob).detach().cpu().numpy())

            all_kl_divs.append(np.mean(approx_kl_divs))

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/policy_loss", np.mean(policy_losses))
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/approx_kl", np.mean(all_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", np.mean(value_losses) + np.mean(policy_losses) + np.mean(entropy_losses))
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

        self.logger.record("time/total_timesteps", self.num_timesteps)
        self.logger.record("time/time_elapsed", int(time.time() - self.start_time))
        self.logger.record("time/iterations", self._n_updates)

        self.logger.dump(step=self.num_timesteps)






class AnomalyDetectionEnv(gym.Env):
    def __init__(self, data, window_size, threshold, reward_system='default'):
        super(AnomalyDetectionEnv, self).__init__()
        self.data = data
        self.window_size = window_size
        self.threshold = threshold
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(256,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.current_step = 0
        self.recent_rewards = []
        self.reward_system = reward_system
        self.previous_mistakes = 0

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.recent_rewards = []
        obs = self._get_observation()
        info = {'current_step': self.current_step}
        return obs, info

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.data) - self.window_size
        obs = self._get_observation()
        reward = self.compute_reward(action)
        info = {'current_step': self.current_step}
        return obs, reward, done, False, info

    
    def compute_reward(self, action):
        actual_value = self.data.iloc[self.current_step]
        is_anomaly = abs(actual_value) > self.threshold
        predicted_label = action  # Assuming action 1 means predicting an anomaly

        previous_mistakes = sum(self.recent_rewards)

        # Binary Rewards
        if self.reward_system == 'default':
            if is_anomaly == (predicted_label == 1):
                reward = 1  # Correct prediction
            else:
                reward = 0.1 if predicted_label == 0 else 0  # Incorrect prediction

        # Active Learning Rewards
        elif self.reward_system == 'alternative':
            reward = 0
            if is_anomaly == (predicted_label == 1):
                reward = 1  # Correct prediction
            if is_anomaly != (predicted_label == 1) and previous_mistakes > 0:
                reward += 0.5  # Bonus reward for learning from mistakes

        self.recent_rewards.append(reward)
        return reward
    


    def _get_observation(self):
        return np.resize(self.data[self.current_step:self.current_step + self.window_size], (256,))



    def _next_observation(self):
        obs = self.data[self.current_step:self.current_step + self.window_size]
        moving_avg = obs.rolling(window=self.window_size).mean()
        rsi = self._calculate_rsi(obs, self.window_size)

        obs = np.nan_to_num(obs, nan=np.nanmean(obs))
        moving_avg = np.nan_to_num(moving_avg, nan=np.nanmean(moving_avg))
        rsi = np.nan_to_num(rsi, nan=np.nanmean(rsi))

        important_features_obs = obs[:self.window_size // 3]
        important_features_moving_avg = moving_avg[:self.window_size // 3]
        important_features_rsi = rsi[:self.window_size // 3]

        combined_obs = np.concatenate([important_features_obs, important_features_moving_avg, important_features_rsi])
        combined_obs = combined_obs.flatten()

        if len(combined_obs) < 256:
            combined_obs = np.pad(combined_obs, (0, 256 - len(combined_obs)))
        elif len(combined_obs) > 256:
            combined_obs = combined_obs[:256]

        return combined_obs

    def _calculate_rsi(self, data, window_size):
        diff = data.diff()
        up = diff.clip(lower=0)
        down = -1 * diff.clip(upper=0)
        ema_up = up.ewm(com=window_size - 1, adjust=False).mean()
        ema_down = down.ewm(com=window_size - 1, adjust=False).mean()

        epsilon = 1e-10
        rs = ema_up / (ema_down + epsilon)

        rsi = 100 - (100 / (1 + rs))
        return rsi



    

class CustomDummyVecEnv(gym.Env):
    def __init__(self, envs):
        super(CustomDummyVecEnv, self).__init__()
        self.envs = envs
        self.num_envs = len(envs)
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(256,), dtype=np.float32)

    def reset(self, seed=None):
        observations = []
        self.reset_infos = []
        for env_idx, env in enumerate(self.envs):
            obs, info = env.reset()
         
            observations.append(self._process_obs(obs))
            self.reset_infos.append(info)
        return np.array(observations), self.reset_infos

    def step(self, actions):
        observations, rewards, dones, infos = [], [], [], []
        for env, action in zip(self.envs, actions):
            obs, rew, done, truncated, info = env.step(action)  # Expecting five return values 
            observations.append(self._process_obs(obs))
            rewards.append(rew)
            dones.append(done)
            infos.append(info)
        return np.array(observations), np.array(rewards), np.array(dones), np.array([False]*self.num_envs), infos  # Include truncated as array of False

    def _process_obs(self, obs):
        if isinstance(obs, dict):
            obs = np.concatenate([obs[key].flatten() for key in sorted(obs.keys())])
        elif isinstance(obs, str):
            obs = np.zeros_like(self.observation_space.low, dtype=self.observation_space.dtype)
        else:
            obs = np.array(obs).flatten()

        if len(obs) < 256:
            obs = np.pad(obs, (0, 256 - len(obs)))
        elif len(obs) > 256:
            obs = obs[:256]

        if obs.shape != (256,):
            raise ValueError(f"Observation shape is {obs.shape}, but expected (256,)")

        return obs

    def _save_obs(self, env_idx, obs):
        obs = np.array(obs).flatten()
        if obs.shape != (256,):
            raise ValueError(f"Observation shape for env {env_idx} is {obs.shape}, but expected (256,)")
        self.buf_obs[env_idx] = obs

    def _obs_from_buf(self):
        return np.copy(self.buf_obs)






def load_and_preprocess_data(file_path, sheet_name):
    # Read the Excel file
    dataframe = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

    # Find the row index where 'TIME' is located
    time_index = dataframe[dataframe.eq('TIME').any(axis=1)].index.tolist()[0]

    # Find the row index where 'CURRENCY (Labels)' is located
    currency_index = dataframe[dataframe.eq('CURRENCY (Labels)').any(axis=1)].index.tolist()[0]

    # Set the row after 'TIME' as the column headers
    dataframe.columns = dataframe.iloc[time_index]

    # Add a prefix to the column names based on the sheet name
    dataframe.columns = [f'{sheet_name}_{col}' for col in dataframe.columns]

    # Drop the rows before the currency labels
    dataframe = dataframe.iloc[currency_index:]

    # Set the first column as the index and remove the index name
    dataframe.set_index(dataframe.columns[0], inplace=True)
    dataframe.index.name = None

    # Remove the 'CURRENCY (Labels)' row if it exists
    dataframe = dataframe.drop('CURRENCY (Labels)', errors='ignore')

    # Remove the unwanted row
    dataframe = dataframe.drop(8, errors='ignore')

    # Convert columns to numeric
    dataframe = dataframe.apply(pd.to_numeric, errors='coerce')

    # Forward fill NaN values
    dataframe = dataframe.ffill()

    # Backward fill any remaining NaN values
    dataframe = dataframe.bfill()

    # Interpolate any remaining NaN values
    dataframe = dataframe.interpolate()

    # Replace NaN values with the mean of the column
    dataframe.fillna(dataframe.mean(), inplace=True)

    # Apply differencing to take into account the temporal structure of the data
    dataframe = dataframe.diff()

    # Replace any NaN values that might have been introduced by differencing
    dataframe.fillna(dataframe.mean(), inplace=True)


    # Maintain the DataFrame structure
    dataframe = pd.DataFrame(dataframe, columns=dataframe.columns, index=dataframe.index)

    if np.isinf(dataframe.values).any():
        print("Infinite values found. Replacing with NaN.")
        dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Replace NaN values with the column mean
    if dataframe.isnull().values.any():
        print("NaN values found. Replacing with mean.")
        # Calculate mean for each column without NaN
        means = dataframe.mean(skipna=True)
        # Replace NaN with the mean of each column
        dataframe.fillna(means, inplace=True)


     # Create anomaly labels based on a threshold (e.g., values beyond 3 standard deviations are anomalies)
    #threshold = 0.9
    #dataframe['anomaly'] = ((dataframe - dataframe.mean()).abs() > threshold * dataframe.std()).any(axis=1).astype(int)

    return dataframe

def create_lagged_features(dataframe, n_lags):
    # Check if dataframe is a DataFrame or a Series
    if isinstance(dataframe, pd.DataFrame):
        # If it's a DataFrame, use multi-dimensional indexing
        original_data = dataframe.iloc[:, 0]
    else:
        # If it's a Series, use single-dimensional indexing
        original_data = dataframe.iloc[:]

    for lag in range(1, n_lags + 1):
        dataframe[f"lag_{lag}"] = original_data.shift(lag)

    # Forward fill NaN values
    dataframe = dataframe.ffill()

    # Backward fill any remaining NaN values
    dataframe = dataframe.bfill()

    return dataframe


def feature_engineering(df):
    df = df.copy()

    # Create a dictionary to store new columns
    new_columns = {}

    # Check if df is a DataFrame or a Series
    if isinstance(df, pd.DataFrame):
        columns = df.columns
    else:  # df is a Series
        df = df.to_frame()  # Convert Series to DataFrame
        columns = df.columns

    for column in columns:
        # Ensure the column data is numeric
        if pd.api.types.is_numeric_dtype(df[column]):
            # Add new columns to the dictionary
            new_columns[f'{column}_RollingMean7'] = df[column].rolling(window=7).mean()
            new_columns[f'{column}_RollingStd7'] = df[column].rolling(window=7).std()
            new_columns[f'{column}_LargeChange'] = (df[column] - df[column].shift(1)).abs() > df[column].std()

    # Convert the dictionary to a DataFrame and concatenate it with the original DataFrame
    df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)

    return df


def create_moving_average(df, window):
    """
    Create a moving average for a given DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame.
    window (int): The window size for the moving average.

    Returns:
    df (pd.DataFrame): The DataFrame with the moving average.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new_columns = {}
        for column in df.columns:
            new_columns[f'{column}_RollingMean{window}'] = df[column].rolling(window=window).mean()
            new_columns[f'{column}_RollingStd{window}'] = df[column].rolling(window=window).std()
            new_columns[f'{column}_LargeChange'] = (df[column] - df[column].shift(1)).abs() > 0.1

        df = df.assign(**new_columns)

    return df


def statistical_features(dataframe):
    stats_features = dataframe.agg(['mean', 'median', 'std', 'skew', 'kurt'])
    return stats_features


def fourier_transform(dataframe):
    trans_signal = dataframe.apply(lambda x: fft(np.array(x)))
    return trans_signal




def print_df(dataframe, message):
    print(message)
    print(dataframe.head())
    print("\n")


def print_dataframe_info(df, df_name):
    if isinstance(df, pd.DataFrame):
        print(f'{df_name} head:')
        print(df.head())
        print(f'{df_name} tail:')
        print(df.tail())
    else:
        print(f'{df_name} is not a DataFrame.')


def merge_dataframes(df1, df2):
    # Reset the index for df1 and df2
    df1.reset_index(inplace=True)
    df2.reset_index(inplace=True)

    # Perform merge using merge() function on the index
    df = pd.merge(df1, df2, on='index')

    # Set appropriate column names as the index
    df.set_index(df.columns[0], inplace=True)

    # Replace ':' and 'Na' with NaN
    df.replace([':', 'Na', 'NaN'], np.nan, inplace=True)

    # Forward fill to handle NaN values
    df.ffill(inplace=True)

    # Fill remaining NaN values with column means
    df.fillna(df.mean(), inplace=True)

    # Remove rows with specific index values
    rows_to_remove = ['NaN', 'Special value', ':']
    df = df[~df.index.isin(rows_to_remove)]

    # Remove rows with NaN index
    df = df[~df.index.isna()]

    # Fill NaN values in the dataset with zero
    df.fillna(value=0, inplace=True)

    # Rename the columns and index
    df.columns.name = 'TIME'
    df.index.name = 'CURRENCY(Labels)'

    return df


def reorder_columns(df):
    # Get the column names
    col_names = df.columns.tolist()

    # Split the column names into two lists based on the sheet name
    sheet1_cols = [col for col in col_names if 'Sheet 1' in col]
    sheet2_cols = [col for col in col_names if 'Sheet 2' in col]

    # Sort the column names within each list
    sheet1_cols.sort()
    sheet2_cols.sort()

    # Combine the sorted lists
    sorted_cols = [None] * (len(sheet1_cols) + len(sheet2_cols))
    sorted_cols[::2] = sheet1_cols
    sorted_cols[1::2] = sheet2_cols

    # Reorder the columns in the dataframe
    df = df[sorted_cols]

    return df




def normalize_dataframe(df):
    # Ensuring the DataFrame contains numeric data
    df = df.select_dtypes(include=[np.number])
    
    # Replace zeros with the mean value of the column
    df.replace(0, df.mean(), inplace=True)
    
    # Apply differencing to take into account the temporal structure of the data
    df = df.diff()
    
    # Drop the first row which is NaN after differencing
    df = df.iloc[1:]
    
    # Check if there are any columns that consist entirely of NaN values
    nan_columns = df.columns[df.isna().all()].tolist()
    
    # If there are any such columns, drop them
    if nan_columns:
        df.drop(nan_columns, axis=1, inplace=True)
    
    # Min-Max scaling to normalize the data between 0 and 1
    if not df.empty:
        df_min = df.min()
        df_max = df.max()
        
        # Avoid division by zero
        df = (df - df_min) / (df_max - df_min + 1e-7)
    
    # Check for inf values
    if np.any(np.isinf(df)):
        print("Infinite values found. Replacing with NaN.")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Check for NaN values again
    if df.isnull().values.any():
        print("NaN values found. Replacing with mean.")
        df.fillna(df.mean(), inplace=True)
    
    # Remove unwanted rows
    rows_to_remove = ['NaN', 'Special value', ':']
    df = df[~df.index.isin(rows_to_remove)]
    
    return df




def validate_observation(obs):
    if not isinstance(obs, dict):
        raise ValueError(f"Observation should be a dictionary, got {type(obs)}")
    for key, value in obs.items():
        if not isinstance(value, np.ndarray):
            raise ValueError(f"Observation value for key '{key}' should be a numpy array, got {type(value)}")
        



def calculate_metrics(df):
    df['ADRate'] = df['tp'] / (df['tp'] + df['fn'])
    df['FARate'] = df['fp'] / (df['fp'] + df['tn']) 

def plot_anomaly_detection_rate(df):
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df, x='currency', y='ADRate', hue='reward_system')
    plt.title('Anomaly Detection Rate per Currency and Reward System')
    plt.xlabel('Currency')
    plt.ylabel('Anomaly Detection Rate (%)')
    plt.legend(title='Reward System')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_false_alarm_rate(df):
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df, x='currency', y='FARate', hue='reward_system')
    plt.title('False Alarm Rate per Currency and Reward System')
    plt.xlabel('Currency')
    plt.ylabel('False Alarm Rate (%)')
    plt.legend(title='Reward System')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_impact_of_threshold_on_anomaly_detection_rate(df):
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x='threshold', y='ADRate', hue='currency', style='reward_system', markers=True, dashes=False)
    plt.title('Impact of Threshold Value on Anomaly Detection Rate')
    plt.xlabel('Threshold')
    plt.ylabel('Anomaly Detection Rate (%)')
    plt.legend(title='Currency / Reward System')
    plt.tight_layout()
    plt.show()

def plot_impact_of_window_size_on_anomaly_detection_rate(df):
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x='window_size', y='ADRate', hue='currency', style='reward_system', markers=True, dashes=False)
    plt.title('Impact of Window Size on Anomaly Detection Rate')
    plt.xlabel('Window Size (months)')
    plt.ylabel('Anomaly Detection Rate (%)')
    plt.legend(title='Currency / Reward System')
    plt.tight_layout()
    plt.show()

def plot_mean_reward(df):
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df, x='currency', y='mean_reward', hue='reward_system')
    plt.title('Mean Reward per Currency and Reward System')
    plt.xlabel('Currency')
    plt.ylabel('Mean Reward')
    plt.legend(title='Reward System')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_precision(df):
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df, x='currency', y='prec', hue='reward_system')
    plt.title('Precision per Currency and Reward System')
    plt.xlabel('Currency')
    plt.ylabel('Precision')
    plt.legend(title='Reward System')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_recall(df):
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df, x='currency', y='recall', hue='reward_system')
    plt.title('Recall per Currency and Reward System')
    plt.xlabel('Currency')
    plt.ylabel('Recall')
    plt.legend(title='Reward System')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_f1(df):
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df, x='currency', y='f1', hue='reward_system')
    plt.title('F1 Score per Currency and Reward System')
    plt.xlabel('Currency')
    plt.ylabel('F1 Score')
    plt.legend(title='Reward System')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(df, currency, threshold, window_size, reward_system):
    subset = df[
        (df['currency'] == currency) & 
        (df['threshold'] == threshold) & 
        (df['window_size'] == window_size) & 
        (df['reward_system'] == reward_system)
    ]

    if subset.empty:
        print(f"No data for currency: {currency}, threshold: {threshold}, window_size: {window_size}, reward_system: {reward_system}")
        return

    y_true = [0] * (subset['tn'].values[0] + subset['fp'].values[0]) + [1] * (subset['fn'].values[0] + subset['tp'].values[0])
    y_pred = [0] * subset['tn'].values[0] + [1] * subset['fp'].values[0] + [0] * subset['fn'].values[0] + [1] * subset['tp'].values[0]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {currency}, Threshold: {threshold}, Window Size: {window_size}, Reward System: {reward_system}')
    plt.show()

    

def plot_impact_of_threshold_on_precision(df):
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x='threshold', y='prec', hue='window_size', style='currency', markers=True, dashes=False)
    plt.title('Impact of Threshold Value on Precision for Different Window Sizes')
    plt.xlabel('Threshold')
    plt.ylabel('Precision')
    plt.legend(title='Window Size / Currency')
    plt.tight_layout()
    plt.show()

def plot_impact_of_window_size_on_precision(df):
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x='window_size', y='prec', hue='threshold', style='currency', markers=True, dashes=False)
    plt.title('Impact of Window Size on Precision for Different Threshold Values')
    plt.xlabel('Window Size (months)')
    plt.ylabel('Precision')
    plt.legend(title='Threshold / Currency')
    plt.tight_layout()
    plt.show()

def plot_impact_of_reward_system_on_anomaly_detection_rate(df):
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df, x='reward_system', y='ADRate', hue='currency')
    plt.title('Impact of Reward System on Anomaly Detection Rate')
    plt.xlabel('Reward System')
    plt.ylabel('Anomaly Detection Rate (%)')
    plt.legend(title='Currency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_impact_of_reward_system_on_false_alarm_rate(df):
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df, x='reward_system', y='FARate', hue='currency')
    plt.title('Impact of Reward System on False Alarm Rate')
    plt.xlabel('Reward System')
    plt.ylabel('False Alarm Rate (%)')
    plt.legend(title='Currency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def aggregate_performance_metrics(df):
    performance_summary = df.describe()
    return performance_summary

def identify_top_performers(df, metric='f1', top_n=3):
    top_performers = df.sort_values(by=metric, ascending=False).head(top_n)
    return top_performers

def visualize_distribution_of_metrics(df):
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df[['prec', 'recall', 'f1', 'ADRate', 'FARate']])
    plt.title('Distribution of Performance Metrics Across All Currencies')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def visualize_mean_performance_per_currency(df):
    mean_metrics = df.groupby('currency')[['prec', 'recall', 'f1', 'ADRate', 'FARate']].mean().reset_index()
    mean_metrics = pd.melt(mean_metrics, id_vars=['currency'], var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=mean_metrics, x='currency', y='Score', hue='Metric')
    plt.title('Mean Performance Metrics per Currency')
    plt.xlabel('Currency')
    plt.ylabel('Mean Score')
    plt.xticks(rotation=45)
    plt.legend(title='Metric')
    plt.tight_layout()
    plt.show()



def plot_anomalies(data, anomalies, threshold_percentile, window_size, reward_system):
    # Create time steps
    time_steps = list(range(len(data)))  # Convert range to list

    # Create a trace for the original time series data
    trace_data = go.Scatter(
        x = time_steps,
        y = data,
        mode = 'lines',
        name = 'Original Data',
        line = dict(color = 'blue'),
        opacity = 0.5,  
    )

    # Create a trace for the anomalies
    trace_anomalies = go.Scatter(
        x = time_steps,
        y = [data[i] if anomalies[i] else None for i in range(len(data))],
        mode = 'lines',  
        name = 'Anomalies',
        line = dict(color = 'red'),
    )

    # Define the layout
    layout = go.Layout(
        title = f'Anomaly Detection (Threshold: {threshold_percentile}, Window Size: {window_size}, Reward System: {reward_system})',
        xaxis = dict(title = 'Time Steps', dtick = 100, range = [0, len(data)]),  
        yaxis = dict(title = 'Data Value'),
    )

    # Create the figure and add the traces
    fig = go.Figure(data=[trace_data, trace_anomalies], layout=layout)

    # Show the figure
    pyo.plot(fig, filename='anomaly_detection.html')




def worker(shared_metrics_list, currency, threshold_percentile, window_size, train_data, test_data, reward_system='default', n_eval_episodes=10):
    try:
        print(f"Processing currency: {currency}")

        # Convert data to numeric
        train_data = train_data.apply(pd.to_numeric, errors='coerce')
        test_data = test_data.apply(pd.to_numeric, errors='coerce')

        # Flatten the data
        train_data_series = pd.Series(train_data.values.flatten())
        test_data_series = pd.Series(test_data.values.flatten())

        # Calculate the threshold
        threshold = np.percentile(train_data_series, threshold_percentile)
        print(f"Calculated threshold for {currency}: {threshold}")

        def create_env():
            env_instance = AnomalyDetectionEnv(train_data_series, window_size, threshold, reward_system=reward_system)
            env = Monitor(env_instance)
            return env

        # Create the environment
        env = DummyVecEnv([create_env])
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
       # print('Environment created successfully')

        # Extract LSTM features
        extractor = CustomLSTMFeaturesExtractor(env.observation_space, features_dim=256)
        lstm_output_size = extractor.get_features_dim()
       # print(f"LSTM output size: {lstm_output_size}")

        # Define policy kwargs
        policy_kwargs = {
            'vec_input_dim': 5,
            'features_dim': 256
        }

        # Initialize the PPO model
        model = CustomPPO(CustomLSTMPolicy, env, policy_kwargs=policy_kwargs, verbose=0, n_steps=256 * 10, learning_rate=0.005, batch_size=256, gamma=0.97,
                          gae_lambda=0.95, clip_range=0.1, ent_coef=0.01)

        # Train the model
        try:
            model.learn(total_timesteps=5_000)  # Adjust to a smaller number for quicker training
           # print('Model trained successfully')
        except Exception as e:
            print(f"Error in worker for currency {currency}: {e}")
            traceback.print_exc()

        # Create the evaluation environment
        eval_env_instance = AnomalyDetectionEnv(test_data_series, window_size, threshold, reward_system=reward_system)
        eval_env = DummyVecEnv([lambda: Monitor(eval_env_instance)])
       # print('Evaluation environment created successfully')

       

        # Evaluate the model
        for episode in range(n_eval_episodes):
             # Initialize metrics
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            true_negatives = 0
            episode_rewards = []
            anomalies_count = 0
            anomalies = []
            try:
                reset_result = eval_env.reset()
                if isinstance(reset_result, tuple) and len(reset_result) == 2:
                    obs, info = reset_result
                else:
                    obs, info = reset_result, {}

                done = False
                episode_reward = 0  
                episode_anomalies = []
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, info = eval_env.step(action)

                    actual_value = eval_env_instance.data.iloc[eval_env_instance.current_step]
                    actual_anomaly = abs(actual_value) > threshold  
                    predicted_anomaly = action == 1  
                    episode_reward += reward

                    # Track anomalies
                    episode_anomalies.append(actual_anomaly)

                    if predicted_anomaly and actual_anomaly:
                        anomalies_count += 1
                        true_positives += 1
                    elif predicted_anomaly and not actual_anomaly:
                        false_positives += 1
                    elif not predicted_anomaly and actual_anomaly:
                        false_negatives += 1
                    elif not predicted_anomaly and not actual_anomaly:
                        true_negatives += 1

                episode_rewards.append(episode_reward)  # Append episode reward
                anomalies.extend(episode_anomalies)

            except Exception as e:
                print(f"Error during episode {episode}: {e}")
                traceback.print_exc()
                continue

        valid_rewards = [reward for reward in episode_rewards if not np.isnan(reward)]
        mean_reward = np.mean(valid_rewards) if valid_rewards else np.nan
        std_reward = np.std(valid_rewards) if valid_rewards else np.nan

        print(f"mean_reward: {mean_reward}, std_reward: {std_reward}")

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        anomaly_detection_rate = (true_positives / (true_positives + false_negatives) * 100) if (true_positives + false_negatives) > 0 else 0
        false_alarm_rate = (false_positives / (false_positives + true_negatives) * 100) if (false_positives + true_negatives) > 0 else 0
        
        
        


        result = {
            'currency': currency,
            'threshold': threshold_percentile,
            'window_size': window_size,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'prec': precision,
            'recall': recall,
            'f1': f1_score,
            'tp': true_positives,
            'fp': false_positives,
            'fn': false_negatives,
            'tn': true_negatives,
            'ADRate': anomaly_detection_rate,
            'FARate': false_alarm_rate,
            'anomalies': anomalies_count,
            'reward_system': reward_system
        }

        shared_metrics_list.append(result)

        result_df = pd.DataFrame([result])
        print(result_df)

        shared_metrics_list.append(result)
        

        timestamps= test_data_series.index.tolist()

           
        
        anomalies = anomalies[:len(timestamps)]

        
        if len(anomalies) < len(timestamps):
            anomalies.extend([False] * (len(timestamps) - len(anomalies)))

       
        elif len(anomalies) > len(timestamps):
            anomalies = anomalies[:len(timestamps)]

        
        assert len(timestamps) == len(anomalies), f"Length mismatch: timestamps ({len(timestamps)}) and anomalies ({len(anomalies)})"

        
        plot_anomalies(test_data_series, anomalies, threshold_percentile, window_size, reward_system)
    
       

    except ValueError as e:
        print(f"Unpacking error for currency {currency}: {e}")
    except torch.cuda.OutOfMemoryError as e:
        print(f"OutOfMemoryError for currency {currency}: {e}")
    except Exception as e:
        print(f"Error in worker for currency {currency}: {e}")
        traceback.print_exc()

    return shared_metrics_list

    



def main():

    file_path = "C:\\Users\\roeun\\Desktop\\Mary\\ert_bil_eur_m_spreadsheet.xlsx"
    print(f"File path: {file_path}")
    try:

        # Load the sheets into separate dataframes
        df1 = load_and_preprocess_data(file_path, 'Sheet 1')
        df2 = load_and_preprocess_data(file_path, 'Sheet 2')

        # Print the head and tail of the dataframes
        print_dataframe_info(df1, 'df1')
        print_dataframe_info(df2, 'df2')

        # Merge the dataframes
        df = merge_dataframes(df1, df2)

        # Reorder the columns
        df = reorder_columns(df)
        print('Data after reorder')
        print(df.head())

        # Normalize the dataframe
        df = normalize_dataframe(df)

        # Print the preprocessed data
        print("\n preprocessed data/merged/normalized :")
        print(df.head())
        print(df.tail())

        # Create lagged features
        n_lags = 3
        df = create_lagged_features(df, n_lags)

        # Print the data after creating lagged features
        print(f"Data after creating lagged features:", df.tail())

        # Create a moving average
        df = create_moving_average(df, 3)

        # Perform feature engineering
        df = feature_engineering(df)

        # df = fourier_transform(df)
        # print("Fourier transformed signal:")
        #print(df.head())

        df = normalize_dataframe(df)
        print_df(df, "Data after normalization:")

        stat_features = statistical_features(df)
        print("Statistical features:")
        print(stat_features.head())


        # Rename the columns to remove '_x' and '_y'
        df.columns = df.columns.str.replace('_x', '')
        df.columns = df.columns.str.replace('_y', '')

        # Drop the 'index' column from df if it exists
        if 'index' in df.columns:
            df = df.drop(columns='index')

       
        # Determine the split point based on the number of columns in the DataFrame
        num_columns = df.shape[1]
        train_end = int(num_columns * 0.8)  
        # Split the data into training and testing sets
        X_train = df.iloc[:, :train_end]
        X_test = df.iloc[:, train_end:]
        print("Number of columns in training data:", X_train.shape[1])
        print("Number of columns in testing data:", X_test.shape[1])
        print('train data', X_train)
        print('test data', X_test)



        all_currencies = df.index.unique().tolist()
        num_currencies = 5
        selected_currencies = random.sample(all_currencies, num_currencies)
       
        
        print("Selected currencies: ", selected_currencies)

        threshold_percentiles = [80, 85, 90]
        window_sizes = [7, 14, 21]
        n_eval_episodes = 10
        reward_systems = ['default', 'alternative']


        with Manager() as manager:
            shared_metrics_list = manager.list()
            processes = []

            try:
                # Define the path for video recording
                video_path = "C:\\Users\\roeun\\Desktop\\Mary\\videos"

               
                if not os.path.exists(video_path):
                    os.makedirs(video_path)

                for reward_system in reward_systems:
                    for threshold_percentile in threshold_percentiles:
                        for window_size in window_sizes:
                            for currency in selected_currencies:
                                train_data = X_train[X_train.index == currency]
                                test_data = X_test[X_test.index == currency]
                              
                                p = Process(target=worker, args=(shared_metrics_list, currency, threshold_percentile, window_size, train_data, test_data,reward_system, n_eval_episodes))
                                p.start()
                                processes.append(p)

                # After all processes have finished
                for p in processes:
                    p.join()

                # Convert the shared list to a DataFrame
                result_df = pd.DataFrame(list(shared_metrics_list))


                # Convert the shared list to a DataFrame
                result_df = pd.DataFrame(list(shared_metrics_list))

                # Print the DataFrame
                print(result_df)

                # Calculate additional metrics
                calculate_metrics(result_df)

                # Aggregated performance metrics
                performance_summary = aggregate_performance_metrics(result_df)
                print("Performance Summary:")
                print(performance_summary)

               
                # Identify top performers based on the calculated score
                top_performers = result_df.nlargest(5, 'prec').drop_duplicates(subset=['currency', 'threshold', 'window_size'])

                print("\nTop Performers :")
                print(top_performers)

                # Visualize distribution of metrics
                visualize_distribution_of_metrics(result_df)

                # Visualize mean performance per currency
                visualize_mean_performance_per_currency(result_df)

                # Plot for top 3 performers
                for _, row in top_performers.iterrows():
                    plot_confusion_matrix(result_df, row['currency'], row['threshold'], row['window_size'], row['reward_system'])

                # Call plotting functions for overall analysis
                plot_anomaly_detection_rate(result_df)
                plot_false_alarm_rate(result_df)
                plot_impact_of_threshold_on_anomaly_detection_rate(result_df)
                plot_impact_of_window_size_on_anomaly_detection_rate(result_df)
                plot_mean_reward(result_df)
                plot_precision(result_df)
                plot_recall(result_df)
                plot_f1(result_df)
                plot_impact_of_threshold_on_precision(result_df)
                plot_impact_of_window_size_on_precision(result_df)
                plot_impact_of_reward_system_on_anomaly_detection_rate(result_df)
                plot_impact_of_reward_system_on_false_alarm_rate(result_df)

                
                



            except Exception as e:
                print(f"An error occurred: {e}")
                for  p in processes: 
                     p.terminate()
                     p.join()

    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
